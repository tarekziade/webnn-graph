// Main ONNX to WebNN conversion logic

use crate::ast::{DataType, GraphJson};
use crate::onnx::types::TypeConversionError;
use onnx::onnx::ModelProto;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OnnxError {
    #[error("failed to read ONNX file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("failed to parse ONNX protobuf: {0}")]
    ProtobufError(String),

    #[error("unsupported ONNX opset version: {0}")]
    UnsupportedOpset(i64),

    #[error("unsupported operator: {op} (node: {node})")]
    UnsupportedOp { op: String, node: String },

    #[error("missing required attribute: {attr} in {op}")]
    MissingAttribute { attr: String, op: String },

    #[error("invalid tensor shape: {0}")]
    InvalidShape(String),

    #[error("type conversion error: {0}")]
    TypeConversion(#[from] TypeConversionError),

    #[error("shape inference failed for node: {0}")]
    ShapeInference(String),
}

/// Sanitize ONNX identifiers for WebNN DSL compatibility
/// Replaces problematic characters that would confuse the parser
pub fn sanitize_identifier(name: &str) -> String {
    name.replace("::", "__").replace(':', "_")
}

/// Conversion options for ONNX to WebNN
#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// Extract weights to external file (default: true)
    pub extract_weights: bool,
    /// Output file path for graph (.webnn or .json)
    pub output_path: String,
    /// Weights file path (.weights)
    pub weights_path: Option<String>,
    /// Manifest file path (.manifest.json)
    pub manifest_path: Option<String>,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            extract_weights: true,
            output_path: "output.webnn".to_string(),
            weights_path: Some("output.weights".to_string()),
            manifest_path: Some("output.manifest.json".to_string()),
        }
    }
}

struct TensorInfo {
    _data_type: DataType,
    _shape: Vec<i64>,
}

/// Main converter structure
pub struct OnnxConverter {
    model: ModelProto,
    graph: GraphJson,
    _value_info: HashMap<String, TensorInfo>,
}

impl OnnxConverter {
    /// Create a new converter from an ONNX model
    pub fn new(model: ModelProto) -> Result<Self, OnnxError> {
        let graph_name = if model.has_graph() {
            let graph = model.get_graph();
            if graph.has_name() {
                graph.get_name().to_string()
            } else {
                "graph".to_string()
            }
        } else {
            "graph".to_string()
        };

        let graph = GraphJson {
            format: "webnn-graph-json".to_string(),
            version: 1,
            name: Some(graph_name),
            inputs: BTreeMap::new(),
            consts: BTreeMap::new(),
            nodes: Vec::new(),
            outputs: BTreeMap::new(),
        };

        Ok(Self {
            model,
            graph,
            _value_info: HashMap::new(),
        })
    }

    /// Extract metadata from ONNX model
    pub fn extract_metadata(&self) -> Result<(), OnnxError> {
        if !self.model.has_graph() {
            return Err(OnnxError::ProtobufError(
                "Missing graph in model".to_string(),
            ));
        }

        let graph = self.model.get_graph();

        // Print basic info
        println!("Model name: {}", self.graph.name.as_ref().unwrap());
        println!("Inputs: {}", graph.get_input().len());
        println!("Outputs: {}", graph.get_output().len());
        println!("Nodes: {}", graph.get_node().len());
        println!("Initializers: {}", graph.get_initializer().len());

        Ok(())
    }

    /// Convert ONNX model to GraphJson
    pub fn convert(mut self, options: &ConvertOptions) -> Result<GraphJson, OnnxError> {
        if !self.model.has_graph() {
            return Err(OnnxError::ProtobufError(
                "Missing graph in model".to_string(),
            ));
        }

        let onnx_graph = self.model.get_graph();

        // Process inputs (exclude initializers)
        let initializer_names: std::collections::HashSet<String> = onnx_graph
            .get_initializer()
            .iter()
            .map(|init| init.get_name().to_string())
            .collect();

        for input in onnx_graph.get_input() {
            let raw_name = input.get_name().to_string();
            let name = sanitize_identifier(&raw_name);

            // Skip if this is an initializer (constant)
            if initializer_names.contains(&raw_name) {
                continue;
            }

            // Get type info
            if input.has_field_type() {
                let type_proto = input.get_field_type();
                if type_proto.has_tensor_type() {
                    let tensor_type = type_proto.get_tensor_type();

                    let data_type = if tensor_type.has_elem_type() {
                        let onnx_type = tensor_type.get_elem_type() as i32;
                        crate::onnx::types::map_onnx_data_type(onnx_type)?
                    } else {
                        DataType::Float32 // Default
                    };

                    let shape = if tensor_type.has_shape() {
                        let shape_proto = tensor_type.get_shape();
                        shape_proto
                            .get_dim()
                            .iter()
                            .map(|dim| {
                                if dim.has_dim_value() {
                                    dim.get_dim_value() as u32
                                } else {
                                    0 // Dynamic dimension
                                }
                            })
                            .collect()
                    } else {
                        vec![]
                    };

                    self.graph
                        .inputs
                        .insert(name, crate::ast::OperandDesc { data_type, shape });
                }
            }
        }

        // Process initializers (constants/weights)
        for initializer in onnx_graph.get_initializer() {
            let name = sanitize_identifier(initializer.get_name());
            let onnx_type = initializer.get_data_type() as i32;
            let data_type = crate::onnx::types::map_onnx_data_type(onnx_type)?;
            let shape: Vec<u32> = initializer.get_dims().iter().map(|d| *d as u32).collect();

            let init = if options.extract_weights {
                // External weights reference (use original name for weights file)
                crate::ast::ConstInit::Weights {
                    r#ref: sanitize_identifier(initializer.get_name()),
                }
            } else {
                // Inline bytes
                let bytes = initializer.get_raw_data().to_vec();
                crate::ast::ConstInit::InlineBytes { bytes }
            };

            self.graph.consts.insert(
                name,
                crate::ast::ConstDecl {
                    data_type,
                    shape,
                    init,
                },
            );
        }

        // Process nodes using OpRegistry
        let registry = crate::onnx::ops::OpRegistry::new();
        let context = crate::onnx::ops::ConversionContext {};

        for onnx_node in onnx_graph.get_node() {
            let converted_nodes = registry.convert_node(onnx_node, &context)?;
            self.graph.nodes.extend(converted_nodes);
        }

        // Process outputs
        for output in onnx_graph.get_output() {
            let name = sanitize_identifier(output.get_name());
            // In WebNN, outputs map output names to the node result names
            self.graph.outputs.insert(name.clone(), name);
        }

        Ok(self.graph)
    }
}

/// Convert an ONNX file to WebNN format with optional weight extraction
pub fn convert_onnx<P: AsRef<Path>>(
    onnx_path: P,
    options: ConvertOptions,
) -> Result<GraphJson, OnnxError> {
    // Read ONNX file
    let onnx_bytes = fs::read(onnx_path.as_ref())?;

    // Parse protobuf
    let model: ModelProto = protobuf::parse_from_bytes(&onnx_bytes)
        .map_err(|e| OnnxError::ProtobufError(e.to_string()))?;

    // Create converter
    let converter = OnnxConverter::new(model.clone())?;

    // Extract metadata for debugging
    converter.extract_metadata()?;

    // Convert to GraphJson
    let graph = converter.convert(&options)?;

    // Extract weights if requested
    if options.extract_weights {
        if let (Some(weights_path), Some(manifest_path)) =
            (&options.weights_path, &options.manifest_path)
        {
            extract_weights_from_onnx(&model, weights_path, manifest_path)?;
        }
    }

    Ok(graph)
}

/// Extract weights from ONNX model to .weights and .manifest.json files
fn extract_weights_from_onnx(
    model: &ModelProto,
    weights_path: &str,
    manifest_path: &str,
) -> Result<(), OnnxError> {
    use crate::weights::{TensorEntry, WeightsManifest};

    if !model.has_graph() {
        return Err(OnnxError::ProtobufError(
            "Missing graph in model".to_string(),
        ));
    }

    let onnx_graph = model.get_graph();
    let mut manifest = WeightsManifest {
        format: "wg-weights-manifest".to_string(),
        version: 1,
        endianness: "little".to_string(),
        tensors: BTreeMap::new(),
    };

    let mut weights_data = Vec::new();
    let mut current_offset = 0u64;

    // Process each initializer
    for initializer in onnx_graph.get_initializer() {
        let name = sanitize_identifier(initializer.get_name());

        // Convert ONNX data type enum to i32, then to WebNN DataType
        let onnx_type = initializer.get_data_type() as i32;
        let data_type = crate::onnx::types::map_onnx_data_type(onnx_type)?;

        let shape: Vec<u32> = initializer.get_dims().iter().map(|d| *d as u32).collect();
        let raw_data = initializer.get_raw_data();
        let byte_length = raw_data.len() as u64;

        // Add to manifest
        manifest.tensors.insert(
            name,
            TensorEntry {
                data_type,
                shape,
                byte_offset: current_offset,
                byte_length,
                layout: None,
            },
        );

        // Append to weights data
        weights_data.extend_from_slice(raw_data);
        current_offset += byte_length;
    }

    // Write weights file
    fs::write(weights_path, &weights_data)?;

    // Write manifest file
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| OnnxError::ProtobufError(e.to_string()))?;
    fs::write(manifest_path, manifest_json)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_options_default() {
        let options = ConvertOptions::default();
        assert!(options.extract_weights);
        assert_eq!(options.output_path, "output.webnn");
    }
}
