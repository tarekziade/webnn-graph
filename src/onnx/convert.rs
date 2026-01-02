// Main ONNX to WebNN conversion logic

use crate::ast::{DataType, GraphJson};
use crate::onnx::types::TypeConversionError;
use onnx::onnx::{ModelProto, TensorProto, TensorProto_DataType};
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;
use thiserror::Error;

const MIN_SUPPORTED_OPSET: i64 = 11;
const MAX_SUPPORTED_OPSET: i64 = 18;

#[derive(Debug, Error)]
pub enum OnnxError {
    #[error("failed to read ONNX file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("failed to parse ONNX protobuf: {0}")]
    ProtobufError(String),

    #[error("unsupported ONNX opset version {version} for domain '{domain}'")]
    UnsupportedOpset { domain: String, version: i64 },

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

/// Infer output shape for an ONNX node based on its operation type and inputs
fn infer_shape(
    node: &onnx::onnx::NodeProto,
    value_shapes: &HashMap<String, Vec<i64>>,
    initializers: &HashMap<String, &TensorProto>,
    const_values: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    let op = node.get_op_type();

    match op {
        // Unary operations that preserve shape
        "Cast" | "Relu" | "Tanh" | "Sigmoid" | "Erf" | "Softmax" | "Gelu" | "Exp" | "Log"
        | "Abs" | "Neg" | "Sqrt" | "LayerNormalization" => {
            let ins = node.get_input();
            if ins.is_empty() {
                return None;
            }
            value_shapes.get(ins[0].as_str()).cloned()
        }

        // Binary operations (with broadcasting) - prefer shape with more dimensions
        // This is a simplification; proper implementation would handle broadcasting rules
        "Add" | "Sub" | "Mul" | "Div" | "Pow" => {
            let ins = node.get_input();
            if ins.len() < 2 {
                return None;
            }

            let shape_a = value_shapes.get(ins[0].as_str());
            let shape_b = value_shapes.get(ins[1].as_str());

            match (shape_a, shape_b) {
                (Some(a), Some(b)) => {
                    // Prefer the shape with more dimensions (likely the activation tensor)
                    if a.len() >= b.len() {
                        Some(a.clone())
                    } else {
                        Some(b.clone())
                    }
                }
                (Some(a), None) => Some(a.clone()),
                (None, Some(b)) => Some(b.clone()),
                (None, None) => None,
            }
        }

        // MatMul (2D matrix multiplication)
        "MatMul" => {
            let ins = node.get_input();
            if ins.len() < 2 {
                return None;
            }

            let a_shape = value_shapes.get(ins[0].as_str())?;
            let b_shape = value_shapes.get(ins[1].as_str())?;

            // Handle 2D case: [M, K] @ [K, N] -> [M, N]
            if a_shape.len() >= 2 && b_shape.len() >= 2 {
                let m = a_shape[a_shape.len() - 2];
                let n = b_shape[b_shape.len() - 1];

                // For higher-dim inputs, preserve batch dimensions
                if a_shape.len() == 2 && b_shape.len() == 2 {
                    return Some(vec![m, n]);
                } else if a_shape.len() > 2 {
                    let mut result = a_shape[..a_shape.len() - 2].to_vec();
                    result.push(m);
                    result.push(n);
                    return Some(result);
                }
            }
            None
        }

        // Transpose preserves shape with permuted dimensions
        "Transpose" => {
            let ins = node.get_input();
            if ins.is_empty() {
                return None;
            }
            let input_shape = value_shapes.get(ins[0].as_str())?;

            // Get perm attribute
            let perm: Vec<usize> = node
                .get_attribute()
                .iter()
                .find(|a| a.get_name() == "perm")
                .map(|a| a.get_ints().iter().map(|&i| i as usize).collect())
                .unwrap_or_else(|| (0..input_shape.len()).rev().collect());

            // Apply permutation
            Some(perm.iter().map(|&i| input_shape[i]).collect())
        }

        // Reduce operations
        "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" => {
            let ins = node.get_input();
            if ins.is_empty() {
                return None;
            }
            let input_shape = value_shapes.get(ins[0].as_str())?;

            // Check keepdims attribute (default is 1/true)
            let keepdims = node
                .get_attribute()
                .iter()
                .find(|a| a.get_name() == "keepdims")
                .and_then(|a| {
                    if a.has_i() {
                        Some(a.get_i() != 0)
                    } else {
                        None
                    }
                })
                .unwrap_or(true);

            // Get axes attribute
            let axes: Vec<i64> = node
                .get_attribute()
                .iter()
                .find(|a| a.get_name() == "axes")
                .map(|a| a.get_ints().to_vec())
                .unwrap_or_default();

            if axes.is_empty() {
                // Reduce all dimensions
                if keepdims {
                    Some(vec![1; input_shape.len()])
                } else {
                    Some(vec![])
                }
            } else {
                // Reduce specific axes
                let mut output_shape = input_shape.clone();
                for &axis in &axes {
                    let idx = if axis < 0 {
                        (input_shape.len() as i64 + axis) as usize
                    } else {
                        axis as usize
                    };
                    if idx < output_shape.len() {
                        if keepdims {
                            output_shape[idx] = 1;
                        } else {
                            output_shape[idx] = -1; // Mark for removal
                        }
                    }
                }
                if !keepdims {
                    output_shape.retain(|&d| d != -1);
                }
                Some(output_shape)
            }
        }

        // Gemm (generalized matrix multiplication)
        "Gemm" => {
            let ins = node.get_input();
            if ins.len() < 2 {
                return None;
            }

            let a_shape = value_shapes.get(ins[0].as_str())?;
            let b_shape = value_shapes.get(ins[1].as_str())?;

            if a_shape.len() != 2 || b_shape.len() != 2 {
                return None;
            }

            // Check transA and transB attributes
            let trans_a = node
                .get_attribute()
                .iter()
                .find(|a| a.get_name() == "transA")
                .and_then(|a| {
                    if a.has_i() {
                        Some(a.get_i() != 0)
                    } else {
                        None
                    }
                })
                .unwrap_or(false);

            let trans_b = node
                .get_attribute()
                .iter()
                .find(|a| a.get_name() == "transB")
                .and_then(|a| {
                    if a.has_i() {
                        Some(a.get_i() != 0)
                    } else {
                        None
                    }
                })
                .unwrap_or(false);

            let m = if trans_a { a_shape[1] } else { a_shape[0] };
            let n = if trans_b { b_shape[0] } else { b_shape[1] };

            Some(vec![m, n])
        }

        "Gather" => {
            let ins = node.get_input();
            if ins.len() < 2 {
                return None;
            }

            let data_shape = value_shapes.get(ins[0].as_str())?;
            let indices_shape = value_shapes.get(ins[1].as_str())?;

            let mut axis = node
                .get_attribute()
                .iter()
                .find(|a| a.get_name() == "axis")
                .and_then(|a| a.has_i().then(|| a.get_i()))
                .unwrap_or(0);

            if axis < 0 {
                axis += data_shape.len() as i64;
            }

            let axis_usize = axis as usize;
            if axis_usize > data_shape.len() {
                return None;
            }

            let mut output = Vec::new();
            output.extend_from_slice(&data_shape[..axis_usize]);
            output.extend(indices_shape.iter().cloned());
            if axis_usize < data_shape.len() {
                output.extend_from_slice(&data_shape[axis_usize + 1..]);
            }
            Some(output)
        }

        "Unsqueeze" => {
            let ins = node.get_input();
            if ins.is_empty() {
                return None;
            }

            let input_shape = value_shapes.get(ins[0].as_str())?.clone();
            let mut axes: Vec<i64> = node
                .get_attribute()
                .iter()
                .find(|a| a.get_name() == "axes")
                .map(|a| a.get_ints().to_vec())
                .unwrap_or_default();

            if axes.is_empty() {
                return None;
            }

            axes.sort();
            let mut output_shape = input_shape;
            for axis in axes {
                let idx = if axis < 0 {
                    (output_shape.len() as i64 + axis + 1) as usize
                } else {
                    axis as usize
                };
                if idx <= output_shape.len() {
                    output_shape.insert(idx, 1);
                }
            }
            Some(output_shape)
        }

        "Concat" => {
            let mut shapes = Vec::new();
            for inp in node.get_input() {
                let shape = value_shapes.get(inp.as_str())?;
                shapes.push(shape.clone());
            }

            if shapes.is_empty() {
                return None;
            }

            let mut axis = node
                .get_attribute()
                .iter()
                .find(|a| a.get_name() == "axis")
                .and_then(|a| a.has_i().then(|| a.get_i()))
                .unwrap_or(0);

            if axis < 0 {
                axis += shapes[0].len() as i64;
            }
            let axis_usize = axis as usize;

            let mut output = shapes[0].clone();
            for shape in shapes.iter().skip(1) {
                if shape.len() != output.len() || axis_usize >= shape.len() {
                    return None;
                }
                output[axis_usize] += shape[axis_usize];
            }
            Some(output)
        }

        "Reshape" => {
            let ins = node.get_input();
            if ins.len() < 2 {
                return None;
            }

            let input_shape = value_shapes.get(ins[0].as_str())?;
            let shape_input = ins[1].as_str();
            let mut target: Vec<i64> = if let Some(values) = const_values.get(shape_input) {
                values.clone()
            } else if let Some(shape_tensor) = initializers.get(shape_input) {
                if !shape_tensor.get_raw_data().is_empty() {
                    if shape_tensor.get_data_type() == TensorProto_DataType::INT32 {
                        shape_tensor
                            .get_raw_data()
                            .chunks_exact(4)
                            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                            .collect()
                    } else {
                        shape_tensor
                            .get_raw_data()
                            .chunks_exact(8)
                            .map(|c| {
                                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                            })
                            .collect()
                    }
                } else if !shape_tensor.get_int64_data().is_empty() {
                    shape_tensor.get_int64_data().to_vec()
                } else if !shape_tensor.get_int32_data().is_empty() {
                    shape_tensor
                        .get_int32_data()
                        .iter()
                        .map(|&v| v as i64)
                        .collect()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            if target.is_empty() {
                return None;
            }

            if target.contains(&-1) {
                let total_input: i64 = input_shape.iter().product();
                let known: i64 = target.iter().filter(|&&d| d != -1).product();
                if known == 0 || total_input % known != 0 {
                    return None;
                }
                if let Some(idx) = target.iter().position(|&d| d == -1) {
                    target[idx] = total_input / known;
                }
            }

            Some(target)
        }

        "Slice" => {
            let ins = node.get_input();
            if ins.is_empty() {
                return None;
            }

            let input_shape = value_shapes.get(ins[0].as_str())?;

            let read_ints = |name: Option<&String>| -> Option<Vec<i64>> {
                if let Some(n) = name {
                    if let Some(v) = const_values.get(n) {
                        return Some(v.clone());
                    }
                    if let Some(t) = initializers.get(n) {
                        let raw = t.get_raw_data();
                        if !raw.is_empty() {
                            if t.get_data_type() == TensorProto_DataType::INT32 {
                                return Some(
                                    raw.chunks_exact(4)
                                        .map(|c| {
                                            i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64
                                        })
                                        .collect(),
                                );
                            } else {
                                return Some(
                                    raw.chunks_exact(8)
                                        .map(|c| {
                                            i64::from_le_bytes([
                                                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                                            ])
                                        })
                                        .collect(),
                                );
                            }
                        } else if !t.get_int64_data().is_empty() {
                            return Some(t.get_int64_data().to_vec());
                        } else if !t.get_int32_data().is_empty() {
                            return Some(t.get_int32_data().iter().map(|&v| v as i64).collect());
                        }
                    }
                }
                None
            };

            let starts = read_ints(ins.get(1))?;
            let ends = read_ints(ins.get(2))?;
            let axes =
                read_ints(ins.get(3)).unwrap_or_else(|| (0..input_shape.len() as i64).collect());
            let steps = read_ints(ins.get(4)).unwrap_or_else(|| vec![1; axes.len()]);

            if axes.len() != starts.len() || axes.len() != ends.len() || axes.len() != steps.len() {
                return None;
            }

            let mut output = input_shape.clone();
            for i in 0..axes.len() {
                let axis = if axes[i] < 0 {
                    (input_shape.len() as i64 + axes[i]) as usize
                } else {
                    axes[i] as usize
                };
                if axis >= output.len() {
                    return None;
                }

                let step = steps[i];
                if step != 1 {
                    return None;
                }

                let dim = input_shape[axis];
                let mut start = starts[i];
                let mut end = ends[i];

                if start < 0 {
                    start += dim;
                }
                if end < 0 {
                    end += dim;
                }

                start = start.max(0);
                end = end.min(dim);

                if end < start {
                    output[axis] = 0;
                } else {
                    output[axis] = end - start;
                }
            }

            Some(output)
        }

        _ => None,
    }
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
    /// Override dynamic dimension values (e.g., batch_size=1, sequence_length=128)
    pub free_dim_overrides: HashMap<String, u32>,
    /// Enable constant folding and shape propagation optimizations
    pub optimize: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            extract_weights: true,
            output_path: "output.webnn".to_string(),
            weights_path: Some("output.weights".to_string()),
            manifest_path: Some("output.manifest.json".to_string()),
            free_dim_overrides: HashMap::new(),
            optimize: false,
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

        // Validate opset imports
        for import in self.model.get_opset_import() {
            let domain = import.get_domain();
            let version = import.get_version();
            let domain_name = if domain.is_empty() {
                "ai.onnx".to_string()
            } else {
                domain.to_string()
            };

            if (domain.is_empty() || domain == "ai.onnx")
                && !(MIN_SUPPORTED_OPSET..=MAX_SUPPORTED_OPSET).contains(&version)
            {
                return Err(OnnxError::UnsupportedOpset {
                    domain: domain_name,
                    version,
                });
            }
        }

        let onnx_graph = self.model.get_graph();
        let mut value_name_map: HashMap<String, String> = HashMap::new();
        let mut effective_overrides = options.free_dim_overrides.clone();
        let mut value_types: HashMap<String, DataType> = HashMap::new();

        // Merge overrides from model metadata if present
        for meta in self.model.get_metadata_props() {
            if meta
                .get_key()
                .eq_ignore_ascii_case("freedimensionoverrides")
            {
                if let Ok(json) = serde_json::from_str::<JsonValue>(meta.get_value()) {
                    let obj = json
                        .get("freeDimensionOverrides")
                        .unwrap_or(&json)
                        .as_object()
                        .cloned();
                    if let Some(map) = obj {
                        for (name, value) in map {
                            if let Some(v) = value.as_u64() {
                                effective_overrides.entry(name.clone()).or_insert(v as u32);
                            }
                        }
                    }
                }
            }
        }

        // Process inputs (exclude initializers)
        let initializer_names: HashSet<String> = onnx_graph
            .get_initializer()
            .iter()
            .map(|init| init.get_name().to_string())
            .collect();

        let default_dim_values: HashMap<&str, u32> = HashMap::from([
            ("batch_size", 1),
            ("batch", 1),
            ("n", 1),
            ("b", 1),
            ("sequence_length", 128),
            ("seq_len", 128),
            ("seq", 128),
            ("s", 128),
            ("t", 128),
        ]);

        let mut auto_applied: Vec<(String, u32)> = Vec::new();
        let mut missing_dims: Vec<(String, String)> = Vec::new();

        let mut resolve_dim_override =
            |dim_param: &str, overrides: &mut HashMap<String, u32>| -> Option<u32> {
                if let Some(v) = overrides.get(dim_param) {
                    return Some(*v);
                }

                let lower = dim_param.to_ascii_lowercase();
                if let Some(v) = overrides.get(&lower) {
                    return Some(*v);
                }

                if let Some(default) = default_dim_values.get(lower.as_str()) {
                    overrides.insert(dim_param.to_string(), *default);
                    auto_applied.push((dim_param.to_string(), *default));
                    return Some(*default);
                }

                None
            };

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
                        let mut resolved = Vec::new();
                        for dim in shape_proto.get_dim() {
                            if dim.has_dim_value() {
                                resolved.push(dim.get_dim_value() as u32);
                            } else if dim.has_dim_param() {
                                let dim_param = dim.get_dim_param();
                                if let Some(v) =
                                    resolve_dim_override(dim_param, &mut effective_overrides)
                                {
                                    resolved.push(v);
                                } else {
                                    missing_dims.push((raw_name.clone(), dim_param.to_string()));
                                    resolved.clear();
                                    break;
                                }
                            } else {
                                missing_dims.push((raw_name.clone(), "<unknown>".to_string()));
                                resolved.clear();
                                break;
                            }
                        }
                        resolved
                    } else {
                        return Err(OnnxError::InvalidShape(format!(
                            "Input '{}' is missing shape information",
                            raw_name
                        )));
                    };

                    if shape.is_empty() {
                        continue;
                    }

                    self.graph.inputs.insert(
                        name.clone(),
                        crate::ast::OperandDesc {
                            data_type: data_type.clone(),
                            shape,
                        },
                    );

                    value_name_map.insert(raw_name.clone(), name.clone());
                    value_name_map.insert(name.clone(), name.clone());
                    value_types.insert(raw_name.clone(), data_type.clone());
                    value_types.insert(name.clone(), data_type);
                }
            }
        }

        if !missing_dims.is_empty() {
            let mut message = "Dynamic dimensions require explicit overrides:\n".to_string();
            for (input, dim) in &missing_dims {
                message.push_str(&format!(
                    " - input '{}' dim '{}': --override-dim {}=<value>\n",
                    input, dim, dim
                ));
            }
            if !auto_applied.is_empty() {
                message.push_str("Auto-applied defaults: ");
                for (idx, (name, value)) in auto_applied.iter().enumerate() {
                    if idx > 0 {
                        message.push_str(", ");
                    }
                    message.push_str(&format!("{}={}", name, value));
                }
                message.push('\n');
            }
            return Err(OnnxError::InvalidShape(message));
        }

        // Process initializers (constants/weights)
        for initializer in onnx_graph.get_initializer() {
            let name = sanitize_identifier(initializer.get_name());
            let raw_data = initializer.get_raw_data();

            // Skip initializers with no data (check both raw_data and typed data fields)
            let has_data = !raw_data.is_empty()
                || !initializer.get_float_data().is_empty()
                || !initializer.get_int32_data().is_empty()
                || !initializer.get_int64_data().is_empty()
                || !initializer.get_double_data().is_empty();

            if !has_data {
                eprintln!("Warning: Skipping initializer '{}' with no data", name);
                continue;
            }

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
                let bytes = raw_data.to_vec();
                crate::ast::ConstInit::InlineBytes { bytes }
            };

            self.graph
                .consts
                .entry(name.clone())
                .or_insert(crate::ast::ConstDecl {
                    data_type: data_type.clone(),
                    shape,
                    init,
                });

            value_name_map.insert(initializer.get_name().to_string(), name.clone());
            value_name_map.insert(name.clone(), name.clone());
            value_types.insert(initializer.get_name().to_string(), data_type.clone());
            value_types.insert(name, data_type);
        }

        // Process nodes using OpRegistry
        let registry = crate::onnx::ops::OpRegistry::new();

        // Build initializers map for resolving constant shapes
        let mut initializers_map = std::collections::HashMap::new();
        for initializer in onnx_graph.get_initializer() {
            // Skip initializers with no data (check both raw_data and typed data fields)
            let has_data = !initializer.get_raw_data().is_empty()
                || !initializer.get_float_data().is_empty()
                || !initializer.get_int32_data().is_empty()
                || !initializer.get_int64_data().is_empty()
                || !initializer.get_double_data().is_empty();

            if !has_data {
                continue;
            }
            initializers_map.insert(initializer.get_name().to_string(), initializer);
        }

        // Build value_shapes map from value_info and inputs for shape inference
        let mut value_shapes = std::collections::HashMap::new();

        // Add input shapes (already validated)
        for (raw_name, mapped_name) in value_name_map.clone() {
            if initializer_names.contains(&raw_name) {
                continue;
            }
            if let Some(input) = onnx_graph
                .get_input()
                .iter()
                .find(|i| i.get_name() == raw_name)
            {
                if input.has_field_type() && input.get_field_type().has_tensor_type() {
                    let type_proto = input.get_field_type().get_tensor_type();
                    if type_proto.has_shape() {
                        let mut shape: Vec<i64> = Vec::new();
                        for dim in type_proto.get_shape().get_dim() {
                            if dim.has_dim_value() {
                                shape.push(dim.get_dim_value());
                            } else if dim.has_dim_param() {
                                if let Some(v) = resolve_dim_override(
                                    dim.get_dim_param(),
                                    &mut effective_overrides,
                                ) {
                                    shape.push(v as i64);
                                }
                            }
                        }
                        if !shape.is_empty() {
                            value_shapes.insert(raw_name.clone(), shape.clone());
                            value_shapes.insert(mapped_name.clone(), shape);
                        }
                    }
                }
            }
        }

        // Add initializer shapes
        for initializer in onnx_graph.get_initializer() {
            // Skip initializers with no data (check both raw_data and typed data fields)
            let has_data = !initializer.get_raw_data().is_empty()
                || !initializer.get_float_data().is_empty()
                || !initializer.get_int32_data().is_empty()
                || !initializer.get_int64_data().is_empty()
                || !initializer.get_double_data().is_empty();

            if !has_data {
                continue;
            }
            let shape: Vec<i64> = initializer.get_dims().to_vec();
            value_shapes.insert(initializer.get_name().to_string(), shape);
        }

        // Add value_info shapes (intermediate tensors from shape inference)
        // Try to resolve dynamic dimensions using overrides
        for value_info in onnx_graph.get_value_info() {
            if value_info.has_field_type() && value_info.get_field_type().has_tensor_type() {
                let type_proto = value_info.get_field_type().get_tensor_type();
                if type_proto.has_shape() {
                    let mut shape: Vec<i64> = Vec::new();
                    let mut unknown = false;

                    for d in type_proto.get_shape().get_dim() {
                        if d.has_dim_value() {
                            shape.push(d.get_dim_value());
                        } else if d.has_dim_param() {
                            if let Some(v) =
                                resolve_dim_override(d.get_dim_param(), &mut effective_overrides)
                            {
                                shape.push(v as i64);
                            } else {
                                unknown = true;
                                break;
                            }
                        } else {
                            unknown = true;
                            break;
                        }
                    }

                    if !unknown && !shape.is_empty() && shape.iter().all(|&d| d > 0) {
                        value_shapes.insert(value_info.get_name().to_string(), shape);
                    }
                }
            }
        }

        // Seed const values with integer initializers and Constant nodes
        let mut const_values: HashMap<String, Vec<i64>> = HashMap::new();
        for (name, initializer) in &initializers_map {
            if initializer.get_data_type() == TensorProto_DataType::INT64
                || initializer.get_data_type() == TensorProto_DataType::INT32
            {
                let raw = initializer.get_raw_data();
                let values = if !raw.is_empty() {
                    if initializer.get_data_type() == TensorProto_DataType::INT32 {
                        raw.chunks_exact(4)
                            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                            .collect()
                    } else {
                        raw.chunks_exact(8)
                            .map(|c| {
                                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                            })
                            .collect()
                    }
                } else if !initializer.get_int64_data().is_empty() {
                    initializer.get_int64_data().to_vec()
                } else if !initializer.get_int32_data().is_empty() {
                    initializer
                        .get_int32_data()
                        .iter()
                        .map(|&v| v as i64)
                        .collect()
                } else {
                    Vec::new()
                };

                if !values.is_empty() {
                    const_values.insert(name.clone(), values);
                }
            }
        }

        for node in onnx_graph.get_node() {
            if node.get_op_type() == "Constant" {
                if let Some(attr) = node
                    .get_attribute()
                    .iter()
                    .find(|a| a.get_name() == "value" && a.has_t())
                {
                    let tensor = attr.get_t();
                    if tensor.get_data_type() == TensorProto_DataType::INT64
                        || tensor.get_data_type() == TensorProto_DataType::INT32
                    {
                        let raw = tensor.get_raw_data();
                        let values = if !raw.is_empty() {
                            if tensor.get_data_type() == TensorProto_DataType::INT32 {
                                raw.chunks_exact(4)
                                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                                    .collect()
                            } else {
                                raw.chunks_exact(8)
                                    .map(|c| {
                                        i64::from_le_bytes([
                                            c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                                        ])
                                    })
                                    .collect()
                            }
                        } else if !tensor.get_int64_data().is_empty() {
                            tensor.get_int64_data().to_vec()
                        } else if !tensor.get_int32_data().is_empty() {
                            tensor.get_int32_data().iter().map(|&v| v as i64).collect()
                        } else {
                            Vec::new()
                        };

                        if let Some(out) = node.get_output().first() {
                            if !values.is_empty() {
                                const_values.insert(out.to_string(), values);
                                value_types.insert(out.to_string(), DataType::Int64);
                            }
                        }
                    }
                }
            }
        }

        // Run the static shape/type inference scaffold to seed shapes/types/constants
        // before lowering. Errors surface early if dynamic dims remain.
        let inferred =
            crate::onnx::shape_inference::infer_static_shapes(&self.model, &effective_overrides)
                .map_err(|e| OnnxError::ShapeInference(e.to_string()))?;

        for (k, v) in inferred.value_shapes {
            value_shapes.entry(k).or_insert(v);
        }
        for (k, v) in inferred.value_types {
            value_types.entry(k).or_insert(v);
        }
        for (k, v) in inferred.const_values {
            const_values.entry(k).or_insert(v);
        }

        // Propagate shapes and fold constant shape expressions in a few passes
        for _ in 0..3 {
            if options.optimize {
                let max_iterations = 10;
                for iteration in 0..max_iterations {
                    let initial_count = value_shapes.len();

                    for onnx_node in onnx_graph.get_node() {
                        let all_outputs_known = onnx_node
                            .get_output()
                            .iter()
                            .all(|out| value_shapes.contains_key(out.as_str()));
                        if all_outputs_known {
                            continue;
                        }

                        if let Some(inferred) =
                            infer_shape(onnx_node, &value_shapes, &initializers_map, &const_values)
                        {
                            if let Some(output_name) = onnx_node.get_output().first() {
                                value_shapes
                                    .entry(output_name.to_string())
                                    .or_insert(inferred);
                            }
                        }
                    }

                    if value_shapes.len() == initial_count {
                        break;
                    }

                    if iteration == max_iterations - 1 {
                        eprintln!(
                            "Warning: Shape propagation reached max iterations ({}/{})",
                            value_shapes.len(),
                            onnx_graph.get_node().len()
                        );
                    }
                }
            }

            let consts_before = const_values.len();

            // Extend const value map for const-foldable shapes
            for node in onnx_graph.get_node() {
                let op_type = node.get_op_type();
                if op_type == "Shape" {
                    if let (Some(inp), Some(out)) =
                        (node.get_input().first(), node.get_output().first())
                    {
                        let out = out.to_string();
                        if let Some(shape) = value_shapes.get(inp).cloned() {
                            if shape.iter().all(|d| *d > 0) {
                                const_values.insert(out.clone(), shape.clone());
                                let inferred_shape = vec![shape.len() as i64];
                                value_shapes
                                    .entry(out.clone())
                                    .or_insert(inferred_shape.clone());
                                value_shapes
                                    .entry(sanitize_identifier(&out))
                                    .or_insert(inferred_shape);
                                value_types.insert(out, DataType::Int64);
                            }
                        }
                    }
                } else if op_type == "Gather" {
                    if let (Some(data_name), Some(indices_name), Some(out)) = (
                        node.get_input().first(),
                        node.get_input().get(1),
                        node.get_output().first(),
                    ) {
                        if let (Some(data), Some(indices)) =
                            (const_values.get(data_name), const_values.get(indices_name))
                        {
                            let axis = node
                                .get_attribute()
                                .iter()
                                .find(|a| a.get_name() == "axis" && a.has_i())
                                .map(|a| a.get_i())
                                .unwrap_or(0);

                            if axis == 0 {
                                let mut gathered = Vec::new();
                                for &idx in indices {
                                    let i = if idx < 0 {
                                        (data.len() as i64 + idx) as usize
                                    } else {
                                        idx as usize
                                    };
                                    if let Some(v) = data.get(i) {
                                        gathered.push(*v);
                                    }
                                }
                                if !gathered.is_empty() {
                                    const_values.insert(out.to_string(), gathered.clone());
                                    let out_shape = if gathered.len() == 1 {
                                        Vec::new()
                                    } else {
                                        vec![gathered.len() as i64]
                                    };
                                    value_shapes
                                        .entry(out.to_string())
                                        .or_insert(out_shape.clone());
                                    value_shapes
                                        .entry(sanitize_identifier(out))
                                        .or_insert(out_shape);
                                    value_types.insert(out.to_string(), DataType::Int64);
                                }
                            }
                        }
                    }
                } else if op_type == "Concat" {
                    let mut axis = 0i64;
                    for attr in node.get_attribute() {
                        if attr.get_name() == "axis" && attr.has_i() {
                            axis = attr.get_i();
                        }
                    }

                    if axis == 0 || axis == -1 {
                        let mut combined = Vec::new();
                        let mut all_known = true;
                        for inp in node.get_input() {
                            if let Some(vals) = const_values.get(inp) {
                                combined.extend_from_slice(vals);
                            } else {
                                all_known = false;
                                break;
                            }
                        }

                        if all_known {
                            if let Some(out) = node.get_output().first() {
                                const_values.insert(out.to_string(), combined.clone());
                                let merged_shape = vec![combined.len() as i64];
                                value_shapes
                                    .entry(out.to_string())
                                    .or_insert(merged_shape.clone());
                                value_shapes
                                    .entry(sanitize_identifier(out))
                                    .or_insert(merged_shape);
                                value_types.insert(out.to_string(), DataType::Int64);
                            }
                        }
                    }
                } else if op_type == "Cast" || op_type == "Unsqueeze" || op_type == "Squeeze" {
                    if let (Some(inp), Some(out)) =
                        (node.get_input().first(), node.get_output().first())
                    {
                        if let Some(vals) = const_values.get(inp).cloned() {
                            const_values.insert(out.to_string(), vals.clone());
                            let out_shape = if vals.len() == 1 {
                                Vec::new()
                            } else {
                                vec![vals.len() as i64]
                            };
                            value_shapes.entry(out.to_string()).or_insert(out_shape);
                            if let Some(dtype) = value_types.get(inp).cloned() {
                                value_types.insert(out.to_string(), dtype);
                            }
                        }
                    }
                }
            }

            if const_values.len() == consts_before {
                break;
            }
        }

        for onnx_node in onnx_graph.get_node() {
            // If all outputs are compile-time constants, emit them directly and skip conversion
            let outputs = onnx_node.get_output();
            if !outputs.is_empty()
                && outputs
                    .iter()
                    .all(|o| const_values.contains_key(o.as_str()))
            {
                let all_scalar = outputs.iter().all(|o| {
                    const_values
                        .get(o.as_str())
                        .map(|v| v.len() == 1)
                        .unwrap_or(false)
                });

                if !all_scalar {
                    // Fallback to normal conversion if any output is not a scalar constant
                    let context = crate::onnx::ops::ConversionContext {
                        initializers: &initializers_map,
                        value_shapes: &value_shapes,
                        const_values: &const_values,
                        value_ids: &value_name_map,
                        value_types: &value_types,
                    };

                    let converted = registry.convert_node(onnx_node, &context)?;

                    for (name, decl) in converted.consts {
                        let decl_dtype = decl.data_type.clone();
                        if let Some(existing) = self.graph.consts.get(&name) {
                            if existing != &decl {
                                return Err(OnnxError::InvalidShape(format!(
                                    "Conflicting constant definitions for '{}'",
                                    name
                                )));
                            }
                        } else {
                            self.graph.consts.insert(name.clone(), decl);
                        }
                        value_name_map.insert(name.clone(), name.clone());
                        value_types.insert(name.clone(), decl_dtype);
                    }

                    for (onnx_out, webnn_id) in converted.output_mappings {
                        value_name_map.insert(onnx_out.clone(), webnn_id.clone());
                        value_name_map.insert(sanitize_identifier(&onnx_out), webnn_id.clone());
                    }

                    self.graph.nodes.extend(converted.nodes);
                    continue;
                }

                for out in outputs {
                    if let Some(values) = const_values.get(out) {
                        let const_name = sanitize_identifier(out);
                        let shape = Vec::new();

                        let decl = crate::ast::ConstDecl {
                            data_type: DataType::Int64,
                            shape,
                            init: crate::ast::ConstInit::InlineBytes {
                                bytes: values[0].to_le_bytes().to_vec(),
                            },
                        };

                        if let Some(existing) = self.graph.consts.get(&const_name) {
                            if existing != &decl {
                                return Err(OnnxError::InvalidShape(format!(
                                    "Conflicting constant definitions for '{}'",
                                    const_name
                                )));
                            }
                        } else {
                            self.graph.consts.insert(const_name.clone(), decl);
                        }

                        value_name_map.insert(out.to_string(), const_name.clone());
                        value_name_map.insert(const_name.clone(), const_name.clone());
                        value_types.insert(out.to_string(), DataType::Int64);
                        value_types.insert(const_name, DataType::Int64);
                    }
                }
                continue;
            }

            let context = crate::onnx::ops::ConversionContext {
                initializers: &initializers_map,
                value_shapes: &value_shapes,
                const_values: &const_values,
                value_ids: &value_name_map,
                value_types: &value_types,
            };

            let converted = registry.convert_node(onnx_node, &context)?;

            for (name, decl) in converted.consts {
                let decl_dtype = decl.data_type.clone();
                if let Some(existing) = self.graph.consts.get(&name) {
                    if existing != &decl {
                        return Err(OnnxError::InvalidShape(format!(
                            "Conflicting constant definitions for '{}'",
                            name
                        )));
                    }
                } else {
                    self.graph.consts.insert(name.clone(), decl);
                }
                value_name_map.insert(name.clone(), name.clone());
                value_types.insert(name.clone(), decl_dtype);
            }

            for (onnx_out, webnn_id) in converted.output_mappings {
                value_name_map.insert(onnx_out.clone(), webnn_id.clone());
                value_name_map.insert(sanitize_identifier(&onnx_out), webnn_id.clone());
            }

            for (onnx_out, dtype) in converted.output_types {
                if let Some(webnn_id) = value_name_map.get(&onnx_out).cloned() {
                    value_types.insert(webnn_id, dtype);
                }
            }

            self.graph.nodes.extend(converted.nodes);
        }

        // Process outputs
        for output in onnx_graph.get_output() {
            let onnx_name = output.get_name();
            if let Some(mapped) = value_name_map.get(onnx_name) {
                self.graph
                    .outputs
                    .insert(sanitize_identifier(onnx_name), mapped.clone());
            } else {
                return Err(OnnxError::InvalidShape(format!(
                    "No WebNN value found for ONNX output '{}'",
                    onnx_name
                )));
            }
        }

        Ok(self.graph)
    }
}

/// Convert an ONNX file to WebNN format with optional weight extraction
pub fn convert_onnx<P: AsRef<Path>>(
    onnx_path: P,
    mut options: ConvertOptions,
) -> Result<GraphJson, OnnxError> {
    // Read ONNX file
    let onnx_path_ref = onnx_path.as_ref();
    let onnx_bytes = fs::read(onnx_path_ref)?;

    // Parse protobuf
    let mut model: ModelProto = protobuf::parse_from_bytes(&onnx_bytes)
        .map_err(|e| OnnxError::ProtobufError(e.to_string()))?;

    // Apply constant folding if optimize flag is set
    if options.optimize {
        eprintln!("Running constant folding...");
        let evaluators = crate::onnx::constant_folding::evaluators::get_evaluators();
        let nodes_folded =
            crate::onnx::constant_folding::fold_constants_in_model(&mut model, &evaluators)?;
        eprintln!("Constant folding: {} nodes folded", nodes_folded);
    }

    // Merge overrides from sidecar dims file if provided implicitly and not already set
    if options.free_dim_overrides.is_empty() {
        let mut sidecar = onnx_path_ref.to_path_buf();
        sidecar.set_extension("dims.json");
        if sidecar.exists() {
            let content = fs::read_to_string(&sidecar)?;
            if let Ok(json) = serde_json::from_str::<JsonValue>(&content) {
                if let Some(obj) = json
                    .get("freeDimensionOverrides")
                    .unwrap_or(&json)
                    .as_object()
                {
                    for (name, value) in obj {
                        if let Some(v) = value.as_u64() {
                            options
                                .free_dim_overrides
                                .entry(name.clone())
                                .or_insert(v as u32);
                        }
                    }
                }
            }
        }
    }

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

        // Convert typed data to bytes if raw_data is empty
        let bytes_to_write: Vec<u8> = if raw_data.is_empty() {
            // Try to extract from typed data fields
            let int64_data = initializer.get_int64_data();
            let float_data = initializer.get_float_data();
            let int32_data = initializer.get_int32_data();
            let double_data = initializer.get_double_data();

            if !int64_data.is_empty() {
                // Convert int64_data to bytes (little-endian)
                int64_data.iter().flat_map(|&v| v.to_le_bytes()).collect()
            } else if !float_data.is_empty() {
                // Convert float_data to bytes (little-endian)
                float_data.iter().flat_map(|&v| v.to_le_bytes()).collect()
            } else if !int32_data.is_empty() {
                // Convert int32_data to bytes (little-endian)
                int32_data.iter().flat_map(|&v| v.to_le_bytes()).collect()
            } else if !double_data.is_empty() {
                // Convert double_data to bytes (little-endian)
                double_data.iter().flat_map(|&v| v.to_le_bytes()).collect()
            } else {
                // No data at all - skip this initializer
                eprintln!("Warning: Skipping initializer '{}' with no data", name);
                continue;
            }
        } else {
            raw_data.to_vec()
        };

        let byte_length = bytes_to_write.len() as u64;

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
        weights_data.extend_from_slice(&bytes_to_write);
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
