// Constant folding for ONNX models
// Eliminates nodes with all-constant inputs by evaluating them at conversion time

pub mod evaluators;

use crate::onnx::convert::OnnxError;
use onnx::onnx::{ModelProto, NodeProto, TensorProto, TensorProto_DataType};
use std::collections::{HashMap, HashSet};

/// Represents constant tensor data with various types
#[derive(Debug, Clone)]
pub enum TensorData {
    Int64(Vec<i64>),
    Int32(Vec<i32>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    UInt8(Vec<u8>),
    Int8(Vec<i8>),
}

impl TensorData {
    /// Get the number of elements in this tensor
    pub fn len(&self) -> usize {
        match self {
            TensorData::Int64(v) => v.len(),
            TensorData::Int32(v) => v.len(),
            TensorData::Float32(v) => v.len(),
            TensorData::Float64(v) => v.len(),
            TensorData::UInt8(v) => v.len(),
            TensorData::Int8(v) => v.len(),
        }
    }

    /// Check if the tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the data type
    pub fn data_type(&self) -> TensorProto_DataType {
        match self {
            TensorData::Int64(_) => TensorProto_DataType::INT64,
            TensorData::Int32(_) => TensorProto_DataType::INT32,
            TensorData::Float32(_) => TensorProto_DataType::FLOAT,
            TensorData::Float64(_) => TensorProto_DataType::DOUBLE,
            TensorData::UInt8(_) => TensorProto_DataType::UINT8,
            TensorData::Int8(_) => TensorProto_DataType::INT8,
        }
    }

    /// Convert to bytes (little-endian)
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            TensorData::Int64(v) => v.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            TensorData::Int32(v) => v.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            TensorData::Float32(v) => v.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            TensorData::Float64(v) => v.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            TensorData::UInt8(v) => v.clone(),
            TensorData::Int8(v) => v.iter().map(|&x| x as u8).collect(),
        }
    }

    /// Create from TensorProto
    pub fn from_tensor_proto(tensor: &TensorProto) -> Result<Self, OnnxError> {
        let raw_data = tensor.get_raw_data();
        let data_type = tensor.get_data_type();

        if !raw_data.is_empty() {
            // Parse from raw bytes
            match data_type {
                TensorProto_DataType::INT64 => {
                    let values = raw_data
                        .chunks_exact(8)
                        .map(|c| {
                            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                        })
                        .collect();
                    Ok(TensorData::Int64(values))
                }
                TensorProto_DataType::INT32 => {
                    let values = raw_data
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Ok(TensorData::Int32(values))
                }
                TensorProto_DataType::FLOAT => {
                    let values = raw_data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Ok(TensorData::Float32(values))
                }
                TensorProto_DataType::DOUBLE => {
                    let values = raw_data
                        .chunks_exact(8)
                        .map(|c| {
                            f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                        })
                        .collect();
                    Ok(TensorData::Float64(values))
                }
                TensorProto_DataType::UINT8 => Ok(TensorData::UInt8(raw_data.to_vec())),
                TensorProto_DataType::INT8 => Ok(TensorData::Int8(
                    raw_data.iter().map(|&x| x as i8).collect(),
                )),
                _ => Err(OnnxError::TypeConversion(
                    crate::onnx::types::TypeConversionError::UnsupportedDataType(data_type as i32),
                )),
            }
        } else {
            // Parse from typed data fields
            match data_type {
                TensorProto_DataType::INT64 => {
                    Ok(TensorData::Int64(tensor.get_int64_data().to_vec()))
                }
                TensorProto_DataType::INT32 => {
                    Ok(TensorData::Int32(tensor.get_int32_data().to_vec()))
                }
                TensorProto_DataType::FLOAT => {
                    Ok(TensorData::Float32(tensor.get_float_data().to_vec()))
                }
                TensorProto_DataType::DOUBLE => {
                    Ok(TensorData::Float64(tensor.get_double_data().to_vec()))
                }
                _ => Err(OnnxError::TypeConversion(
                    crate::onnx::types::TypeConversionError::UnsupportedDataType(data_type as i32),
                )),
            }
        }
    }
}

/// Represents a constant tensor with its shape and type
#[derive(Debug, Clone)]
pub struct ConstantTensor {
    pub data: TensorData,
    pub shape: Vec<i64>,
    pub data_type: TensorProto_DataType,
}

impl ConstantTensor {
    /// Create a ConstantTensor from a TensorProto
    pub fn from_tensor_proto(tensor: &TensorProto) -> Result<Self, OnnxError> {
        let data = TensorData::from_tensor_proto(tensor)?;
        let shape = tensor.get_dims().to_vec();
        let data_type = tensor.get_data_type();

        Ok(ConstantTensor {
            data,
            shape,
            data_type,
        })
    }

    /// Convert to TensorProto
    pub fn to_tensor_proto(&self, name: &str) -> TensorProto {
        let mut proto = TensorProto::new();
        proto.set_name(name.to_string());
        proto.set_data_type(self.data_type);
        proto.set_dims(self.shape.clone());
        proto.set_raw_data(self.data.to_bytes());
        proto
    }

    /// Get the total number of elements
    pub fn numel(&self) -> i64 {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        }
    }
}

/// Context for constant folding operations
#[derive(Debug)]
pub struct ConstantFoldingContext<'a> {
    /// Map from tensor name to constant value
    pub constants: HashMap<String, ConstantTensor>,
    /// Original ONNX initializers (for reference)
    pub initializers: &'a HashMap<String, &'a TensorProto>,
}

impl<'a> ConstantFoldingContext<'a> {
    /// Create a new context from initializers
    pub fn new(initializers: &'a HashMap<String, &'a TensorProto>) -> Result<Self, OnnxError> {
        let mut constants = HashMap::new();

        for (name, tensor) in initializers.iter() {
            // Only add tensors with data
            if !tensor.get_raw_data().is_empty()
                || !tensor.get_int64_data().is_empty()
                || !tensor.get_int32_data().is_empty()
                || !tensor.get_float_data().is_empty()
                || !tensor.get_double_data().is_empty()
            {
                match ConstantTensor::from_tensor_proto(tensor) {
                    Ok(ct) => {
                        constants.insert((*name).clone(), ct);
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to parse initializer '{}': {}", name, e);
                    }
                }
            }
        }

        Ok(ConstantFoldingContext {
            constants,
            initializers,
        })
    }

    /// Check if a value is a constant
    pub fn is_constant(&self, name: &str) -> bool {
        self.constants.contains_key(name)
    }

    /// Get a constant by name
    pub fn get_constant(&self, name: &str) -> Option<&ConstantTensor> {
        self.constants.get(name)
    }

    /// Add a new constant
    pub fn add_constant(&mut self, name: String, tensor: ConstantTensor) {
        self.constants.insert(name, tensor);
    }
}

/// Result of a constant folding pass
#[derive(Debug, Default)]
pub struct FoldingResult {
    /// New initializers to add to the model
    pub new_initializers: Vec<TensorProto>,
    /// Node indices to remove from the graph
    pub nodes_to_remove: HashSet<usize>,
    /// Number of nodes folded in this pass
    pub nodes_folded: usize,
}

/// Trait for operations that support constant evaluation
pub trait ConstantEvaluator {
    /// Get the operation type this evaluator handles
    fn op_type(&self) -> &str;

    /// Check if this evaluator can handle the given node
    fn can_evaluate(&self, node: &NodeProto, ctx: &ConstantFoldingContext) -> bool;

    /// Evaluate the node with constant inputs, returning output tensors
    fn evaluate(
        &self,
        node: &NodeProto,
        ctx: &ConstantFoldingContext,
    ) -> Result<Vec<ConstantTensor>, OnnxError>;
}

/// Build the initial context from model initializers
fn build_context<'a>(
    _model: &'a ModelProto,
    initializers_map: &'a HashMap<String, &'a TensorProto>,
) -> Result<ConstantFoldingContext<'a>, OnnxError> {
    ConstantFoldingContext::new(initializers_map)
}

/// Identify nodes that have all constant inputs
fn identify_constant_nodes(
    model: &ModelProto,
    ctx: &ConstantFoldingContext,
    evaluators: &[Box<dyn ConstantEvaluator>],
) -> Result<Vec<usize>, OnnxError> {
    let graph = model.get_graph();
    let mut constant_nodes = Vec::new();

    for (idx, node) in graph.get_node().iter().enumerate() {
        // Check if any evaluator can handle this node
        let can_evaluate = evaluators.iter().any(|e| e.can_evaluate(node, ctx));

        if can_evaluate {
            constant_nodes.push(idx);
        }
    }

    Ok(constant_nodes)
}

/// Evaluate constant nodes and return the folding result
fn evaluate_constant_nodes(
    model: &ModelProto,
    constant_node_indices: &[usize],
    ctx: &mut ConstantFoldingContext,
    evaluators: &[Box<dyn ConstantEvaluator>],
) -> Result<FoldingResult, OnnxError> {
    let graph = model.get_graph();
    let mut result = FoldingResult::default();

    for &idx in constant_node_indices {
        let node = &graph.get_node()[idx];

        // Find an evaluator that can handle this node
        let evaluator = evaluators.iter().find(|e| e.can_evaluate(node, ctx));

        if let Some(evaluator) = evaluator {
            match evaluator.evaluate(node, ctx) {
                Ok(output_tensors) => {
                    // Add outputs as new initializers
                    for (i, tensor) in output_tensors.iter().enumerate() {
                        if i < node.get_output().len() {
                            let output_name = &node.get_output()[i];
                            let proto = tensor.to_tensor_proto(output_name);
                            result.new_initializers.push(proto.clone());

                            // Add to context for subsequent evaluations
                            ctx.add_constant(output_name.to_string(), tensor.clone());
                        }
                    }

                    result.nodes_to_remove.insert(idx);
                    result.nodes_folded += 1;
                }
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to evaluate constant node '{}' ({}): {}",
                        node.get_name(),
                        node.get_op_type(),
                        e
                    );
                }
            }
        }
    }

    Ok(result)
}

/// Main entry point: fold constants in an ONNX model
pub fn fold_constants_in_model(
    model: &mut ModelProto,
    evaluators: &[Box<dyn ConstantEvaluator>],
) -> Result<usize, OnnxError> {
    let mut total_folded = 0;
    let max_iterations = 10;

    // Build initializers map
    let graph = model.get_graph();
    let mut initializers_map: HashMap<String, &TensorProto> = HashMap::new();
    for init in graph.get_initializer() {
        initializers_map.insert(init.get_name().to_string(), init);
    }

    for iteration in 0..max_iterations {
        // 1. Build context from current initializers
        let initializers_map_ref: HashMap<String, &TensorProto> = model
            .get_graph()
            .get_initializer()
            .iter()
            .map(|init| (init.get_name().to_string(), init))
            .collect();

        let mut ctx = build_context(model, &initializers_map_ref)?;

        // 2. Identify constant nodes
        let constant_nodes = identify_constant_nodes(model, &ctx, evaluators)?;

        if constant_nodes.is_empty() {
            break;
        }

        // 3. Evaluate constant nodes
        let result = evaluate_constant_nodes(model, &constant_nodes, &mut ctx, evaluators)?;

        if result.nodes_folded == 0 {
            break;
        }

        // 4. Add new initializers to the model
        let graph_mut = model.mut_graph();
        for init in result.new_initializers {
            graph_mut.mut_initializer().push(init);
        }

        // 5. Remove evaluated nodes
        let nodes = graph_mut.get_node().to_vec();
        graph_mut.clear_node();
        for (idx, node) in nodes.into_iter().enumerate() {
            if !result.nodes_to_remove.contains(&idx) {
                graph_mut.mut_node().push(node);
            }
        }

        total_folded += result.nodes_folded;

        eprintln!(
            "Constant folding iteration {}: {} nodes folded",
            iteration + 1,
            result.nodes_folded
        );
    }

    Ok(total_folded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_data_len() {
        let data = TensorData::Int64(vec![1, 2, 3]);
        assert_eq!(data.len(), 3);

        let data = TensorData::Float32(vec![1.0, 2.0]);
        assert_eq!(data.len(), 2);
    }

    #[test]
    fn test_tensor_data_to_bytes() {
        let data = TensorData::Int32(vec![1, 2, 3]);
        let bytes = data.to_bytes();
        assert_eq!(bytes.len(), 12); // 3 * 4 bytes

        let data = TensorData::Int64(vec![1, 2]);
        let bytes = data.to_bytes();
        assert_eq!(bytes.len(), 16); // 2 * 8 bytes
    }

    #[test]
    fn test_constant_tensor_numel() {
        let ct = ConstantTensor {
            data: TensorData::Int64(vec![1, 2, 3, 4, 5, 6]),
            shape: vec![2, 3],
            data_type: TensorProto_DataType::INT64,
        };
        assert_eq!(ct.numel(), 6);

        let ct = ConstantTensor {
            data: TensorData::Int64(vec![42]),
            shape: vec![],
            data_type: TensorProto_DataType::INT64,
        };
        assert_eq!(ct.numel(), 1);
    }
}
