// Concat operation evaluator
// Concatenates tensors along a specified axis

use crate::onnx::constant_folding::{
    ConstantEvaluator as EvaluatorTrait, ConstantFoldingContext, ConstantTensor, TensorData,
};
use crate::onnx::convert::OnnxError;
use crate::protos::onnx::NodeProto;

pub struct ConcatEvaluator;

impl ConcatEvaluator {
    /// Concatenate tensor data along axis
    fn concat_data(
        tensors: &[&ConstantTensor],
        axis: i64,
    ) -> Result<(TensorData, Vec<i64>), OnnxError> {
        if tensors.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Concat requires at least one input".to_string(),
            ));
        }

        let first_shape = &tensors[0].shape;
        let rank = first_shape.len();

        // Normalize axis
        let normalized_axis = if axis < 0 {
            (rank as i64 + axis) as usize
        } else {
            axis as usize
        };

        if rank == 0 {
            // All inputs are scalars, treat as 1D concatenation
            return Self::concat_scalars(tensors);
        }

        if normalized_axis >= rank {
            return Err(OnnxError::InvalidShape(format!(
                "Concat axis {} out of bounds for rank {}",
                axis, rank
            )));
        }

        // For 1D tensors or when concatenating along the only dimension
        if rank == 1 || normalized_axis == 0 {
            return Self::concat_along_axis_0(tensors);
        }

        // For higher-rank tensors with axis != 0, we'd need proper striding
        Err(OnnxError::UnsupportedOp {
            op: "Concat".to_string(),
            node: format!(
                "axis={} with rank={} (only axis=0 or 1D tensors currently supported)",
                axis, rank
            ),
        })
    }

    /// Concatenate scalar tensors
    fn concat_scalars(tensors: &[&ConstantTensor]) -> Result<(TensorData, Vec<i64>), OnnxError> {
        // All inputs should be scalars, concatenate into 1D array
        let first_type = tensors[0].data_type;

        // Check all have same type
        for tensor in tensors {
            if tensor.data_type != first_type {
                return Err(OnnxError::InvalidShape(
                    "Concat requires all inputs to have the same type".to_string(),
                ));
            }
        }

        match &tensors[0].data {
            TensorData::Int64(_) => {
                let mut result = Vec::new();
                for tensor in tensors {
                    if let TensorData::Int64(ref v) = tensor.data {
                        result.extend_from_slice(v);
                    }
                }
                Ok((TensorData::Int64(result.clone()), vec![result.len() as i64]))
            }
            TensorData::Int32(_) => {
                let mut result = Vec::new();
                for tensor in tensors {
                    if let TensorData::Int32(ref v) = tensor.data {
                        result.extend_from_slice(v);
                    }
                }
                Ok((TensorData::Int32(result.clone()), vec![result.len() as i64]))
            }
            TensorData::Float32(_) => {
                let mut result = Vec::new();
                for tensor in tensors {
                    if let TensorData::Float32(ref v) = tensor.data {
                        result.extend_from_slice(v);
                    }
                }
                Ok((
                    TensorData::Float32(result.clone()),
                    vec![result.len() as i64],
                ))
            }
            _ => Err(OnnxError::UnsupportedOp {
                op: "Concat".to_string(),
                node: format!("data type {:?} not supported", first_type),
            }),
        }
    }

    /// Concatenate along axis 0 (most common for shape operations)
    fn concat_along_axis_0(
        tensors: &[&ConstantTensor],
    ) -> Result<(TensorData, Vec<i64>), OnnxError> {
        let first_type = tensors[0].data_type;
        let first_shape = &tensors[0].shape;

        // All inputs must have same rank and same dimensions except at axis 0
        for tensor in tensors.iter().skip(1) {
            if tensor.data_type != first_type {
                return Err(OnnxError::InvalidShape(
                    "Concat requires all inputs to have the same type".to_string(),
                ));
            }
            if tensor.shape.len() != first_shape.len() {
                return Err(OnnxError::InvalidShape(
                    "Concat requires all inputs to have the same rank".to_string(),
                ));
            }
            for (i, (&d1, &d2)) in first_shape.iter().zip(&tensor.shape).enumerate() {
                if i != 0 && d1 != d2 {
                    return Err(OnnxError::InvalidShape(format!(
                        "Concat requires all inputs to have the same dimensions except at concat axis, \
                         got mismatch at dimension {}: {} vs {}",
                        i, d1, d2
                    )));
                }
            }
        }

        // Calculate output shape
        let mut output_shape = first_shape.clone();
        if !output_shape.is_empty() {
            output_shape[0] = tensors.iter().map(|t| t.shape[0]).sum();
        }

        // Concatenate data
        match &tensors[0].data {
            TensorData::Int64(_) => {
                let mut result = Vec::new();
                for tensor in tensors {
                    if let TensorData::Int64(ref v) = tensor.data {
                        result.extend_from_slice(v);
                    }
                }
                Ok((TensorData::Int64(result), output_shape))
            }
            TensorData::Int32(_) => {
                let mut result = Vec::new();
                for tensor in tensors {
                    if let TensorData::Int32(ref v) = tensor.data {
                        result.extend_from_slice(v);
                    }
                }
                Ok((TensorData::Int32(result), output_shape))
            }
            TensorData::Float32(_) => {
                let mut result = Vec::new();
                for tensor in tensors {
                    if let TensorData::Float32(ref v) = tensor.data {
                        result.extend_from_slice(v);
                    }
                }
                Ok((TensorData::Float32(result), output_shape))
            }
            _ => Err(OnnxError::UnsupportedOp {
                op: "Concat".to_string(),
                node: format!("data type {:?} not supported", first_type),
            }),
        }
    }
}

impl EvaluatorTrait for ConcatEvaluator {
    fn op_type(&self) -> &str {
        "Concat"
    }

    fn can_evaluate(&self, node: &NodeProto, ctx: &ConstantFoldingContext) -> bool {
        if node.op_type.as_str() != "Concat" {
            return false;
        }

        // All inputs must be constants
        node.input
            .as_slice()
            .iter()
            .all(|input| ctx.is_constant(input.as_str()))
    }

    fn evaluate(
        &self,
        node: &NodeProto,
        ctx: &ConstantFoldingContext,
    ) -> Result<Vec<ConstantTensor>, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::MissingAttribute {
                attr: "inputs".to_string(),
                op: "Concat".to_string(),
            });
        }

        // Get axis attribute (required for Concat)
        let axis = node
            .attribute
            .as_slice()
            .iter()
            .find(|a| a.name.as_str() == "axis")
            .map(|a| a.i)
            .ok_or_else(|| OnnxError::MissingAttribute {
                attr: "axis".to_string(),
                op: "Concat".to_string(),
            })?;

        // Gather all input tensors
        let mut input_tensors = Vec::new();
        for input_name in inputs.iter() {
            let tensor = ctx.get_constant(input_name.as_str()).ok_or_else(|| {
                OnnxError::ShapeInference(format!("Input tensor '{}' not found", input_name))
            })?;
            input_tensors.push(tensor);
        }

        // Perform concatenation
        let (output_data, output_shape) = Self::concat_data(&input_tensors, axis)?;

        let output = ConstantTensor {
            data: output_data.clone(),
            shape: output_shape,
            data_type: output_data.data_type().into(),
        };

        Ok(vec![output])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::{AttributeProto, NodeProto, TensorProto, TensorProto_DataType};
    use std::collections::HashMap;

    #[test]
    fn test_concat_1d_tensors() {
        // Test concatenating 1D int64 tensors
        // This is the common pattern for shape operations

        let tensor1 = TensorProto {
            name: "t1".to_string(),
            data_type: TensorProto_DataType::Int64.into(),
            dims: vec![2],
            int64_data: vec![2, 128],
            ..Default::default()
        };

        let tensor1_static: &'static TensorProto = Box::leak(Box::new(tensor1));

        let tensor2 = TensorProto {
            name: "t2".to_string(),
            data_type: TensorProto_DataType::Int64.into(),
            dims: vec![1],
            int64_data: vec![384],
            ..Default::default()
        };

        let tensor2_static: &'static TensorProto = Box::leak(Box::new(tensor2));

        let mut init_map = HashMap::new();
        init_map.insert("t1".to_string(), tensor1_static);
        init_map.insert("t2".to_string(), tensor2_static);

        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = ConcatEvaluator;

        // Create Concat node
        let mut node = NodeProto {
            op_type: "Concat".to_string(),
            input: vec!["t1".to_string(), "t2".to_string()],
            output: vec!["concatenated".to_string()],
            ..Default::default()
        };

        // Add axis attribute
        let axis_attr = AttributeProto {
            name: "axis".to_string(),
            i: 0,
            ..Default::default()
        };
        node.attribute.push(axis_attr);

        // Evaluate
        assert!(evaluator.can_evaluate(&node, &ctx));
        let result = evaluator.evaluate(&node, &ctx).unwrap();

        assert_eq!(result.len(), 1);
        let output = &result[0];

        if let TensorData::Int64(ref values) = output.data {
            assert_eq!(values, &vec![2, 128, 384]);
        } else {
            panic!("Expected Int64 data");
        }

        assert_eq!(output.shape, vec![3]);
    }

    #[test]
    fn test_concat_scalars() {
        // Test concatenating scalar tensors into 1D array

        let scalar1 = TensorProto {
            name: "s1".to_string(),
            data_type: TensorProto_DataType::Int64.into(),
            dims: vec![],
            int64_data: vec![12],
            ..Default::default()
        };

        let scalar1_static: &'static TensorProto = Box::leak(Box::new(scalar1));

        let scalar2 = TensorProto {
            name: "s2".to_string(),
            data_type: TensorProto_DataType::Int64.into(),
            dims: vec![],
            int64_data: vec![64],
            ..Default::default()
        };

        let scalar2_static: &'static TensorProto = Box::leak(Box::new(scalar2));

        let mut init_map = HashMap::new();
        init_map.insert("s1".to_string(), scalar1_static);
        init_map.insert("s2".to_string(), scalar2_static);

        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = ConcatEvaluator;

        let mut node = NodeProto {
            op_type: "Concat".to_string(),
            input: vec!["s1".to_string(), "s2".to_string()],
            output: vec!["result".to_string()],
            ..Default::default()
        };

        let axis_attr = AttributeProto {
            name: "axis".to_string(),
            i: 0,
            ..Default::default()
        };
        node.attribute.push(axis_attr);

        let result = evaluator.evaluate(&node, &ctx).unwrap();
        let output = &result[0];

        if let TensorData::Int64(ref values) = output.data {
            assert_eq!(values, &vec![12, 64]);
        } else {
            panic!("Expected Int64 data");
        }

        assert_eq!(output.shape, vec![2]);
    }
}
