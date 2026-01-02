// Reshape-related operation evaluators
// Unsqueeze, Squeeze, Cast operations

use crate::onnx::constant_folding::{
    ConstantEvaluator as EvaluatorTrait, ConstantFoldingContext, ConstantTensor, TensorData,
};
use crate::onnx::convert::OnnxError;
use onnx::onnx::{NodeProto, TensorProto_DataType};

/// Unsqueeze operation: add dimensions of size 1
pub struct UnsqueezeEvaluator;

impl EvaluatorTrait for UnsqueezeEvaluator {
    fn op_type(&self) -> &str {
        "Unsqueeze"
    }

    fn can_evaluate(&self, node: &NodeProto, ctx: &ConstantFoldingContext) -> bool {
        if node.get_op_type() != "Unsqueeze" {
            return false;
        }

        // Input must be constant
        if let Some(input) = node.get_input().first() {
            return ctx.is_constant(input.as_str());
        }

        false
    }

    fn evaluate(
        &self,
        node: &NodeProto,
        ctx: &ConstantFoldingContext,
    ) -> Result<Vec<ConstantTensor>, OnnxError> {
        let input_name = node
            .get_input()
            .first()
            .ok_or_else(|| OnnxError::MissingAttribute {
                attr: "input".to_string(),
                op: "Unsqueeze".to_string(),
            })?;

        let input_tensor = ctx.get_constant(input_name.as_str()).ok_or_else(|| {
            OnnxError::ShapeInference(format!("Input tensor '{}' not found", input_name))
        })?;

        // Get axes attribute
        let mut axes: Vec<i64> = node
            .get_attribute()
            .iter()
            .find(|a| a.get_name() == "axes")
            .map(|a| a.get_ints().to_vec())
            .unwrap_or_default();

        if axes.is_empty() {
            return Err(OnnxError::MissingAttribute {
                attr: "axes".to_string(),
                op: "Unsqueeze".to_string(),
            });
        }

        // Compute output shape
        let mut output_shape = input_tensor.shape.clone();
        axes.sort();

        for &axis in &axes {
            let idx = if axis < 0 {
                (output_shape.len() as i64 + axis + 1) as usize
            } else {
                axis as usize
            };

            if idx <= output_shape.len() {
                output_shape.insert(idx, 1);
            }
        }

        // Data remains the same, only shape changes
        let output = ConstantTensor {
            data: input_tensor.data.clone(),
            shape: output_shape,
            data_type: input_tensor.data_type,
        };

        Ok(vec![output])
    }
}

/// Squeeze operation: remove dimensions of size 1
pub struct SqueezeEvaluator;

impl EvaluatorTrait for SqueezeEvaluator {
    fn op_type(&self) -> &str {
        "Squeeze"
    }

    fn can_evaluate(&self, node: &NodeProto, ctx: &ConstantFoldingContext) -> bool {
        if node.get_op_type() != "Squeeze" {
            return false;
        }

        // Input must be constant
        if let Some(input) = node.get_input().first() {
            return ctx.is_constant(input.as_str());
        }

        false
    }

    fn evaluate(
        &self,
        node: &NodeProto,
        ctx: &ConstantFoldingContext,
    ) -> Result<Vec<ConstantTensor>, OnnxError> {
        let input_name = node
            .get_input()
            .first()
            .ok_or_else(|| OnnxError::MissingAttribute {
                attr: "input".to_string(),
                op: "Squeeze".to_string(),
            })?;

        let input_tensor = ctx.get_constant(input_name.as_str()).ok_or_else(|| {
            OnnxError::ShapeInference(format!("Input tensor '{}' not found", input_name))
        })?;

        // Get axes attribute (if not specified, squeeze all dimensions of size 1)
        let axes: Vec<i64> = node
            .get_attribute()
            .iter()
            .find(|a| a.get_name() == "axes")
            .map(|a| a.get_ints().to_vec())
            .unwrap_or_default();

        // Compute output shape
        let output_shape: Vec<i64> = if axes.is_empty() {
            // Squeeze all dimensions of size 1
            input_tensor
                .shape
                .iter()
                .filter(|&&d| d != 1)
                .copied()
                .collect()
        } else {
            // Squeeze only specified axes
            let axes_set: std::collections::HashSet<i64> = axes.iter().copied().collect();

            // Normalize negative axes
            let rank = input_tensor.shape.len() as i64;
            let normalized_axes: std::collections::HashSet<i64> = axes_set
                .iter()
                .map(|&ax| if ax < 0 { rank + ax } else { ax })
                .collect();

            input_tensor
                .shape
                .iter()
                .enumerate()
                .filter(|(i, &d)| !(d == 1 && normalized_axes.contains(&(*i as i64))))
                .map(|(_, &d)| d)
                .collect()
        };

        // Data remains the same, only shape changes
        let output = ConstantTensor {
            data: input_tensor.data.clone(),
            shape: output_shape,
            data_type: input_tensor.data_type,
        };

        Ok(vec![output])
    }
}

/// Cast operation: convert data type
pub struct CastEvaluator;

impl CastEvaluator {
    /// Cast tensor data to a new type
    fn cast_data(
        data: &TensorData,
        target_type: TensorProto_DataType,
    ) -> Result<TensorData, OnnxError> {
        match (data, target_type) {
            // Int64 → other types
            (TensorData::Int64(v), TensorProto_DataType::INT64) => Ok(TensorData::Int64(v.clone())),
            (TensorData::Int64(v), TensorProto_DataType::INT32) => {
                Ok(TensorData::Int32(v.iter().map(|&x| x as i32).collect()))
            }
            (TensorData::Int64(v), TensorProto_DataType::FLOAT) => {
                Ok(TensorData::Float32(v.iter().map(|&x| x as f32).collect()))
            }
            (TensorData::Int64(v), TensorProto_DataType::DOUBLE) => {
                Ok(TensorData::Float64(v.iter().map(|&x| x as f64).collect()))
            }

            // Int32 → other types
            (TensorData::Int32(v), TensorProto_DataType::INT32) => Ok(TensorData::Int32(v.clone())),
            (TensorData::Int32(v), TensorProto_DataType::INT64) => {
                Ok(TensorData::Int64(v.iter().map(|&x| x as i64).collect()))
            }
            (TensorData::Int32(v), TensorProto_DataType::FLOAT) => {
                Ok(TensorData::Float32(v.iter().map(|&x| x as f32).collect()))
            }
            (TensorData::Int32(v), TensorProto_DataType::DOUBLE) => {
                Ok(TensorData::Float64(v.iter().map(|&x| x as f64).collect()))
            }

            // Float32 → other types
            (TensorData::Float32(v), TensorProto_DataType::FLOAT) => {
                Ok(TensorData::Float32(v.clone()))
            }
            (TensorData::Float32(v), TensorProto_DataType::DOUBLE) => {
                Ok(TensorData::Float64(v.iter().map(|&x| x as f64).collect()))
            }
            (TensorData::Float32(v), TensorProto_DataType::INT64) => {
                Ok(TensorData::Int64(v.iter().map(|&x| x as i64).collect()))
            }
            (TensorData::Float32(v), TensorProto_DataType::INT32) => {
                Ok(TensorData::Int32(v.iter().map(|&x| x as i32).collect()))
            }

            // Float64 → other types
            (TensorData::Float64(v), TensorProto_DataType::DOUBLE) => {
                Ok(TensorData::Float64(v.clone()))
            }
            (TensorData::Float64(v), TensorProto_DataType::FLOAT) => {
                Ok(TensorData::Float32(v.iter().map(|&x| x as f32).collect()))
            }
            (TensorData::Float64(v), TensorProto_DataType::INT64) => {
                Ok(TensorData::Int64(v.iter().map(|&x| x as i64).collect()))
            }
            (TensorData::Float64(v), TensorProto_DataType::INT32) => {
                Ok(TensorData::Int32(v.iter().map(|&x| x as i32).collect()))
            }

            _ => Err(OnnxError::UnsupportedOp {
                op: "Cast".to_string(),
                node: format!(
                    "cast from {:?} to {:?} not supported",
                    data.data_type(),
                    target_type
                ),
            }),
        }
    }
}

impl EvaluatorTrait for CastEvaluator {
    fn op_type(&self) -> &str {
        "Cast"
    }

    fn can_evaluate(&self, node: &NodeProto, ctx: &ConstantFoldingContext) -> bool {
        if node.get_op_type() != "Cast" {
            return false;
        }

        // Input must be constant
        if let Some(input) = node.get_input().first() {
            return ctx.is_constant(input.as_str());
        }

        false
    }

    fn evaluate(
        &self,
        node: &NodeProto,
        ctx: &ConstantFoldingContext,
    ) -> Result<Vec<ConstantTensor>, OnnxError> {
        let input_name = node
            .get_input()
            .first()
            .ok_or_else(|| OnnxError::MissingAttribute {
                attr: "input".to_string(),
                op: "Cast".to_string(),
            })?;

        let input_tensor = ctx.get_constant(input_name.as_str()).ok_or_else(|| {
            OnnxError::ShapeInference(format!("Input tensor '{}' not found", input_name))
        })?;

        // Get 'to' attribute (target type)
        let target_type_i32 = node
            .get_attribute()
            .iter()
            .find(|a| a.get_name() == "to" && a.has_i())
            .map(|a| a.get_i() as i32)
            .ok_or_else(|| OnnxError::MissingAttribute {
                attr: "to".to_string(),
                op: "Cast".to_string(),
            })?;

        // Convert i32 to TensorProto_DataType enum
        let target_type = match target_type_i32 {
            1 => TensorProto_DataType::FLOAT,
            2 => TensorProto_DataType::UINT8,
            3 => TensorProto_DataType::INT8,
            5 => TensorProto_DataType::INT16,
            6 => TensorProto_DataType::INT32,
            7 => TensorProto_DataType::INT64,
            11 => TensorProto_DataType::DOUBLE,
            _ => {
                return Err(OnnxError::UnsupportedOp {
                    op: "Cast".to_string(),
                    node: format!("unsupported target type: {}", target_type_i32),
                })
            }
        };

        // Perform cast
        let output_data = Self::cast_data(&input_tensor.data, target_type)?;

        let output = ConstantTensor {
            data: output_data,
            shape: input_tensor.shape.clone(),
            data_type: target_type,
        };

        Ok(vec![output])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx::onnx::{AttributeProto, TensorProto};
    use std::collections::HashMap;

    #[test]
    fn test_unsqueeze() {
        let mut tensor = TensorProto::new();
        tensor.set_name("input".to_string());
        tensor.set_data_type(TensorProto_DataType::INT64);
        tensor.set_dims(vec![3]);
        tensor.set_int64_data(vec![2, 128, 384]);

        let tensor_static: &'static TensorProto = Box::leak(Box::new(tensor));

        let mut init_map = HashMap::new();
        init_map.insert("input".to_string(), tensor_static);

        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = UnsqueezeEvaluator;

        // Unsqueeze at axes [0]
        let mut node = NodeProto::new();
        node.set_op_type("Unsqueeze".to_string());
        node.set_input(protobuf::RepeatedField::from_vec(vec!["input".to_string()]));
        node.set_output(protobuf::RepeatedField::from_vec(
            vec!["output".to_string()],
        ));

        let mut axes_attr = AttributeProto::new();
        axes_attr.set_name("axes".to_string());
        axes_attr.set_ints(vec![0]);
        node.set_attribute(protobuf::RepeatedField::from_vec(vec![axes_attr]));

        let result = evaluator.evaluate(&node, &ctx).unwrap();
        let output = &result[0];

        assert_eq!(output.shape, vec![1, 3]); // Added dimension at front

        if let TensorData::Int64(ref values) = output.data {
            assert_eq!(values, &vec![2, 128, 384]); // Data unchanged
        } else {
            panic!("Expected Int64 data");
        }
    }

    #[test]
    fn test_squeeze() {
        let mut tensor = TensorProto::new();
        tensor.set_name("input".to_string());
        tensor.set_data_type(TensorProto_DataType::INT64);
        tensor.set_dims(vec![1, 3, 1]);
        tensor.set_int64_data(vec![2, 128, 384]);

        let tensor_static: &'static TensorProto = Box::leak(Box::new(tensor));

        let mut init_map = HashMap::new();
        init_map.insert("input".to_string(), tensor_static);

        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = SqueezeEvaluator;

        // Squeeze all dimensions of size 1
        let mut node = NodeProto::new();
        node.set_op_type("Squeeze".to_string());
        node.set_input(protobuf::RepeatedField::from_vec(vec!["input".to_string()]));
        node.set_output(protobuf::RepeatedField::from_vec(
            vec!["output".to_string()],
        ));

        let result = evaluator.evaluate(&node, &ctx).unwrap();
        let output = &result[0];

        assert_eq!(output.shape, vec![3]); // Removed dimensions of size 1

        if let TensorData::Int64(ref values) = output.data {
            assert_eq!(values, &vec![2, 128, 384]); // Data unchanged
        } else {
            panic!("Expected Int64 data");
        }
    }

    #[test]
    fn test_cast_int64_to_int32() {
        let mut tensor = TensorProto::new();
        tensor.set_name("input".to_string());
        tensor.set_data_type(TensorProto_DataType::INT64);
        tensor.set_dims(vec![3]);
        tensor.set_int64_data(vec![2, 128, 384]);

        let tensor_static: &'static TensorProto = Box::leak(Box::new(tensor));

        let mut init_map = HashMap::new();
        init_map.insert("input".to_string(), tensor_static);

        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = CastEvaluator;

        // Cast to INT32
        let mut node = NodeProto::new();
        node.set_op_type("Cast".to_string());
        node.set_input(protobuf::RepeatedField::from_vec(vec!["input".to_string()]));
        node.set_output(protobuf::RepeatedField::from_vec(
            vec!["output".to_string()],
        ));

        let mut to_attr = AttributeProto::new();
        to_attr.set_name("to".to_string());
        to_attr.set_i(6); // INT32 = 6 in ONNX
        node.set_attribute(protobuf::RepeatedField::from_vec(vec![to_attr]));

        let result = evaluator.evaluate(&node, &ctx).unwrap();
        let output = &result[0];

        assert_eq!(output.data_type, TensorProto_DataType::INT32);

        if let TensorData::Int32(ref values) = output.data {
            assert_eq!(values, &vec![2, 128, 384]);
        } else {
            panic!("Expected Int32 data");
        }
    }
}
