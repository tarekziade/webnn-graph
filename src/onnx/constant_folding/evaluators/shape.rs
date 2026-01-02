// Shape operation evaluator
// Extracts the shape of a tensor as a 1D int64 tensor

use crate::onnx::constant_folding::{
    ConstantEvaluator as EvaluatorTrait, ConstantFoldingContext, ConstantTensor, TensorData,
};
use crate::onnx::convert::OnnxError;
use onnx::onnx::{NodeProto, TensorProto_DataType};

pub struct ShapeEvaluator;

impl EvaluatorTrait for ShapeEvaluator {
    fn op_type(&self) -> &str {
        "Shape"
    }

    fn can_evaluate(&self, node: &NodeProto, ctx: &ConstantFoldingContext) -> bool {
        if node.get_op_type() != "Shape" {
            return false;
        }

        // Shape operation requires that we know the input's shape
        // The input doesn't need to be a constant, but we need its shape metadata
        if let Some(input_name) = node.get_input().first() {
            // Check if we have this as a constant (which includes shape info)
            if ctx.is_constant(input_name.as_str()) {
                return true;
            }
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
                op: "Shape".to_string(),
            })?;

        let input_tensor = ctx.get_constant(input_name.as_str()).ok_or_else(|| {
            OnnxError::ShapeInference(format!(
                "Input tensor '{}' not found in constants",
                input_name
            ))
        })?;

        // Extract shape as int64 vector
        let shape_values: Vec<i64> = input_tensor.shape.clone();

        // Create output tensor (1D int64 array)
        let output = ConstantTensor {
            data: TensorData::Int64(shape_values.clone()),
            shape: vec![shape_values.len() as i64],
            data_type: TensorProto_DataType::INT64,
        };

        Ok(vec![output])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx::onnx::TensorProto;
    use std::collections::HashMap;

    #[test]
    fn test_shape_evaluator() {
        // Create a test tensor with shape [2, 3, 4]
        let mut tensor = TensorProto::new();
        tensor.set_name("test_input".to_string());
        tensor.set_data_type(TensorProto_DataType::FLOAT);
        tensor.set_dims(vec![2, 3, 4]);
        tensor.set_raw_data(vec![0u8; 4 * 2 * 3 * 4]); // 24 floats

        // We need to leak the tensor to get a 'static reference for the test
        // In production code, the model owns the tensors
        let leaked_tensor: &'static TensorProto = Box::leak(Box::new(tensor));

        let mut init_map = HashMap::new();
        init_map.insert("test_input".to_string(), leaked_tensor);

        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = ShapeEvaluator;

        // Create a Shape node
        let mut node = NodeProto::new();
        node.set_op_type("Shape".to_string());
        node.set_input(protobuf::RepeatedField::from_vec(vec![
            "test_input".to_string()
        ]));
        node.set_output(protobuf::RepeatedField::from_vec(vec![
            "test_output".to_string()
        ]));

        // Check can_evaluate
        assert!(evaluator.can_evaluate(&node, &ctx));

        // Evaluate
        let result = evaluator.evaluate(&node, &ctx).unwrap();
        assert_eq!(result.len(), 1);

        let output = &result[0];
        assert_eq!(output.shape, vec![3]); // Output is 1D with 3 elements
        assert_eq!(output.data_type, TensorProto_DataType::INT64);

        if let TensorData::Int64(ref values) = output.data {
            assert_eq!(values, &vec![2, 3, 4]);
        } else {
            panic!("Expected Int64 data");
        }
    }
}
