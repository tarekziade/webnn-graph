// Constant operation evaluator
// Extracts inline constant values from Constant nodes

use crate::onnx::constant_folding::{
    ConstantEvaluator as EvaluatorTrait, ConstantFoldingContext, ConstantTensor,
};
use crate::onnx::convert::OnnxError;
use onnx::onnx::NodeProto;

pub struct ConstantEvaluator;

impl EvaluatorTrait for ConstantEvaluator {
    fn op_type(&self) -> &str {
        "Constant"
    }

    fn can_evaluate(&self, node: &NodeProto, _ctx: &ConstantFoldingContext) -> bool {
        if node.get_op_type() != "Constant" {
            return false;
        }

        // Check if node has a 'value' attribute with tensor data
        node.get_attribute()
            .iter()
            .any(|a| a.get_name() == "value" && a.has_t())
    }

    fn evaluate(
        &self,
        node: &NodeProto,
        _ctx: &ConstantFoldingContext,
    ) -> Result<Vec<ConstantTensor>, OnnxError> {
        let value_attr = node
            .get_attribute()
            .iter()
            .find(|a| a.get_name() == "value" && a.has_t())
            .ok_or_else(|| OnnxError::MissingAttribute {
                attr: "value".to_string(),
                op: "Constant".to_string(),
            })?;

        let tensor_proto = value_attr.get_t();

        // Convert TensorProto to ConstantTensor
        let constant_tensor = ConstantTensor::from_tensor_proto(tensor_proto)?;

        Ok(vec![constant_tensor])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::constant_folding::TensorData;
    use onnx::onnx::{AttributeProto, NodeProto, TensorProto, TensorProto_DataType};
    use std::collections::HashMap;

    #[test]
    fn test_constant_evaluator() {
        let init_map = HashMap::new();
        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = ConstantEvaluator;

        // Create a Constant node with inline tensor
        let mut tensor = TensorProto::new();
        tensor.set_data_type(TensorProto_DataType::INT64);
        tensor.set_dims(vec![3]);
        tensor.set_int64_data(vec![1, 2, 3]);

        let mut value_attr = AttributeProto::new();
        value_attr.set_name("value".to_string());
        value_attr.set_t(tensor);

        let mut node = NodeProto::new();
        node.set_op_type("Constant".to_string());
        node.set_output(protobuf::RepeatedField::from_vec(vec![
            "const_output".to_string()
        ]));
        node.set_attribute(protobuf::RepeatedField::from_vec(vec![value_attr]));

        // Check can_evaluate
        assert!(evaluator.can_evaluate(&node, &ctx));

        // Evaluate
        let result = evaluator.evaluate(&node, &ctx).unwrap();
        assert_eq!(result.len(), 1);

        let output = &result[0];
        assert_eq!(output.shape, vec![3]);
        assert_eq!(output.data_type, TensorProto_DataType::INT64);

        if let TensorData::Int64(ref values) = output.data {
            assert_eq!(values, &vec![1, 2, 3]);
        } else {
            panic!("Expected Int64 data");
        }
    }

    #[test]
    fn test_constant_scalar() {
        let init_map = HashMap::new();
        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = ConstantEvaluator;

        // Create a scalar constant
        let mut tensor = TensorProto::new();
        tensor.set_data_type(TensorProto_DataType::INT64);
        tensor.set_dims(vec![]); // Scalar
        tensor.set_int64_data(vec![42]);

        let mut value_attr = AttributeProto::new();
        value_attr.set_name("value".to_string());
        value_attr.set_t(tensor);

        let mut node = NodeProto::new();
        node.set_op_type("Constant".to_string());
        node.set_output(protobuf::RepeatedField::from_vec(vec![
            "scalar_const".to_string()
        ]));
        node.set_attribute(protobuf::RepeatedField::from_vec(vec![value_attr]));

        let result = evaluator.evaluate(&node, &ctx).unwrap();
        let output = &result[0];

        assert_eq!(output.shape, Vec::<i64>::new()); // Scalar
        if let TensorData::Int64(ref values) = output.data {
            assert_eq!(values, &vec![42]);
        } else {
            panic!("Expected Int64 data");
        }
    }
}
