// Elementwise binary operators: Add, Sub, Mul, Div, Pow

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, OpHandler};
use onnx::onnx::NodeProto;
use serde_json::Map;

pub struct ElementwiseHandler;

impl OpHandler for ElementwiseHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type, "Add" | "Sub" | "Mul" | "Div" | "Pow")
    }

    fn convert(
        &self,
        node: &NodeProto,
        _context: &ConversionContext,
    ) -> Result<Vec<Node>, OnnxError> {
        let op_type = node.get_op_type();
        let node_name = if node.has_name() {
            node.get_name().to_string()
        } else {
            "unnamed".to_string()
        };

        let inputs = node.get_input();
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects 2 inputs, got {}",
                op_type,
                inputs.len()
            )));
        }

        let output_name = if node.get_output().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.get_output()[0].to_string())
        };

        // Sanitize input names
        let input0 = sanitize_identifier(&inputs[0].to_string());
        let input1 = sanitize_identifier(&inputs[1].to_string());

        // Map ONNX operator to WebNN operator (lowercase)
        let webnn_op = match op_type {
            "Add" => "add",
            "Sub" => "sub",
            "Mul" => "mul",
            "Div" => "div",
            "Pow" => "pow",
            _ => {
                return Err(OnnxError::UnsupportedOp {
                    op: op_type.to_string(),
                    node: node_name,
                })
            }
        };

        Ok(vec![Node {
            id: output_name,
            op: webnn_op.to_string(),
            inputs: vec![input0, input1],
            options: Map::new(),
            outputs: None,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx::onnx::NodeProto;

    fn create_test_node(op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        let mut node = NodeProto::new();
        node.set_op_type(op_type.to_string());
        node.set_name(format!("test_{}", op_type.to_lowercase()));
        node.set_input(protobuf::RepeatedField::from_vec(
            inputs.iter().map(|s| s.to_string()).collect(),
        ));
        node.set_output(protobuf::RepeatedField::from_vec(
            outputs.iter().map(|s| s.to_string()).collect(),
        ));
        node
    }

    #[test]
    fn test_elementwise_handler_supports() {
        let handler = ElementwiseHandler;
        assert!(handler.supports("Add"));
        assert!(handler.supports("Sub"));
        assert!(handler.supports("Mul"));
        assert!(handler.supports("Div"));
        assert!(handler.supports("Pow"));
        assert!(!handler.supports("MatMul"));
    }

    #[test]
    fn test_convert_add() {
        let handler = ElementwiseHandler;
        let node = create_test_node("Add", vec!["a", "b"], vec!["c"]);
        let context = ConversionContext {};

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "add");
        assert_eq!(result[0].inputs, vec!["a", "b"]);
        assert_eq!(result[0].id, "c");
    }

    #[test]
    fn test_convert_mul() {
        let handler = ElementwiseHandler;
        let node = create_test_node("Mul", vec!["x", "y"], vec!["z"]);
        let context = ConversionContext {};

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "mul");
        assert_eq!(result[0].inputs, vec!["x", "y"]);
    }

    #[test]
    fn test_convert_div() {
        let handler = ElementwiseHandler;
        let node = create_test_node("Div", vec!["a", "b"], vec!["c"]);
        let context = ConversionContext {};

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "div");
    }
}
