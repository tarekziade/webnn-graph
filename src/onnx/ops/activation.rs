// Activation and unary math operators: Relu, Gelu, Tanh, Sigmoid, Sqrt, Exp, Log, Abs, Neg, Erf

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, OpHandler};
use onnx::onnx::NodeProto;
use serde_json::Map;

pub struct ActivationHandler;

impl OpHandler for ActivationHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "Relu" | "Gelu" | "Tanh" | "Sigmoid" | "Sqrt" | "Exp" | "Log" | "Abs" | "Neg" | "Erf"
        )
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

        // Map ONNX operator to WebNN operation name
        let webnn_op = match op_type {
            "Relu" => "relu",
            "Gelu" => "gelu",
            "Tanh" => "tanh",
            "Sigmoid" => "sigmoid",
            "Sqrt" => "sqrt",
            "Exp" => "exp",
            "Log" => "log",
            "Abs" => "abs",
            "Neg" => "neg",
            "Erf" => "erf",
            _ => {
                return Err(OnnxError::UnsupportedOp {
                    op: op_type.to_string(),
                    node: node_name,
                })
            }
        };

        self.convert_unary(node, &node_name, webnn_op)
    }
}

impl ActivationHandler {
    /// Convert ONNX unary/activation operation to WebNN
    fn convert_unary(
        &self,
        node: &NodeProto,
        node_name: &str,
        webnn_op: &str,
    ) -> Result<Vec<Node>, OnnxError> {
        let inputs = node.get_input();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects 1 input, got {}",
                webnn_op,
                inputs.len()
            )));
        }

        let output_name = if node.get_output().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.get_output()[0].to_string())
        };

        let input0 = sanitize_identifier(&inputs[0].to_string());

        let options = Map::new();

        Ok(vec![Node {
            id: output_name,
            op: webnn_op.to_string(),
            inputs: vec![input0],
            options,
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
    fn test_activation_handler_supports() {
        let handler = ActivationHandler;
        assert!(handler.supports("Relu"));
        assert!(handler.supports("Gelu"));
        assert!(handler.supports("Tanh"));
        assert!(handler.supports("Sigmoid"));
        assert!(handler.supports("Sqrt"));
        assert!(handler.supports("Exp"));
        assert!(handler.supports("Log"));
        assert!(handler.supports("Abs"));
        assert!(handler.supports("Neg"));
        assert!(handler.supports("Erf"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_relu() {
        let handler = ActivationHandler;
        let node = create_test_node("Relu", vec!["x"], vec!["y"]);
        let context = ConversionContext {
            initializers: std::collections::HashMap::new(),
            value_shapes: std::collections::HashMap::new(),
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "relu");
        assert_eq!(result[0].inputs, vec!["x"]);
    }

    #[test]
    fn test_convert_sqrt() {
        let handler = ActivationHandler;
        let node = create_test_node("Sqrt", vec!["x"], vec!["y"]);
        let context = ConversionContext {
            initializers: std::collections::HashMap::new(),
            value_shapes: std::collections::HashMap::new(),
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "sqrt");
        assert_eq!(result[0].inputs, vec!["x"]);
    }

    #[test]
    fn test_convert_gelu() {
        let handler = ActivationHandler;
        let node = create_test_node("Gelu", vec!["x"], vec!["y"]);
        let context = ConversionContext {
            initializers: std::collections::HashMap::new(),
            value_shapes: std::collections::HashMap::new(),
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "gelu");
    }
}
