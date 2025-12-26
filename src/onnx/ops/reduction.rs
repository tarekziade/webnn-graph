// Reduction operators: ReduceMean, ReduceSum, ReduceMax, ReduceMin

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, OpHandler};
use onnx::onnx::NodeProto;
use serde_json::Map;

pub struct ReductionHandler;

impl OpHandler for ReductionHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin"
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

        match op_type {
            "ReduceMean" => self.convert_reduce(node, &node_name, "reduceMean"),
            "ReduceSum" => self.convert_reduce(node, &node_name, "reduceSum"),
            "ReduceMax" => self.convert_reduce(node, &node_name, "reduceMax"),
            "ReduceMin" => self.convert_reduce(node, &node_name, "reduceMin"),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

impl ReductionHandler {
    /// Convert ONNX reduce operations to WebNN reduce operations
    fn convert_reduce(
        &self,
        node: &NodeProto,
        node_name: &str,
        webnn_op: &str,
    ) -> Result<Vec<Node>, OnnxError> {
        let inputs = node.get_input();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects at least 1 input",
                webnn_op
            )));
        }

        // Extract attributes
        let mut axes: Option<Vec<i64>> = None;
        let mut keepdims = 1i64; // ONNX default is 1 (keep dimensions)

        for attr in node.get_attribute() {
            match attr.get_name() {
                "axes" => {
                    axes = Some(attr.get_ints().to_vec());
                }
                "keepdims" => {
                    if attr.has_i() {
                        keepdims = attr.get_i();
                    }
                }
                _ => {}
            }
        }

        let output_name = if node.get_output().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.get_output()[0].to_string())
        };

        let input0 = sanitize_identifier(&inputs[0].to_string());

        let mut options = Map::new();

        // Add axes if specified
        if let Some(axes_values) = axes {
            options.insert("axes".to_string(), serde_json::json!(axes_values));
        }

        // Add keepDims option (WebNN uses keepDimensions)
        options.insert(
            "keepDimensions".to_string(),
            serde_json::json!(keepdims != 0),
        );

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
    use onnx::onnx::{AttributeProto, NodeProto};

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

    fn add_int_attribute(node: &mut NodeProto, name: &str, value: i64) {
        let mut attr = AttributeProto::new();
        attr.set_name(name.to_string());
        attr.set_i(value);
        node.mut_attribute().push(attr);
    }

    fn add_ints_attribute(node: &mut NodeProto, name: &str, values: Vec<i64>) {
        let mut attr = AttributeProto::new();
        attr.set_name(name.to_string());
        attr.set_ints(values);
        node.mut_attribute().push(attr);
    }

    #[test]
    fn test_reduction_handler_supports() {
        let handler = ReductionHandler;
        assert!(handler.supports("ReduceMean"));
        assert!(handler.supports("ReduceSum"));
        assert!(handler.supports("ReduceMax"));
        assert!(handler.supports("ReduceMin"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_reduce_mean() {
        let handler = ReductionHandler;
        let mut node = create_test_node("ReduceMean", vec!["x"], vec!["y"]);
        add_ints_attribute(&mut node, "axes", vec![1, 2]);
        add_int_attribute(&mut node, "keepdims", 1);
        let context = ConversionContext {
            initializers: std::collections::HashMap::new(),
            value_shapes: std::collections::HashMap::new(),
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "reduceMean");
        assert_eq!(result[0].inputs, vec!["x"]);
        assert!(result[0].options.contains_key("axes"));
        assert!(result[0].options.contains_key("keepDimensions"));
    }

    #[test]
    fn test_convert_reduce_sum() {
        let handler = ReductionHandler;
        let mut node = create_test_node("ReduceSum", vec!["x"], vec!["y"]);
        add_ints_attribute(&mut node, "axes", vec![0]);
        let context = ConversionContext {
            initializers: std::collections::HashMap::new(),
            value_shapes: std::collections::HashMap::new(),
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "reduceSum");
    }
}
