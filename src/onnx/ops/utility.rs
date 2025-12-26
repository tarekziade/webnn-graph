// Utility operators: Shape, Gather, Slice

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, OpHandler};
use onnx::onnx::NodeProto;
use serde_json::Map;

pub struct UtilityHandler;

impl OpHandler for UtilityHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type, "Shape" | "Gather" | "Slice")
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
            "Shape" => self.convert_shape(node, &node_name),
            "Gather" => self.convert_gather(node, &node_name),
            "Slice" => self.convert_slice(node, &node_name),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

impl UtilityHandler {
    /// Convert ONNX Shape to WebNN shape operation
    /// Returns a 1D tensor containing the dimensions of the input
    fn convert_shape(&self, node: &NodeProto, node_name: &str) -> Result<Vec<Node>, OnnxError> {
        let inputs = node.get_input();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Shape expects 1 input, got {}",
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

        // WebNN doesn't have a direct shape operation, but we can use identity
        // and mark it with metadata that this is a shape operation
        Ok(vec![Node {
            id: output_name,
            op: "shape".to_string(),
            inputs: vec![input0],
            options,
            outputs: None,
        }])
    }

    /// Convert ONNX Gather to WebNN gather
    /// Gathers elements along a specified axis using indices
    fn convert_gather(&self, node: &NodeProto, node_name: &str) -> Result<Vec<Node>, OnnxError> {
        let inputs = node.get_input();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Gather expects 2 inputs (data, indices), got {}",
                inputs.len()
            )));
        }

        // Extract axis attribute (default: 0)
        let mut axis = 0i64;
        for attr in node.get_attribute() {
            if attr.get_name() == "axis" && attr.has_i() {
                axis = attr.get_i();
            }
        }

        let output_name = if node.get_output().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.get_output()[0].to_string())
        };

        let input0 = sanitize_identifier(&inputs[0].to_string());
        let input1 = sanitize_identifier(&inputs[1].to_string());

        let mut options = Map::new();
        options.insert("axis".to_string(), serde_json::json!(axis));

        Ok(vec![Node {
            id: output_name,
            op: "gather".to_string(),
            inputs: vec![input0, input1],
            options,
            outputs: None,
        }])
    }

    /// Convert ONNX Slice to WebNN slice
    /// Extracts a slice from the input tensor
    fn convert_slice(&self, node: &NodeProto, node_name: &str) -> Result<Vec<Node>, OnnxError> {
        let inputs = node.get_input();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Slice expects at least 1 input".to_string(),
            ));
        }

        let output_name = if node.get_output().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.get_output()[0].to_string())
        };

        let input0 = sanitize_identifier(&inputs[0].to_string());

        let mut options = Map::new();

        // In opset >= 10, starts/ends/axes/steps are inputs
        // In older opsets, they're attributes
        if inputs.len() >= 3 {
            // starts, ends are required inputs (indices 1, 2)
            let input1 = sanitize_identifier(&inputs[1].to_string());
            let input2 = sanitize_identifier(&inputs[2].to_string());
            options.insert("starts".to_string(), serde_json::json!(input1));
            options.insert("ends".to_string(), serde_json::json!(input2));

            if inputs.len() >= 4 {
                // axes is optional input (index 3)
                let input3 = sanitize_identifier(&inputs[3].to_string());
                options.insert("axes".to_string(), serde_json::json!(input3));
            }
            if inputs.len() >= 5 {
                // steps is optional input (index 4)
                let input4 = sanitize_identifier(&inputs[4].to_string());
                options.insert("steps".to_string(), serde_json::json!(input4));
            }
        } else {
            // Extract from attributes (older opset)
            for attr in node.get_attribute() {
                match attr.get_name() {
                    "starts" => {
                        options.insert(
                            "starts".to_string(),
                            serde_json::json!(attr.get_ints().to_vec()),
                        );
                    }
                    "ends" => {
                        options.insert(
                            "ends".to_string(),
                            serde_json::json!(attr.get_ints().to_vec()),
                        );
                    }
                    "axes" => {
                        options.insert(
                            "axes".to_string(),
                            serde_json::json!(attr.get_ints().to_vec()),
                        );
                    }
                    "steps" => {
                        options.insert(
                            "steps".to_string(),
                            serde_json::json!(attr.get_ints().to_vec()),
                        );
                    }
                    _ => {}
                }
            }
        }

        Ok(vec![Node {
            id: output_name,
            op: "slice".to_string(),
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

    #[test]
    fn test_utility_handler_supports() {
        let handler = UtilityHandler;
        assert!(handler.supports("Shape"));
        assert!(handler.supports("Gather"));
        assert!(handler.supports("Slice"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_shape() {
        let handler = UtilityHandler;
        let node = create_test_node("Shape", vec!["x"], vec!["shape"]);
        let context = ConversionContext {};

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "shape");
        assert_eq!(result[0].inputs, vec!["x"]);
    }

    #[test]
    fn test_convert_gather() {
        let handler = UtilityHandler;
        let mut node = create_test_node("Gather", vec!["data", "indices"], vec!["output"]);
        add_int_attribute(&mut node, "axis", 1);
        let context = ConversionContext {};

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "gather");
        assert_eq!(result[0].inputs.len(), 2);
        assert!(result[0].options.contains_key("axis"));
    }

    #[test]
    fn test_convert_slice() {
        let handler = UtilityHandler;
        let node = create_test_node("Slice", vec!["x", "starts", "ends"], vec!["output"]);
        let context = ConversionContext {};

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "slice");
        assert_eq!(result[0].inputs, vec!["x"]);
        assert!(result[0].options.contains_key("starts"));
    }
}
