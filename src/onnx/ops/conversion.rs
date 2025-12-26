// Type conversion and constant operators: Cast, Constant

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, OpHandler};
use onnx::onnx::NodeProto;
use serde_json::Map;

pub struct ConversionHandler;

impl OpHandler for ConversionHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type, "Cast" | "Constant")
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
            "Cast" => self.convert_cast(node, &node_name),
            "Constant" => self.convert_constant(node, &node_name),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

impl ConversionHandler {
    /// Convert ONNX Cast to WebNN cast
    /// ONNX Cast converts tensor data type
    fn convert_cast(&self, node: &NodeProto, node_name: &str) -> Result<Vec<Node>, OnnxError> {
        let inputs = node.get_input();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Cast expects 1 input, got {}",
                inputs.len()
            )));
        }

        // Extract 'to' attribute (target data type)
        let mut to_type: Option<i64> = None;
        for attr in node.get_attribute() {
            if attr.get_name() == "to" && attr.has_i() {
                to_type = Some(attr.get_i());
            }
        }

        if to_type.is_none() {
            return Err(OnnxError::MissingAttribute {
                attr: "to".to_string(),
                op: "Cast".to_string(),
            });
        }

        let output_name = if node.get_output().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.get_output()[0].to_string())
        };

        let input0 = sanitize_identifier(&inputs[0].to_string());

        // Map ONNX type to WebNN DataType
        let target_type = crate::onnx::types::map_onnx_data_type(to_type.unwrap() as i32)?;

        let mut options = Map::new();
        options.insert(
            "to".to_string(),
            serde_json::json!(format!("{:?}", target_type)),
        );

        Ok(vec![Node {
            id: output_name,
            op: "cast".to_string(),
            inputs: vec![input0],
            options,
            outputs: None,
        }])
    }

    /// Convert ONNX Constant to WebNN constant
    /// ONNX Constant creates an inline constant tensor
    fn convert_constant(&self, node: &NodeProto, node_name: &str) -> Result<Vec<Node>, OnnxError> {
        let output_name = if node.get_output().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.get_output()[0].to_string())
        };

        // Extract 'value' attribute (TensorProto)
        let mut value_tensor = None;
        for attr in node.get_attribute() {
            if attr.get_name() == "value" && attr.has_t() {
                value_tensor = Some(attr.get_t());
            }
        }

        if value_tensor.is_none() {
            return Err(OnnxError::MissingAttribute {
                attr: "value".to_string(),
                op: "Constant".to_string(),
            });
        }

        let tensor = value_tensor.unwrap();
        let onnx_type = tensor.get_data_type() as i32;
        let data_type = crate::onnx::types::map_onnx_data_type(onnx_type)?;

        let shape: Vec<i64> = tensor.get_dims().to_vec();
        let raw_data = tensor.get_raw_data().to_vec();

        let mut options = Map::new();
        options.insert(
            "dataType".to_string(),
            serde_json::json!(format!("{:?}", data_type)),
        );
        options.insert("shape".to_string(), serde_json::json!(shape));

        // Store raw bytes as base64 for now (WebNN implementation can decode)
        let b64_data =
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &raw_data);
        options.insert("data".to_string(), serde_json::json!(b64_data));

        Ok(vec![Node {
            id: output_name,
            op: "constant".to_string(),
            inputs: vec![],
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
    fn test_conversion_handler_supports() {
        let handler = ConversionHandler;
        assert!(handler.supports("Cast"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_cast() {
        let handler = ConversionHandler;
        let mut node = create_test_node("Cast", vec!["x"], vec!["y"]);
        add_int_attribute(&mut node, "to", 7); // INT64
        let context = ConversionContext {
            initializers: std::collections::HashMap::new(),
            value_shapes: std::collections::HashMap::new(),
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "cast");
        assert_eq!(result[0].inputs, vec!["x"]);
        assert!(result[0].options.contains_key("to"));
    }
}
