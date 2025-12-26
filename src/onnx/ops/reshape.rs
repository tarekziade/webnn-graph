// Reshape operators: Reshape, Transpose, Concat, Split, Unsqueeze, Squeeze

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, OpHandler};
use onnx::onnx::NodeProto;
use serde_json::Map;

pub struct ReshapeHandler;

impl OpHandler for ReshapeHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "Reshape" | "Transpose" | "Concat" | "Split" | "Unsqueeze" | "Squeeze"
        )
    }

    fn convert<'a>(
        &self,
        node: &NodeProto,
        context: &ConversionContext<'a>,
    ) -> Result<Vec<Node>, OnnxError> {
        let op_type = node.get_op_type();
        let node_name = if node.has_name() {
            node.get_name().to_string()
        } else {
            "unnamed".to_string()
        };

        match op_type {
            "Reshape" => self.convert_reshape(node, &node_name, context),
            "Transpose" => self.convert_transpose(node, &node_name),
            "Concat" => self.convert_concat(node, &node_name),
            "Split" => self.convert_split(node, &node_name),
            "Unsqueeze" => self.convert_unsqueeze(node, &node_name),
            "Squeeze" => self.convert_squeeze(node, &node_name),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

impl ReshapeHandler {
    /// Convert ONNX Reshape to WebNN reshape
    /// ONNX Reshape takes shape as a second input (constant tensor)
    /// WebNN reshape takes newShape as a static array option
    fn convert_reshape<'a>(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &crate::onnx::ops::ConversionContext<'a>,
    ) -> Result<Vec<Node>, OnnxError> {
        let inputs = node.get_input();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Reshape expects 2 inputs (data, shape), got {}",
                inputs.len()
            )));
        }

        let output_name = if node.get_output().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.get_output()[0].to_string())
        };

        let input0 = sanitize_identifier(&inputs[0].to_string());
        let shape_input_name = &inputs[1];

        // Resolve shape from initializers - WebNN requires static newShape (no -1 for inference)
        let shape_values: Vec<i64> = if let Some(initializer) =
            context.initializers.get(shape_input_name.as_str())
        {
            // Extract int64 values from the initializer
            let raw_data = initializer.get_raw_data();
            if raw_data.is_empty() {
                // Try int64_data field
                initializer.get_int64_data().to_vec()
            } else {
                // Parse from raw bytes (little-endian int64)
                raw_data
                    .chunks_exact(8)
                    .map(|chunk| {
                        i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect()
            }
        } else {
            return Err(OnnxError::InvalidShape(format!(
                "Reshape shape input '{}' not found in initializers. WebNN requires static newShape.",
                shape_input_name
            )));
        };

        // Handle -1 (dimension inference marker) - compute the inferred dimension
        // WebNN requires all dimensions to be explicit, so we need to resolve -1 values
        let shape_values: Vec<u32> = if shape_values.contains(&-1) {
            // Need to infer the -1 dimension based on input tensor shape
            let input_name = &inputs[0];

            // Look up input tensor shape
            let input_shape = context
                .value_shapes
                .get(input_name.as_str())
                .ok_or_else(|| {
                    OnnxError::InvalidShape(format!(
                        "Cannot infer reshape dimension: input '{}' shape not found",
                        input_name
                    ))
                })?;

            // Validate all input dimensions are positive (WebNN requirement)
            if input_shape.iter().any(|&d| d <= 0) {
                return Err(OnnxError::InvalidShape(format!(
                    "Cannot infer reshape dimension: input '{}' has dynamic/unknown dimensions {:?}. \
                    WebNN requires all dimensions to be statically known (> 0). \
                    Please ensure onnx-simplifier fully resolved all dimensions.",
                    input_name, input_shape
                )));
            }

            // Calculate total elements in input tensor
            let total_elements: i64 = input_shape.iter().product();

            // Calculate product of known dimensions and infer the -1 dimension
            let mut inferred_shape = Vec::new();
            let mut known_product: i64 = 1;
            let mut infer_index = None;

            for (i, &dim) in shape_values.iter().enumerate() {
                if dim == -1 {
                    if infer_index.is_some() {
                        return Err(OnnxError::InvalidShape(
                            "Reshape cannot have multiple -1 dimensions".to_string(),
                        ));
                    }
                    infer_index = Some(i);
                    inferred_shape.push(0); // Placeholder
                } else {
                    known_product *= dim;
                    inferred_shape.push(dim as u32);
                }
            }

            // Compute inferred dimension
            if let Some(idx) = infer_index {
                let inferred_dim = total_elements / known_product;
                if inferred_dim <= 0 || total_elements % known_product != 0 {
                    return Err(OnnxError::InvalidShape(format!(
                        "Cannot infer reshape dimension: {} elements cannot be reshaped to {:?}",
                        total_elements, shape_values
                    )));
                }
                inferred_shape[idx] = inferred_dim as u32;
            }

            inferred_shape
        } else {
            // All dimensions are positive, safe to convert
            shape_values.iter().map(|&v| v as u32).collect()
        };

        let mut options = Map::new();
        options.insert("newShape".to_string(), serde_json::json!(shape_values));

        Ok(vec![Node {
            id: output_name,
            op: "reshape".to_string(),
            inputs: vec![input0],
            options,
            outputs: None,
        }])
    }

    /// Convert ONNX Transpose to WebNN transpose
    fn convert_transpose(&self, node: &NodeProto, node_name: &str) -> Result<Vec<Node>, OnnxError> {
        let inputs = node.get_input();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Transpose expects 1 input, got {}",
                inputs.len()
            )));
        }

        // Extract perm attribute (permutation)
        let mut perm: Option<Vec<i64>> = None;
        for attr in node.get_attribute() {
            if attr.get_name() == "perm" {
                perm = Some(attr.get_ints().to_vec());
            }
        }

        let output_name = if node.get_output().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.get_output()[0].to_string())
        };

        let input0 = sanitize_identifier(&inputs[0].to_string());

        let mut options = Map::new();
        if let Some(perm_values) = perm {
            options.insert("permutation".to_string(), serde_json::json!(perm_values));
        }

        Ok(vec![Node {
            id: output_name,
            op: "transpose".to_string(),
            inputs: vec![input0],
            options,
            outputs: None,
        }])
    }

    /// Convert ONNX Concat to WebNN concat
    fn convert_concat(&self, node: &NodeProto, node_name: &str) -> Result<Vec<Node>, OnnxError> {
        let inputs = node.get_input();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Concat expects at least 2 inputs, got {}",
                inputs.len()
            )));
        }

        // Extract axis attribute (required in ONNX)
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

        let sanitized_inputs: Vec<String> = inputs
            .iter()
            .map(|s| sanitize_identifier(&s.to_string()))
            .collect();

        let mut options = Map::new();
        options.insert("axis".to_string(), serde_json::json!(axis));

        Ok(vec![Node {
            id: output_name,
            op: "concat".to_string(),
            inputs: sanitized_inputs,
            options,
            outputs: None,
        }])
    }

    /// Convert ONNX Split to WebNN split
    fn convert_split(&self, node: &NodeProto, node_name: &str) -> Result<Vec<Node>, OnnxError> {
        let inputs = node.get_input();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Split expects at least 1 input".to_string(),
            ));
        }

        // Extract axis attribute
        let mut axis = 0i64;
        let mut splits: Option<Vec<i64>> = None;

        for attr in node.get_attribute() {
            match attr.get_name() {
                "axis" => {
                    if attr.has_i() {
                        axis = attr.get_i();
                    }
                }
                "split" => {
                    splits = Some(attr.get_ints().to_vec());
                }
                _ => {}
            }
        }

        let outputs = node.get_output();
        if outputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Split expects at least 1 output".to_string(),
            ));
        }

        let input0 = sanitize_identifier(&inputs[0].to_string());
        let sanitized_outputs: Vec<String> = outputs
            .iter()
            .map(|s| sanitize_identifier(&s.to_string()))
            .collect();

        let mut options = Map::new();
        options.insert("axis".to_string(), serde_json::json!(axis));
        if let Some(split_values) = splits {
            options.insert("splits".to_string(), serde_json::json!(split_values));
        }

        // WebNN split returns multiple outputs
        Ok(vec![Node {
            id: format!("{}_split", node_name),
            op: "split".to_string(),
            inputs: vec![input0],
            options,
            outputs: Some(sanitized_outputs),
        }])
    }

    /// Convert ONNX Unsqueeze to WebNN expand
    /// ONNX Unsqueeze adds dimensions at specified axes
    /// In opset >= 13, axes is a second input; in earlier opsets, it's an attribute
    fn convert_unsqueeze(&self, node: &NodeProto, node_name: &str) -> Result<Vec<Node>, OnnxError> {
        let inputs = node.get_input();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Unsqueeze expects at least 1 input".to_string(),
            ));
        }

        // Extract axes attribute (opset < 13) or use second input (opset >= 13)
        let mut axes: Option<Vec<i64>> = None;
        for attr in node.get_attribute() {
            if attr.get_name() == "axes" {
                axes = Some(attr.get_ints().to_vec());
            }
        }

        let output_name = if node.get_output().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.get_output()[0].to_string())
        };

        let input0 = sanitize_identifier(&inputs[0].to_string());

        let mut options = Map::new();

        // If axes is an attribute, store it
        if let Some(axes_values) = axes {
            options.insert("axes".to_string(), serde_json::json!(axes_values));
        } else if inputs.len() >= 2 {
            // axes is provided as a second input (opset >= 13)
            let input1 = sanitize_identifier(&inputs[1].to_string());
            options.insert("axes".to_string(), serde_json::json!(input1));
        }

        // WebNN doesn't have unsqueeze, so we'll use expand with axes parameter
        Ok(vec![Node {
            id: output_name,
            op: "expand".to_string(),
            inputs: vec![input0],
            options,
            outputs: None,
        }])
    }

    /// Convert ONNX Squeeze to WebNN reshape
    /// ONNX Squeeze removes dimensions of size 1
    fn convert_squeeze(&self, node: &NodeProto, node_name: &str) -> Result<Vec<Node>, OnnxError> {
        let inputs = node.get_input();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Squeeze expects at least 1 input".to_string(),
            ));
        }

        // Extract axes attribute (opset < 13) or use second input (opset >= 13)
        let mut axes: Option<Vec<i64>> = None;
        for attr in node.get_attribute() {
            if attr.get_name() == "axes" {
                axes = Some(attr.get_ints().to_vec());
            }
        }

        let output_name = if node.get_output().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.get_output()[0].to_string())
        };

        let input0 = sanitize_identifier(&inputs[0].to_string());

        let mut options = Map::new();

        // If axes is specified, store it
        if let Some(axes_values) = axes {
            options.insert("axes".to_string(), serde_json::json!(axes_values));
        } else if inputs.len() >= 2 {
            // axes is provided as a second input (opset >= 13)
            let input1 = sanitize_identifier(&inputs[1].to_string());
            options.insert("axes".to_string(), serde_json::json!(input1));
        }

        // WebNN doesn't have squeeze, so we'll use reshape with axes parameter
        Ok(vec![Node {
            id: output_name,
            op: "reshape".to_string(),
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
    fn test_reshape_handler_supports() {
        let handler = ReshapeHandler;
        assert!(handler.supports("Reshape"));
        assert!(handler.supports("Transpose"));
        assert!(handler.supports("Concat"));
        assert!(handler.supports("Split"));
        assert!(handler.supports("Unsqueeze"));
        assert!(handler.supports("Squeeze"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_reshape() {
        let handler = ReshapeHandler;
        let node = create_test_node("Reshape", vec!["data", "shape"], vec!["reshaped"]);

        // Create a mock shape initializer [1, 2, 3, 4]
        let mut shape_tensor = onnx::onnx::TensorProto::new();
        shape_tensor.set_name("shape".to_string());
        shape_tensor.set_data_type(onnx::onnx::TensorProto_DataType::INT64);
        shape_tensor.set_int64_data(vec![1, 2, 3, 4]);

        let mut initializers = std::collections::HashMap::new();
        initializers.insert("shape".to_string(), &shape_tensor);

        // Add input shape for inference (24 elements = 1*2*3*4)
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("data".to_string(), vec![2, 3, 4]); // 24 elements

        let context = ConversionContext {
            initializers,
            value_shapes,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "reshape");
        assert_eq!(result[0].inputs, vec!["data"]);
        assert_eq!(result[0].id, "reshaped");
        // Verify the newShape is now a static array
        assert_eq!(
            result[0].options.get("newShape"),
            Some(&serde_json::json!([1, 2, 3, 4]))
        );
    }

    #[test]
    fn test_convert_transpose() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Transpose", vec!["x"], vec!["y"]);
        add_ints_attribute(&mut node, "perm", vec![1, 0, 2]);
        let context = ConversionContext {
            initializers: std::collections::HashMap::new(),
            value_shapes: std::collections::HashMap::new(),
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "transpose");
        assert_eq!(result[0].inputs, vec!["x"]);
        assert!(result[0].options.contains_key("permutation"));
    }

    #[test]
    fn test_convert_concat() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Concat", vec!["a", "b", "c"], vec!["result"]);
        add_int_attribute(&mut node, "axis", 1);
        let context = ConversionContext {
            initializers: std::collections::HashMap::new(),
            value_shapes: std::collections::HashMap::new(),
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "concat");
        assert_eq!(result[0].inputs.len(), 3);
        assert!(result[0].options.contains_key("axis"));
    }

    #[test]
    fn test_convert_split() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Split", vec!["x"], vec!["y1", "y2"]);
        add_int_attribute(&mut node, "axis", 0);
        let context = ConversionContext {
            initializers: std::collections::HashMap::new(),
            value_shapes: std::collections::HashMap::new(),
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "split");
        assert!(result[0].outputs.is_some());
    }

    #[test]
    fn test_convert_unsqueeze() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Unsqueeze", vec!["x"], vec!["y"]);
        add_ints_attribute(&mut node, "axes", vec![0, 2]);
        let context = ConversionContext {
            initializers: std::collections::HashMap::new(),
            value_shapes: std::collections::HashMap::new(),
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "expand");
        assert_eq!(result[0].inputs, vec!["x"]);
        assert!(result[0].options.contains_key("axes"));
    }

    #[test]
    fn test_convert_squeeze() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Squeeze", vec!["x"], vec!["y"]);
        add_ints_attribute(&mut node, "axes", vec![1]);
        let context = ConversionContext {
            initializers: std::collections::HashMap::new(),
            value_shapes: std::collections::HashMap::new(),
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "reshape");
        assert_eq!(result[0].inputs, vec!["x"]);
        assert!(result[0].options.contains_key("axes"));
    }
}
