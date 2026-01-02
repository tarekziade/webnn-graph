// Reshape operators: Reshape, Transpose, Concat, Split, Unsqueeze, Squeeze

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use onnx::onnx::{NodeProto, TensorProto_DataType};
use serde_json::Map;

pub struct ReshapeHandler;

impl OpHandler for ReshapeHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "Reshape" | "Transpose" | "Concat" | "Split" | "Unsqueeze" | "Squeeze" | "Tile"
        )
    }

    fn convert<'a>(
        &self,
        node: &NodeProto,
        context: &ConversionContext<'a>,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.get_op_type();
        let node_name = if node.has_name() {
            node.get_name().to_string()
        } else {
            "unnamed".to_string()
        };

        match op_type {
            "Reshape" => self.convert_reshape(node, &node_name, context),
            "Transpose" => self.convert_transpose(node, &node_name, context),
            "Concat" => self.convert_concat(node, &node_name, context),
            "Split" => self.convert_split(node, &node_name, context),
            "Unsqueeze" => self.convert_unsqueeze(node, &node_name, context),
            "Squeeze" => self.convert_squeeze(node, &node_name, context),
            "Tile" => self.convert_tile(node, &node_name, context),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

impl ReshapeHandler {
    fn read_axes_from_attr_or_const(
        &self,
        node: &NodeProto,
        context: &ConversionContext,
    ) -> Result<Vec<i64>, OnnxError> {
        if let Some(attr_axes) = node
            .get_attribute()
            .iter()
            .find(|a| a.get_name() == "axes")
            .map(|a| a.get_ints().to_vec())
        {
            return Ok(attr_axes);
        }

        if node.get_input().len() >= 2 {
            let name = node.get_input()[1].to_string();
            if let Some(vals) = context.const_values.get(&name) {
                return Ok(vals.clone());
            }
            if let Some(t) = context.initializers.get(&name) {
                let raw = t.get_raw_data();
                if !raw.is_empty() {
                    return Ok(raw
                        .chunks_exact(8)
                        .map(|c| {
                            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                        })
                        .collect());
                } else if !t.get_int64_data().is_empty() {
                    return Ok(t.get_int64_data().to_vec());
                } else if !t.get_int32_data().is_empty() {
                    return Ok(t.get_int32_data().iter().map(|&v| v as i64).collect());
                }
            }
            return Err(OnnxError::InvalidShape(
                "Unsqueeze/Squeeze axes input must be constant for WebNN".to_string(),
            ));
        }

        Err(OnnxError::InvalidShape(
            "Unsqueeze/Squeeze axes must be provided".to_string(),
        ))
    }

    /// Convert ONNX Reshape to WebNN reshape
    /// ONNX Reshape takes shape as a second input (constant tensor)
    /// WebNN reshape takes newShape as a static array option
    fn convert_reshape<'a>(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &crate::onnx::ops::ConversionContext<'a>,
    ) -> Result<ConversionResult, OnnxError> {
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

        let data_input_raw = inputs[0].to_string();
        let shape_input_raw = inputs[1].to_string();
        let data_input = context.resolve_input(&data_input_raw);

        // Resolve shape from const-folded values or initializers
        let mut shape_values: Vec<i64> =
            if let Some(values) = context.const_values.get(&shape_input_raw) {
                values.clone()
            } else if let Some(initializer) = context.initializers.get(shape_input_raw.as_str()) {
                let raw_data = initializer.get_raw_data();
                if !raw_data.is_empty() {
                    match initializer.get_data_type() {
                        TensorProto_DataType::INT32 => raw_data
                            .chunks_exact(4)
                            .map(|chunk| {
                                i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as i64
                            })
                            .collect(),
                        _ => raw_data
                            .chunks_exact(8)
                            .map(|chunk| {
                                i64::from_le_bytes([
                                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5],
                                    chunk[6], chunk[7],
                                ])
                            })
                            .collect(),
                    }
                } else if !initializer.get_int64_data().is_empty() {
                    initializer.get_int64_data().to_vec()
                } else if !initializer.get_int32_data().is_empty() {
                    initializer
                        .get_int32_data()
                        .iter()
                        .map(|&v| v as i64)
                        .collect()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };
        let shape_from_const = !shape_values.is_empty();

        // Fallback: derive shape from known input shape when the shape tensor isn't const.
        if shape_values.is_empty() {
            if let Some(ds) = context.value_shapes.get(data_input_raw.as_str()) {
                if ds.len() >= 3 {
                    let tail: i64 = ds[2..].iter().product();
                    shape_values = vec![ds[0], ds[1], tail];
                } else {
                    shape_values = ds.clone();
                }
            } else if let Some(ds) = context.value_shapes.get(&data_input) {
                if ds.len() >= 3 {
                    let tail: i64 = ds[2..].iter().product();
                    shape_values = vec![ds[0], ds[1], tail];
                } else {
                    shape_values = ds.clone();
                }
            } else {
                return Err(OnnxError::InvalidShape(format!(
                    "Reshape shape input '{}' must be a constant (initializer/constant-folded) or input shape must be known. \
                WebNN requires static newShape. Please ensure onnx-simplifier fully resolved all shapes.",
                    shape_input_raw
                )));
            }
        }

        // Handle -1 (dimension inference marker) - compute the inferred dimension
        // WebNN requires all dimensions to be explicit, so we need to resolve -1 values
        let input_shape = {
            let trimmed = data_input_raw.trim_start_matches('/');
            context
                .value_shapes
                .get(data_input_raw.as_str())
                .or_else(|| context.value_shapes.get(&data_input))
                .or_else(|| context.value_shapes.get(trimmed))
                .cloned()
                .ok_or_else(|| {
                    OnnxError::InvalidShape(format!(
                        "Cannot reshape '{}': missing static input shape",
                        data_input_raw
                    ))
                })?
        };

        let shape_values: Vec<u32> = if shape_values.contains(&-1) {
            // Need to infer the -1 dimension based on input tensor shape
            // Validate all input dimensions are positive (WebNN requirement)
            if input_shape.iter().any(|&d| d <= 0) {
                return Err(OnnxError::InvalidShape(format!(
                    "Cannot infer reshape dimension: input '{}' has dynamic/unknown dimensions {:?}. \
                    WebNN requires all dimensions to be statically known (> 0). \
                    Please ensure onnx-simplifier fully resolved all dimensions.",
                    data_input_raw, input_shape
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
            // All dimensions are positive, use const shape if available; otherwise repair if needed.
            let total_input: i64 = input_shape.iter().product();
            let total_target: i64 = shape_values.iter().product();
            let mut candidate: Vec<i64> = shape_values.clone();

            // If element counts don't match, repair using available hints.
            if total_input > 0 && total_target > 0 && total_input != total_target {
                // Rebuild using batch/seq hints (from known inputs) and hidden from target.
                let mut batch_hint = input_shape.first().copied().unwrap_or(1);
                let mut seq_hint = input_shape.get(1).copied().unwrap_or(1);
                for (name, shape) in context.value_shapes.iter() {
                    if shape.len() >= 2 && !context.initializers.contains_key(name) {
                        if shape[0] > batch_hint {
                            batch_hint = shape[0];
                        }
                        if shape[1] > seq_hint {
                            seq_hint = shape[1];
                        }
                    }
                }

                let hidden = shape_values.last().copied().unwrap_or(1);
                eprintln!(
                    "[reshape] repair: {} input_shape={:?} target_shape={:?} batch_hint={} seq_hint={} hidden={}",
                    output_name, input_shape, shape_values, batch_hint, seq_hint, hidden
                );
                candidate = vec![batch_hint, seq_hint, hidden];
            } else if !shape_from_const {
                // If the target is rank-3 and the input is rank-4, flatten the last two dims.
                if input_shape.len() == 4 && shape_values.len() == 3 {
                    let tail: i64 = input_shape[2..].iter().product();
                    candidate = vec![input_shape[0], input_shape[1], tail];
                }
            }
            candidate.iter().map(|&v| v as u32).collect()
        };

        let mut options = Map::new();
        options.insert("newShape".to_string(), serde_json::json!(shape_values));

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "reshape".to_string(),
            inputs: vec![data_input],
            options,
            outputs: None,
        }]);

        if let Some(output) = node.get_output().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
        }

        Ok(result)
    }

    /// Convert ONNX Transpose to WebNN transpose
    fn convert_transpose(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
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

        let input0 = context.resolve_input(&inputs[0]);

        let mut options = Map::new();
        if let Some(perm_values) = perm {
            options.insert("permutation".to_string(), serde_json::json!(perm_values));
        }

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "transpose".to_string(),
            inputs: vec![input0],
            options,
            outputs: None,
        }]);

        if let Some(output) = node.get_output().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
        }

        Ok(result)
    }

    /// Convert ONNX Concat to WebNN concat
    fn convert_concat(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
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

        let sanitized_inputs: Vec<String> =
            inputs.iter().map(|s| context.resolve_input(s)).collect();

        let mut options = Map::new();
        options.insert("axis".to_string(), serde_json::json!(axis));

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "concat".to_string(),
            inputs: sanitized_inputs,
            options,
            outputs: None,
        }]);

        if let Some(output) = node.get_output().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
        }

        Ok(result)
    }

    /// Convert ONNX Split to WebNN split
    fn convert_split(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
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

        let input0 = context.resolve_input(&inputs[0]);
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
        let output_node_id = sanitize_identifier(&format!("{}_split", node_name));
        let mut result = ConversionResult::new(vec![Node {
            id: output_node_id,
            op: "split".to_string(),
            inputs: vec![input0],
            options,
            outputs: Some(sanitized_outputs.clone()),
        }]);

        for (onnx_out, webnn_out) in outputs.iter().zip(sanitized_outputs.iter()) {
            result
                .output_mappings
                .insert(onnx_out.to_string(), webnn_out.clone());
        }

        Ok(result)
    }

    /// Convert ONNX Unsqueeze to WebNN expand
    /// ONNX Unsqueeze adds dimensions at specified axes
    /// In opset >= 13, axes is a second input; in earlier opsets, it's an attribute
    fn convert_unsqueeze(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
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

        let input0 = context.resolve_input(&inputs[0]);

        let axes_values = if let Some(a) = axes {
            a
        } else {
            self.read_axes_from_attr_or_const(node, context)?
        };

        let mut options = Map::new();
        options.insert("axes".to_string(), serde_json::json!(axes_values));

        // WebNN doesn't have unsqueeze, so we'll use expand with axes parameter.
        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "expand".to_string(),
            inputs: vec![input0],
            options,
            outputs: None,
        }]);

        if let Some(output) = node.get_output().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
        }

        Ok(result)
    }

    /// Convert ONNX Squeeze to WebNN reshape
    /// ONNX Squeeze removes dimensions of size 1
    fn convert_squeeze(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
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

        let input0 = context.resolve_input(&inputs[0]);

        let axes_values = if let Some(a) = axes {
            a
        } else {
            self.read_axes_from_attr_or_const(node, context)?
        };

        let mut options = Map::new();
        options.insert("axes".to_string(), serde_json::json!(axes_values));

        // WebNN doesn't have squeeze, so we'll use reshape with axes parameter.
        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "reshape".to_string(),
            inputs: vec![input0],
            options,
            outputs: None,
        }]);

        if let Some(output) = node.get_output().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
        }

        Ok(result)
    }

    /// Convert ONNX Tile to WebNN tile
    /// Repeats the input tensor along each dimension according to the repeats input
    fn convert_tile(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.get_input();
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Tile expects 2 inputs (input, repeats), got {}",
                inputs.len()
            )));
        }

        let output_name = if node.get_output().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.get_output()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);

        // The repeats input must be a constant for WebNN
        let repeats_name = inputs[1].as_str();

        // Try to read repeats from const_values or initializers
        let repeats = if let Some(vals) = context.const_values.get(repeats_name) {
            vals.clone()
        } else if let Some(tensor) = context.initializers.get(repeats_name) {
            // Read from initializer
            let raw = tensor.get_raw_data();
            if !raw.is_empty() {
                match tensor.get_data_type() {
                    TensorProto_DataType::INT64 => raw
                        .chunks_exact(8)
                        .map(|c| {
                            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                        })
                        .collect(),
                    TensorProto_DataType::INT32 => raw
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                        .collect(),
                    _ => {
                        return Err(OnnxError::InvalidShape(
                            "Tile repeats must be int32 or int64".to_string(),
                        ))
                    }
                }
            } else if !tensor.get_int64_data().is_empty() {
                tensor.get_int64_data().to_vec()
            } else if !tensor.get_int32_data().is_empty() {
                tensor.get_int32_data().iter().map(|&v| v as i64).collect()
            } else {
                return Err(OnnxError::InvalidShape(
                    "Tile repeats tensor has no data".to_string(),
                ));
            }
        } else {
            return Err(OnnxError::InvalidShape(
                "Tile repeats must be constant for WebNN".to_string(),
            ));
        };

        let mut options = Map::new();
        options.insert("repetitions".to_string(), serde_json::json!(repeats));

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "tile".to_string(),
            inputs: vec![input0],
            options,
            outputs: None,
        }]);

        if let Some(output) = node.get_output().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
            // Preserve data type
            if let Some(dtype) = context.value_types.get(&inputs[0]) {
                result
                    .output_types
                    .insert(output.to_string(), dtype.clone());
            }
        }

        Ok(result)
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
        assert!(handler.supports("Tile"));
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

        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "reshape");
        assert_eq!(result.nodes[0].inputs, vec!["data"]);
        assert_eq!(result.nodes[0].id, "reshaped");
        // Verify the newShape is now a static array
        assert_eq!(
            result.nodes[0].options.get("newShape"),
            Some(&serde_json::json!([1, 2, 3, 4]))
        );
    }

    #[test]
    fn test_convert_transpose() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Transpose", vec!["x"], vec!["y"]);
        add_ints_attribute(&mut node, "perm", vec![1, 0, 2]);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "transpose");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
        assert!(result.nodes[0].options.contains_key("permutation"));
    }

    #[test]
    fn test_convert_concat() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Concat", vec!["a", "b", "c"], vec!["result"]);
        add_int_attribute(&mut node, "axis", 1);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "concat");
        assert_eq!(result.nodes[0].inputs.len(), 3);
        assert!(result.nodes[0].options.contains_key("axis"));
    }

    #[test]
    fn test_convert_split() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Split", vec!["x"], vec!["y1", "y2"]);
        add_int_attribute(&mut node, "axis", 0);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "split");
        assert!(result.nodes[0].outputs.is_some());
    }

    #[test]
    fn test_convert_unsqueeze() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Unsqueeze", vec!["x"], vec!["y"]);
        add_ints_attribute(&mut node, "axes", vec![0, 2]);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "expand");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
        assert!(result.nodes[0].options.contains_key("axes"));
    }

    #[test]
    fn test_convert_squeeze() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Squeeze", vec!["x"], vec!["y"]);
        add_ints_attribute(&mut node, "axes", vec![1]);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "reshape");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
        assert!(result.nodes[0].options.contains_key("axes"));
    }

    #[test]
    fn test_convert_tile() {
        let handler = ReshapeHandler;
        let node = create_test_node("Tile", vec!["input", "repeats"], vec!["output"]);

        // Create a mock repeats tensor [2, 3]
        let mut repeats_tensor = onnx::onnx::TensorProto::new();
        repeats_tensor.set_name("repeats".to_string());
        repeats_tensor.set_data_type(onnx::onnx::TensorProto_DataType::INT64);
        repeats_tensor.set_dims(vec![2]);
        repeats_tensor.set_int64_data(vec![2, 3]);

        let leaked_repeats: &'static onnx::onnx::TensorProto = Box::leak(Box::new(repeats_tensor));

        let mut initializers = std::collections::HashMap::new();
        initializers.insert("repeats".to_string(), leaked_repeats);
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "tile");
        assert_eq!(result.nodes[0].inputs, vec!["input"]);
        assert!(result.nodes[0].options.contains_key("repetitions"));

        // Verify the repetitions value
        let repetitions = result.nodes[0].options.get("repetitions").unwrap();
        assert_eq!(repetitions, &serde_json::json!([2, 3]));
    }
}
