// MatMul and Gemm operator handlers

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, OpHandler};
use onnx::onnx::NodeProto;
use serde_json::Map;

pub struct MatMulHandler;

impl OpHandler for MatMulHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type, "MatMul" | "Gemm")
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
            "MatMul" => self.convert_matmul(node, &node_name),
            "Gemm" => self.convert_gemm(node, &node_name),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

impl MatMulHandler {
    /// Convert ONNX MatMul to WebNN matmul
    /// MatMul: Y = A @ B
    fn convert_matmul(&self, node: &NodeProto, node_name: &str) -> Result<Vec<Node>, OnnxError> {
        let inputs = node.get_input();
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidShape(format!(
                "MatMul expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let output_name = if node.get_output().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.get_output()[0].to_string())
        };

        let input0 = sanitize_identifier(&inputs[0].to_string());
        let input1 = sanitize_identifier(&inputs[1].to_string());

        Ok(vec![Node {
            id: output_name.clone(),
            op: "matmul".to_string(),
            inputs: vec![input0, input1],
            options: Map::new(),
            outputs: None,
        }])
    }

    /// Convert ONNX Gemm to WebNN operations
    /// Gemm: Y = alpha * A' @ B' + beta * C
    /// where A' = transpose(A) if transA else A
    fn convert_gemm(&self, node: &NodeProto, node_name: &str) -> Result<Vec<Node>, OnnxError> {
        let inputs = node.get_input();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Gemm expects at least 2 inputs, got {}",
                inputs.len()
            )));
        }

        // Extract attributes
        let mut alpha = 1.0f32;
        let mut beta = 1.0f32;
        let mut trans_a = false;
        let mut trans_b = false;

        for attr in node.get_attribute() {
            match attr.get_name() {
                "alpha" => {
                    if attr.has_f() {
                        alpha = attr.get_f();
                    }
                }
                "beta" => {
                    if attr.has_f() {
                        beta = attr.get_f();
                    }
                }
                "transA" => {
                    if attr.has_i() {
                        trans_a = attr.get_i() != 0;
                    }
                }
                "transB" => {
                    if attr.has_i() {
                        trans_b = attr.get_i() != 0;
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
        let input1 = sanitize_identifier(&inputs[1].to_string());
        let input2 = if inputs.len() > 2 {
            sanitize_identifier(&inputs[2].to_string())
        } else {
            String::new()
        };

        let mut nodes = Vec::new();
        let mut current_result = format!("{}_matmul", node_name);

        // Handle transpose A if needed
        let input_a = if trans_a {
            let trans_a_name = format!("{}_transposeA", node_name);
            nodes.push(Node {
                id: trans_a_name.clone(),
                op: "transpose".to_string(),
                inputs: vec![input0.clone()],
                options: {
                    let mut opts = Map::new();
                    opts.insert(
                        "permutation".to_string(),
                        serde_json::json!([1, 0]), // Swap last two dimensions
                    );
                    opts
                },
                outputs: None,
            });
            trans_a_name
        } else {
            input0.clone()
        };

        // Handle transpose B if needed
        let input_b = if trans_b {
            let trans_b_name = format!("{}_transposeB", node_name);
            nodes.push(Node {
                id: trans_b_name.clone(),
                op: "transpose".to_string(),
                inputs: vec![input1.clone()],
                options: {
                    let mut opts = Map::new();
                    opts.insert(
                        "permutation".to_string(),
                        serde_json::json!([1, 0]), // Swap last two dimensions
                    );
                    opts
                },
                outputs: None,
            });
            trans_b_name
        } else {
            input1.clone()
        };

        // MatMul: A @ B
        nodes.push(Node {
            id: current_result.clone(),
            op: "matmul".to_string(),
            inputs: vec![input_a, input_b],
            options: Map::new(),
            outputs: None,
        });

        // Scale by alpha if not 1.0
        if (alpha - 1.0).abs() > f32::EPSILON {
            let scaled = format!("{}_scaled", node_name);
            nodes.push(Node {
                id: scaled.clone(),
                op: "mul".to_string(),
                inputs: vec![current_result.clone(), format!("{}_alpha", node_name)],
                options: Map::new(),
                outputs: None,
            });
            current_result = scaled;
            // Note: alpha constant would need to be added to the graph's consts
        }

        // Add beta * C if C is provided
        if inputs.len() > 2 {
            let bias_input = if (beta - 1.0).abs() > f32::EPSILON {
                // Scale C by beta
                let scaled_c = format!("{}_scaled_c", node_name);
                nodes.push(Node {
                    id: scaled_c.clone(),
                    op: "mul".to_string(),
                    inputs: vec![input2.clone(), format!("{}_beta", node_name)],
                    options: Map::new(),
                    outputs: None,
                });
                scaled_c
            } else {
                input2.clone()
            };

            // Add the bias
            nodes.push(Node {
                id: output_name.clone(),
                op: "add".to_string(),
                inputs: vec![current_result, bias_input],
                options: Map::new(),
                outputs: None,
            });
        } else {
            // No bias, just rename the final result
            if current_result != output_name {
                // If we have intermediate nodes, the last one should output to output_name
                if let Some(last_node) = nodes.last_mut() {
                    last_node.id = output_name.clone();
                }
            }
        }

        Ok(nodes)
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
    fn test_matmul_handler_supports() {
        let handler = MatMulHandler;
        assert!(handler.supports("MatMul"));
        assert!(handler.supports("Gemm"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_matmul() {
        let handler = MatMulHandler;
        let node = create_test_node("MatMul", vec!["a", "b"], vec!["c"]);
        let context = ConversionContext {
            initializers: std::collections::HashMap::new(),
            value_shapes: std::collections::HashMap::new(),
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "matmul");
        assert_eq!(result[0].inputs, vec!["a", "b"]);
        assert_eq!(result[0].id, "c");
    }

    #[test]
    fn test_convert_gemm_simple() {
        let handler = MatMulHandler;
        let node = create_test_node("Gemm", vec!["a", "b"], vec!["c"]);
        let context = ConversionContext {
            initializers: std::collections::HashMap::new(),
            value_shapes: std::collections::HashMap::new(),
        };

        let result = handler.convert(&node, &context).unwrap();
        // Simple Gemm without C, alpha=1, beta=1 should produce just matmul
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "matmul");
    }
}
