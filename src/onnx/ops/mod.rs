// Operator handler trait and registry

use crate::ast::Node;
use crate::onnx::convert::OnnxError;
use onnx::onnx::{NodeProto, TensorProto};
use std::collections::HashMap;

pub mod activation;
pub mod conversion;
pub mod elementwise;
pub mod matmul;
pub mod normalization;
pub mod reduction;
pub mod reshape;
pub mod utility;

use activation::ActivationHandler;
use conversion::ConversionHandler;
use elementwise::ElementwiseHandler;
use matmul::MatMulHandler;
use normalization::NormalizationHandler;
use reduction::ReductionHandler;
use reshape::ReshapeHandler;
use utility::UtilityHandler;

/// Context for operator conversion
pub struct ConversionContext<'a> {
    /// Map of initializer names to TensorProto (for resolving constant shapes)
    pub initializers: HashMap<String, &'a TensorProto>,
    /// Map of value names to their shapes (for shape inference)
    pub value_shapes: HashMap<String, Vec<i64>>,
}

/// Trait for handling ONNX operator conversion
pub trait OpHandler {
    /// Check if this handler supports the given operator type
    fn supports(&self, op_type: &str) -> bool;

    /// Convert an ONNX node to WebNN node(s)
    fn convert<'a>(
        &self,
        node: &NodeProto,
        context: &ConversionContext<'a>,
    ) -> Result<Vec<Node>, OnnxError>;
}

/// Registry for operator handlers
pub struct OpRegistry {
    handlers: Vec<Box<dyn OpHandler>>,
}

impl OpRegistry {
    /// Create a new operator registry with all handlers
    pub fn new() -> Self {
        let handlers: Vec<Box<dyn OpHandler>> = vec![
            Box::new(MatMulHandler),
            Box::new(ElementwiseHandler),
            Box::new(NormalizationHandler),
            Box::new(ReshapeHandler),
            Box::new(ConversionHandler),
            Box::new(UtilityHandler),
            Box::new(ReductionHandler),
            Box::new(ActivationHandler),
        ];

        OpRegistry { handlers }
    }

    /// Convert an ONNX node using the appropriate handler
    pub fn convert_node<'a>(
        &self,
        node: &NodeProto,
        context: &ConversionContext<'a>,
    ) -> Result<Vec<Node>, OnnxError> {
        let op_type = node.get_op_type();

        for handler in &self.handlers {
            if handler.supports(op_type) {
                return handler.convert(node, context);
            }
        }

        // No handler found
        let node_name = if node.has_name() {
            node.get_name().to_string()
        } else {
            "<unnamed>".to_string()
        };

        Err(OnnxError::UnsupportedOp {
            op: op_type.to_string(),
            node: node_name,
        })
    }
}

impl Default for OpRegistry {
    fn default() -> Self {
        Self::new()
    }
}
