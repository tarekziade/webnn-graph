// Operator handler trait and registry

use crate::ast::{ConstDecl, Node};
use crate::onnx::convert::OnnxError;
use onnx::onnx::{NodeProto, TensorProto};
use std::collections::HashMap;

pub mod activation;
pub mod comparison;
pub mod conversion;
pub mod elementwise;
pub mod matmul;
pub mod normalization;
pub mod reduction;
pub mod reshape;
pub mod utility;

use activation::ActivationHandler;
use comparison::ComparisonHandler;
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
    pub initializers: &'a HashMap<String, &'a TensorProto>,
    /// Map of value names to their shapes (for shape inference)
    pub value_shapes: &'a HashMap<String, Vec<i64>>,
    /// Map of value names to constant integer contents (for const folding)
    pub const_values: &'a HashMap<String, Vec<i64>>,
    /// Map of ONNX value names to WebNN value identifiers
    pub value_ids: &'a HashMap<String, String>,
    /// Map of value names to data types
    pub value_types: &'a HashMap<String, crate::ast::DataType>,
}

impl<'a> ConversionContext<'a> {
    pub fn resolve_input(&self, name: &str) -> String {
        if let Some(mapped) = self.value_ids.get(name) {
            return mapped.clone();
        }

        let sanitized = crate::onnx::convert::sanitize_identifier(name);
        if let Some(mapped) = self.value_ids.get(&sanitized) {
            return mapped.clone();
        }

        sanitized
    }
}

/// Results of converting a single ONNX node
pub struct ConversionResult {
    pub nodes: Vec<Node>,
    pub consts: Vec<(String, ConstDecl)>,
    /// ONNX output name -> WebNN value id
    pub output_mappings: HashMap<String, String>,
    /// ONNX output name -> data type
    pub output_types: HashMap<String, crate::ast::DataType>,
}

impl ConversionResult {
    pub fn new(nodes: Vec<Node>) -> Self {
        Self {
            nodes,
            consts: Vec::new(),
            output_mappings: HashMap::new(),
            output_types: HashMap::new(),
        }
    }
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
    ) -> Result<ConversionResult, OnnxError>;
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
            Box::new(ComparisonHandler),
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
    ) -> Result<ConversionResult, OnnxError> {
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
