// Re-export ONNX protos from webnn-onnx-utils to ensure type compatibility
pub mod onnx {
    pub use webnn_onnx_utils::protos::onnx::*;

    // Type alias for compatibility with old code expecting TensorProto_DataType
    pub use tensor_proto::DataType as TensorProto_DataType;
}
