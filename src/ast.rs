use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphJson {
    pub format: String, // "webnn-graph-json"
    pub version: u32,   // 1
    pub inputs: BTreeMap<String, OperandDesc>,
    #[serde(default)]
    pub consts: BTreeMap<String, ConstDecl>,
    pub nodes: Vec<Node>,
    // output_name -> value reference name
    pub outputs: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OperandDesc {
    #[serde(rename = "dataType")]
    pub data_type: DataType,
    pub shape: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DataType {
    #[serde(rename = "float32")]
    Float32,
    #[serde(rename = "float16")]
    Float16,
    #[serde(rename = "int32")]
    Int32,
    #[serde(rename = "uint32")]
    Uint32,
    #[serde(rename = "int64")]
    Int64,
    #[serde(rename = "uint64")]
    Uint64,
    #[serde(rename = "int8")]
    Int8,
    #[serde(rename = "uint8")]
    Uint8,
}

impl DataType {
    pub fn from_wg(s: &str) -> Option<Self> {
        match s {
            "f32" => Some(Self::Float32),
            "f16" => Some(Self::Float16),
            "i32" => Some(Self::Int32),
            "u32" => Some(Self::Uint32),
            "i64" => Some(Self::Int64),
            "u64" => Some(Self::Uint64),
            "i8" => Some(Self::Int8),
            "u8" => Some(Self::Uint8),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConstDecl {
    #[serde(rename = "dataType")]
    pub data_type: DataType,
    pub shape: Vec<u32>,
    pub init: ConstInit,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum ConstInit {
    Weights { r#ref: String },
    Scalar { value: serde_json::Value },
    InlineBytes { bytes: Vec<u8> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub op: String,
    pub inputs: Vec<String>,
    #[serde(default)]
    pub options: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub outputs: Option<Vec<String>>,
}

pub fn new_graph_json() -> GraphJson {
    GraphJson {
        format: "webnn-graph-json".to_string(),
        version: 1,
        inputs: BTreeMap::new(),
        consts: BTreeMap::new(),
        nodes: Vec::new(),
        outputs: BTreeMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datatype_from_wg() {
        assert_eq!(DataType::from_wg("f32"), Some(DataType::Float32));
        assert_eq!(DataType::from_wg("f16"), Some(DataType::Float16));
        assert_eq!(DataType::from_wg("i32"), Some(DataType::Int32));
        assert_eq!(DataType::from_wg("u32"), Some(DataType::Uint32));
        assert_eq!(DataType::from_wg("i64"), Some(DataType::Int64));
        assert_eq!(DataType::from_wg("u64"), Some(DataType::Uint64));
        assert_eq!(DataType::from_wg("i8"), Some(DataType::Int8));
        assert_eq!(DataType::from_wg("u8"), Some(DataType::Uint8));
        assert_eq!(DataType::from_wg("invalid"), None);
        assert_eq!(DataType::from_wg("float32"), None);
    }

    #[test]
    fn test_new_graph_json() {
        let graph = new_graph_json();
        assert_eq!(graph.format, "webnn-graph-json");
        assert_eq!(graph.version, 1);
        assert!(graph.inputs.is_empty());
        assert!(graph.consts.is_empty());
        assert!(graph.nodes.is_empty());
        assert!(graph.outputs.is_empty());
    }

    #[test]
    fn test_operand_desc_equality() {
        let desc1 = OperandDesc {
            data_type: DataType::Float32,
            shape: vec![1, 2, 3],
        };
        let desc2 = OperandDesc {
            data_type: DataType::Float32,
            shape: vec![1, 2, 3],
        };
        let desc3 = OperandDesc {
            data_type: DataType::Float16,
            shape: vec![1, 2, 3],
        };
        assert_eq!(desc1, desc2);
        assert_ne!(desc1, desc3);
    }

    #[test]
    fn test_const_init_variants() {
        let weights_init = ConstInit::Weights {
            r#ref: "W".to_string(),
        };
        let scalar_init = ConstInit::Scalar {
            value: serde_json::json!(1.0),
        };
        let bytes_init = ConstInit::InlineBytes {
            bytes: vec![1, 2, 3, 4],
        };

        // Test that they're different variants
        assert!(matches!(weights_init, ConstInit::Weights { .. }));
        assert!(matches!(scalar_init, ConstInit::Scalar { .. }));
        assert!(matches!(bytes_init, ConstInit::InlineBytes { .. }));
    }
}
