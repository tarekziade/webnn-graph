use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::ast::DataType;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightsManifest {
    pub format: String, // "wg-weights-manifest"
    pub version: u32,   // 1
    pub endianness: String,
    pub tensors: BTreeMap<String, TensorEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    #[serde(rename = "dataType")]
    pub data_type: DataType,
    pub shape: Vec<u32>,
    #[serde(rename = "byteOffset")]
    pub byte_offset: u64,
    #[serde(rename = "byteLength")]
    pub byte_length: u64,
    #[serde(default)]
    pub layout: Option<String>,
}

pub fn dtype_size(dt: &DataType) -> u64 {
    match dt {
        DataType::Float32 => 4,
        DataType::Float16 => 2,
        DataType::Int32 => 4,
        DataType::Uint32 => 4,
        DataType::Int64 => 8,
        DataType::Uint64 => 8,
        DataType::Int8 => 1,
        DataType::Uint8 => 1,
    }
}

pub fn numel(shape: &[u32]) -> u64 {
    shape
        .iter()
        .fold(1u64, |acc, &d| acc.saturating_mul(d as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(dtype_size(&DataType::Float32), 4);
        assert_eq!(dtype_size(&DataType::Float16), 2);
        assert_eq!(dtype_size(&DataType::Int32), 4);
        assert_eq!(dtype_size(&DataType::Uint32), 4);
        assert_eq!(dtype_size(&DataType::Int64), 8);
        assert_eq!(dtype_size(&DataType::Uint64), 8);
        assert_eq!(dtype_size(&DataType::Int8), 1);
        assert_eq!(dtype_size(&DataType::Uint8), 1);
    }

    #[test]
    fn test_numel() {
        assert_eq!(numel(&[]), 1);
        assert_eq!(numel(&[10]), 10);
        assert_eq!(numel(&[2, 3]), 6);
        assert_eq!(numel(&[2, 3, 4]), 24);
        assert_eq!(numel(&[1, 2048]), 2048);
        assert_eq!(numel(&[2048, 1000]), 2048000);
    }

    #[test]
    fn test_numel_large_values() {
        // Test with large values - saturating_mul ensures no panic on overflow
        // u32::MAX is 4294967295, and multiplying multiple of these as u64
        // should handle large numbers correctly
        let large_shape = vec![u32::MAX, u32::MAX];
        let result = numel(&large_shape);
        // The actual result is u32::MAX * u32::MAX = 18446744065119617025
        // This fits in u64, so saturating_mul doesn't trigger saturation
        assert_eq!(result, 18446744065119617025u64);

        // Test with even larger shape that would trigger saturation
        let very_large = vec![u32::MAX; 10];
        let result2 = numel(&very_large);
        // With 10 multiplications of u32::MAX, we will hit saturation
        assert_eq!(result2, u64::MAX);
    }
}
