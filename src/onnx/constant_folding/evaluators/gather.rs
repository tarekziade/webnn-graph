// Gather operation evaluator
// Gathers elements from a data tensor at indices

use crate::onnx::constant_folding::{
    ConstantEvaluator as EvaluatorTrait, ConstantFoldingContext, ConstantTensor, TensorData,
};
use crate::onnx::convert::OnnxError;
use crate::protos::onnx::NodeProto;

pub struct GatherEvaluator;

impl GatherEvaluator {
    /// Gather elements from data along specified axis using indices
    fn gather_data(
        data: &TensorData,
        data_shape: &[i64],
        indices: &[i64],
        axis: i64,
    ) -> Result<(TensorData, Vec<i64>), OnnxError> {
        // Normalize axis
        let normalized_axis = if axis < 0 {
            (data_shape.len() as i64 + axis) as usize
        } else {
            axis as usize
        };

        if normalized_axis >= data_shape.len() {
            return Err(OnnxError::InvalidShape(format!(
                "Gather axis {} out of bounds for shape {:?}",
                axis, data_shape
            )));
        }

        // For now, only support axis=0 which is the most common case for shape operations
        if normalized_axis != 0 {
            return Err(OnnxError::UnsupportedOp {
                op: "Gather".to_string(),
                node: format!("axis={} (only axis=0 is currently supported)", axis),
            });
        }

        // For axis=0, we're selecting entire slices along the first dimension
        // This is commonly used for extracting specific shape dimensions
        match data {
            TensorData::Int64(values) => {
                let mut gathered = Vec::new();
                for &idx in indices {
                    let i = if idx < 0 {
                        (values.len() as i64 + idx) as usize
                    } else {
                        idx as usize
                    };
                    if i >= values.len() {
                        return Err(OnnxError::InvalidShape(format!(
                            "Gather index {} out of bounds for data length {}",
                            idx,
                            values.len()
                        )));
                    }
                    gathered.push(values[i]);
                }

                // Output shape: replace axis dimension with indices shape
                let mut output_shape = Vec::new();
                for (i, &dim) in data_shape.iter().enumerate() {
                    if i == normalized_axis {
                        // Replace with indices dimensions
                        if indices.len() > 1 {
                            output_shape.push(indices.len() as i64);
                        }
                        // If indices is scalar (len=1), we effectively squeeze this dim
                    } else {
                        output_shape.push(dim);
                    }
                }

                // If indices is scalar and data is 1D, output is scalar (empty shape)
                if indices.len() == 1 && data_shape.len() == 1 {
                    output_shape.clear();
                }

                Ok((TensorData::Int64(gathered), output_shape))
            }
            TensorData::Int32(values) => {
                let mut gathered = Vec::new();
                for &idx in indices {
                    let i = if idx < 0 {
                        (values.len() as i64 + idx) as usize
                    } else {
                        idx as usize
                    };
                    if i >= values.len() {
                        return Err(OnnxError::InvalidShape(format!(
                            "Gather index {} out of bounds for data length {}",
                            idx,
                            values.len()
                        )));
                    }
                    gathered.push(values[i]);
                }

                let mut output_shape = Vec::new();
                for (i, &dim) in data_shape.iter().enumerate() {
                    if i == normalized_axis {
                        if indices.len() > 1 {
                            output_shape.push(indices.len() as i64);
                        }
                    } else {
                        output_shape.push(dim);
                    }
                }

                if indices.len() == 1 && data_shape.len() == 1 {
                    output_shape.clear();
                }

                Ok((TensorData::Int32(gathered), output_shape))
            }
            TensorData::Float32(values) => {
                let mut gathered = Vec::new();
                for &idx in indices {
                    let i = if idx < 0 {
                        (values.len() as i64 + idx) as usize
                    } else {
                        idx as usize
                    };
                    if i >= values.len() {
                        return Err(OnnxError::InvalidShape(format!(
                            "Gather index {} out of bounds for data length {}",
                            idx,
                            values.len()
                        )));
                    }
                    gathered.push(values[i]);
                }

                let mut output_shape = Vec::new();
                for (i, &dim) in data_shape.iter().enumerate() {
                    if i == normalized_axis {
                        if indices.len() > 1 {
                            output_shape.push(indices.len() as i64);
                        }
                    } else {
                        output_shape.push(dim);
                    }
                }

                if indices.len() == 1 && data_shape.len() == 1 {
                    output_shape.clear();
                }

                Ok((TensorData::Float32(gathered), output_shape))
            }
            _ => Err(OnnxError::UnsupportedOp {
                op: "Gather".to_string(),
                node: format!("data type {:?} not supported", data.data_type()),
            }),
        }
    }

    /// Get indices as int64 vector
    fn get_indices(tensor: &ConstantTensor) -> Result<Vec<i64>, OnnxError> {
        match &tensor.data {
            TensorData::Int64(v) => Ok(v.clone()),
            TensorData::Int32(v) => Ok(v.iter().map(|&x| x as i64).collect()),
            _ => Err(OnnxError::InvalidShape(format!(
                "Gather indices must be integer type, got {:?}",
                tensor.data_type
            ))),
        }
    }
}

impl EvaluatorTrait for GatherEvaluator {
    fn op_type(&self) -> &str {
        "Gather"
    }

    fn can_evaluate(&self, node: &NodeProto, ctx: &ConstantFoldingContext) -> bool {
        if node.op_type.as_str() != "Gather" {
            return false;
        }

        // Need both data and indices to be constants
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return false;
        }

        ctx.is_constant(inputs[0].as_str()) && ctx.is_constant(inputs[1].as_str())
    }

    fn evaluate(
        &self,
        node: &NodeProto,
        ctx: &ConstantFoldingContext,
    ) -> Result<Vec<ConstantTensor>, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return Err(OnnxError::MissingAttribute {
                attr: "inputs".to_string(),
                op: "Gather".to_string(),
            });
        }

        let data_tensor = ctx.get_constant(inputs[0].as_str()).ok_or_else(|| {
            OnnxError::ShapeInference(format!("Data tensor '{}' not found", inputs[0]))
        })?;

        let indices_tensor = ctx.get_constant(inputs[1].as_str()).ok_or_else(|| {
            OnnxError::ShapeInference(format!("Indices tensor '{}' not found", inputs[1]))
        })?;

        // Get axis attribute (default 0)
        let axis = node
            .attribute
            .as_slice()
            .iter()
            .find(|a| a.name.as_str() == "axis")
            .map(|a| a.i)
            .unwrap_or(0);

        // Convert indices to int64
        let indices = Self::get_indices(indices_tensor)?;

        // Perform gather
        let (output_data, output_shape) =
            Self::gather_data(&data_tensor.data, &data_tensor.shape, &indices, axis)?;

        let output = ConstantTensor {
            data: output_data.clone(),
            shape: output_shape,
            data_type: output_data.data_type().into(),
        };

        Ok(vec![output])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::{NodeProto, TensorProto, TensorProto_DataType};
    use std::collections::HashMap;

    #[test]
    fn test_gather_shape_dimensions() {
        // Test gathering specific dimensions from a shape tensor
        // This is the common pattern: Shape → Gather

        // Create context with a shape tensor [2, 128, 384]
        let shape_tensor = TensorProto {
            name: "shape".to_string(),
            data_type: TensorProto_DataType::Int64.into(),
            dims: vec![3],
            int64_data: vec![2, 128, 384],
            ..Default::default()
        };

        let shape_tensor_static: &'static TensorProto = Box::leak(Box::new(shape_tensor));

        // Create indices tensor [0, 1]  (gather first two dimensions)
        let indices_tensor = TensorProto {
            name: "indices".to_string(),
            data_type: TensorProto_DataType::Int64.into(),
            dims: vec![2],
            int64_data: vec![0, 1],
            ..Default::default()
        };

        let indices_tensor_static: &'static TensorProto = Box::leak(Box::new(indices_tensor));

        let mut init_map = HashMap::new();
        init_map.insert("shape".to_string(), shape_tensor_static);
        init_map.insert("indices".to_string(), indices_tensor_static);

        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = GatherEvaluator;

        // Create Gather node
        let node = NodeProto {
            op_type: "Gather".to_string(),
            input: vec!["shape".to_string(), "indices".to_string()],
            output: vec!["gathered".to_string()],
            ..Default::default()
        };

        // Evaluate
        assert!(evaluator.can_evaluate(&node, &ctx));
        let result = evaluator.evaluate(&node, &ctx).unwrap();

        assert_eq!(result.len(), 1);
        let output = &result[0];

        if let TensorData::Int64(ref values) = output.data {
            assert_eq!(values, &vec![2, 128]);
        } else {
            panic!("Expected Int64 data");
        }

        assert_eq!(output.shape, vec![2]);
    }

    #[test]
    fn test_gather_scalar_index() {
        // Test gathering a single element (scalar result)

        let data_tensor = TensorProto {
            name: "data".to_string(),
            data_type: TensorProto_DataType::Int64.into(),
            dims: vec![4],
            int64_data: vec![10, 20, 30, 40],
            ..Default::default()
        };

        let data_tensor_static: &'static TensorProto = Box::leak(Box::new(data_tensor));

        let index_tensor = TensorProto {
            name: "index".to_string(),
            data_type: TensorProto_DataType::Int64.into(),
            dims: vec![],        // Scalar
            int64_data: vec![2], // Get index 2 (value 30)
            ..Default::default()
        };

        let index_tensor_static: &'static TensorProto = Box::leak(Box::new(index_tensor));

        let mut init_map = HashMap::new();
        init_map.insert("data".to_string(), data_tensor_static);
        init_map.insert("index".to_string(), index_tensor_static);

        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = GatherEvaluator;

        let node = NodeProto {
            op_type: "Gather".to_string(),
            input: vec!["data".to_string(), "index".to_string()],
            output: vec!["result".to_string()],
            ..Default::default()
        };

        let result = evaluator.evaluate(&node, &ctx).unwrap();
        let output = &result[0];

        if let TensorData::Int64(ref values) = output.data {
            assert_eq!(values, &vec![30]);
        } else {
            panic!("Expected Int64 data");
        }

        assert_eq!(output.shape, Vec::<i64>::new()); // Scalar output
    }
}
