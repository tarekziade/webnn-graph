// Constant evaluators for ONNX operations

mod concat;
mod constant;
mod constant_of_shape;
mod gather;
mod range;
mod reshape_ops;
mod shape;

pub use concat::ConcatEvaluator;
pub use constant::ConstantEvaluator as ConstantOpEvaluator;
pub use constant_of_shape::ConstantOfShapeEvaluator;
pub use gather::GatherEvaluator;
pub use range::RangeEvaluator;
pub use reshape_ops::{CastEvaluator, SqueezeEvaluator, UnsqueezeEvaluator};
pub use shape::ShapeEvaluator;

use crate::onnx::constant_folding::ConstantEvaluator;

/// Get all built-in evaluators
pub fn get_evaluators() -> Vec<Box<dyn ConstantEvaluator>> {
    vec![
        Box::new(ShapeEvaluator),
        Box::new(GatherEvaluator),
        Box::new(ConcatEvaluator),
        Box::new(UnsqueezeEvaluator),
        Box::new(SqueezeEvaluator),
        Box::new(CastEvaluator),
        Box::new(RangeEvaluator),
        Box::new(ConstantOfShapeEvaluator),
        Box::new(ConstantOpEvaluator),
    ]
}
