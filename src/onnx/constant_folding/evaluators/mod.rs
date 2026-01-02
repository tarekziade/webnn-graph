// Constant evaluators for ONNX operations

mod concat;
mod constant;
mod gather;
mod reshape_ops;
mod shape;

pub use concat::ConcatEvaluator;
pub use constant::ConstantEvaluator as ConstantOpEvaluator;
pub use gather::GatherEvaluator;
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
        Box::new(ConstantOpEvaluator),
    ]
}
