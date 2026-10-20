//! Fitting algorithms for peak shapes in the deconvolution process.

mod fit;
pub use fit::{Fit, FitModel, IterativeFit};
#[cfg(feature = "rayon")]
pub use fit::{ParFit, ParFitModel, ParIterativeFit};

mod three_point;
pub use three_point::{ThreePoint, ThreePointModel, ThreePointStencil};
