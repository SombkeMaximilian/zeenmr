//! Fitting algorithms for peak shapes in the deconvolution process.

mod fit;
pub use fit::Fit;
#[cfg(feature = "rayon")]
pub use fit::ParFit;

mod three_point;
pub use three_point::{ThreePoint, ThreePointStencil};
