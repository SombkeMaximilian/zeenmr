//! Fitting algorithms for peak shapes in the deconvolution process.

mod fit;
pub use fit::Fit;

mod iterative_refinement;
pub use iterative_refinement::{IterativeRefinement, ThreePointStencil};
