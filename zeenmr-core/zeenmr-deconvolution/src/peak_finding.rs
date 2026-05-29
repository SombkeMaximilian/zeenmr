//! Peak finding algorithms for the deconvolution process.

mod curvature_analysis;
pub use curvature_analysis::CurvatureAnalysis;

mod find;
pub use find::Find;

mod peak;
pub use peak::Peak;
