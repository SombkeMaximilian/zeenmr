//! Peak finding algorithms for the deconvolution process.

mod curvature_analysis;
pub use curvature_analysis::CurvatureAnalysis;

mod find_peaks;
pub use find_peaks::FindPeaks;

mod peak;
pub use peak::Peak;
