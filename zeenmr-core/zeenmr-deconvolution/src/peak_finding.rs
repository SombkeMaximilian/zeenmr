//! Peak finding algorithms for the deconvolution process.

pub mod error {
    //! Peak finding error types.

    pub use crate::peak_finding::curvature_analysis::{CurvatureError, CurvatureErrorKind};
}

mod curvature_analysis;
pub use curvature_analysis::CurvatureAnalysis;

mod find;
pub use find::{Find, Peak};
