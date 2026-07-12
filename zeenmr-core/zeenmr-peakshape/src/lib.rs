#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Data structures for representing and manipulating NMR peak shapes.

mod peak_shape;
pub use peak_shape::{FromArray, PeakShape};

mod evaluate;
pub use evaluate::Evaluate;

pub mod iter {
    //! Iterator adapter traits for peak shapes.

    pub use crate::evaluate::{EvaluateMap, Superposition, SuperpositionMap};
    #[cfg(feature = "rayon")]
    pub use crate::evaluate::{ParEvaluateMap, ParSuperposition, ParSuperpositionMap};
}

mod lorentzian;
pub use lorentzian::Lorentzian;

mod gaussian;
pub use gaussian::Gaussian;

mod voigt;
pub use voigt::Voigt;
