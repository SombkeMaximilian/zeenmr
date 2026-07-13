#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Data structures for representing and manipulating NMR peak shapes.

mod peak_shape;
pub use peak_shape::{FromArray, PeakShape};

mod evaluate;
pub use evaluate::{Evaluate, EvaluateParts};

mod batch_superposition;
pub use batch_superposition::{BatchSuperposition, FuseWidth, FusedBatchSuperposition, Strategy};
#[cfg(feature = "rayon")]
pub use batch_superposition::{ParBatchSuperposition, ParFusedBatchSuperposition};

pub mod iter {
    //! Iterator adapter traits for peak shapes.

    pub use crate::evaluate::{EvaluateMap, FusedSuperposition, Superposition};
    #[cfg(feature = "rayon")]
    pub use crate::evaluate::{ParEvaluateMap, ParSuperposition};
}

mod lorentzian;
pub use lorentzian::Lorentzian;

mod gaussian;
pub use gaussian::Gaussian;

mod voigt;
pub use voigt::Voigt;

mod util;
