#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! Data structures for representing and manipulating NMR peak shapes.

mod lorentzian;
pub use lorentzian::Lorentzian;

mod peak_shape_traits;
pub use peak_shape_traits::{Evaluate, PeakShape};

pub mod iter {
    //! Iterator adapter traits for peak shapes.

    pub use crate::peak_shape_traits::{EvaluateMap, Superposition, SuperpositionMap};
    #[cfg(feature = "rayon")]
    pub use crate::peak_shape_traits::{ParEvaluateMap, ParSuperposition, ParSuperpositionMap};
}
