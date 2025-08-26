#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! Data structures for representing and manipulating NMR peak shapes.

mod peak_shape_traits;
pub use peak_shape_traits::{Evaluate, EvaluateMap, PeakShape, Superposition, SuperpositionMap};
#[cfg(feature = "rayon")]
pub use peak_shape_traits::{ParEvaluateMap, ParSuperposition, ParSuperpositionMap};
