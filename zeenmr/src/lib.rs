#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! This crate provides data structures and algorithms for handling NMR spectra.
//!
//! # Feature flags
//!
//! All feature flags are additive, meaning that they can be combined. The
//! following feature flags are available:
//!
//! | Flag      | Description                                         | Dependencies         |
//! |-----------|---------------------------------------------------- |----------------------|
//! | `rayon`   | Parallel implementations of certain features.       | [Rayon]              |
//! | `serde`   | Serialization and deserialization.                  | [Serde]              |
//!
//! [Rayon]: https://docs.rs/rayon
//! [Serde]: https://docs.rs/serde

#[doc(inline)]
pub use zeenmr_spectrum as spectrum;

#[doc(inline)]
pub use zeenmr_parse as parse;

#[doc(inline)]
pub use zeenmr_peakshape as peak_shape;

#[doc(inline)]
pub use zeenmr_deconvolution as deconvolution;
