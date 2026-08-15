//! Types and traits for representing spectrum intensities.
//!
//! # Design & Acknowledgement
//!
//! The multidimensional array implementation featured in this module is heavily
//! inspired by [numpy]. While there are equivalent Rust crates, they, much like
//! numpy itself, support a lot of features we simply do not need and/or are
//! in their own unstable phase. Further, NMR and spectroscopy in general
//! have quite specific algorithmic needs, letting us aggressively optimize
//! the cases we care about while putting much less priority on some features
//! that a general purpose multidimensional array implementation would have to
//! worry about.
//!
//! [numpy]: https://numpy.org/

mod array;
pub use array::{
    Array, Array1D, Array2D, Array3D, ArrayArc, ArrayCow, ArrayDyn, ArrayOwned, ArrayRc, ArrayView,
    ArrayViewMut,
};

mod storage;
pub use storage::{Storage, StorageMut, StorageOwned};

mod layout;
pub use layout::{ArrayIndex, DimOrder, LaneGeometry, Layout, Shape, Strides};

mod lane;
pub use lane::{Lane, LaneMut};

pub mod diagnostic_1d;

pub mod iter;
