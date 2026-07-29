//! Types and traits for representing spectrum intensities.

mod storage;
pub use storage::{Storage, StorageMut, StorageOwned};

mod dimension;
pub use dimension::{DimIndex, Dimension, StaticDim, DynDim, Shape, ArrayIndex, Strides, Layout};

pub mod diagnostic_1d;
