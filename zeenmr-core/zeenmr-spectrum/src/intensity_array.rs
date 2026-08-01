//! Types and traits for representing spectrum intensities.

mod storage;
pub use storage::{Storage, StorageMut, StorageOwned};

mod dimension;
pub use dimension::{ArrayIndex, DimIndex, Dimension, DynDim, Lane, Layout, Shape, StaticDim, Strides};

pub mod diagnostic_1d;

pub mod iter;
