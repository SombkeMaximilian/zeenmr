//! Types and traits for representing spectrum intensities.

mod array;
pub use array::{
    Array, Array1D, Array2D, Array3D, ArrayArc, ArrayCow, ArrayDyn, ArrayOwned, ArrayRc, ArrayView,
    ArrayViewMut, Lane, LaneMut,
};

mod storage;
pub use storage::{Storage, StorageMut, StorageOwned};

mod dimension;
pub use dimension::{
    ArrayIndex, DimIndex, DimOrder, Dimension, DynDim, LaneGeometry, Layout, Shape, StaticDim,
    Strides,
};

pub mod diagnostic_1d;

pub mod iter;
