//! Types and traits for representing spectrum intensities.

mod array;
pub use array::{
    Array, Array1D, Array2D, Array3D, ArrayArc, ArrayCow, ArrayDyn, ArrayOwned, ArrayRc, ArrayView,
    ArrayViewMut,
};

mod storage;
pub use storage::{Storage, StorageMut, StorageOwned};

mod dimension;
pub use dimension::{
    ArrayIndex, DimIndex, DimOrder, Dimension, DynDim, Lane, Layout, Shape, StaticDim, Strides,
};

pub mod diagnostic_1d;

pub mod iter;
