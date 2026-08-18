//! Types and traits for representing spectrum intensities.
//!
//! # Storage
//!
//! The array type owns shape, strides, and bounds checking. Storage only
//! holds the elements of the array.
//!
//! There are five general storage traits in this module, summarized in the
//! following table:
//!
//! | Trait               | Properties                                        |
//! |---------------------|---------------------------------------------------|
//! | [`Storage`]         | stable base pointer and length                    |
//! | [`RawStorage`]      | stable base pointer, generally not sound to deref |
//! | [`StorageMut`]      | stable length, writes observable from the others  |
//! | [`RawStorageMut`]   | generally not sound to deref                      |
//! | [`StorageOwned`]    | constructible from `Vec<T>` or iterators          |
//!
//! Accompanying them are implementations for standard library types commonly
//! used for heap allocated buffers, such as `Vec<T>` and `Box<[T]>`, the slice
//! types `[T]`, and two new pointer wrapper types implementing the raw storage
//! traits, [`Access`] and [`AccessMut`].
//!
//! ## Immutable
//!
//! The [`Storage`] trait is the general storage abstraction. Any type that can
//! provide a slice view to an underlying buffer can serve as array storage.
//! However, it must have one property:
//!
//! For any given value, every call to [`Storage::as_slice`] must return a slice
//! with the same base pointer and length, up until that value is modified other
//! than through a slice returned from it. For example, calling [`Vec::push`] or
//! [`Cow::to_mut`] may reallocate the underlying buffer. Returning a different
//! base pointer and length *after* this point is allowed.
//!
//! [`Cow::to_mut`]: std::borrow::Cow::to_mut
//!
//! [`RawStorage`] provides an abstraction for situations where returning a
//! slice is not possible, either because the storage type itself cannot do
//! that, or because it wouldn't be possible in some situation. Instead, it
//! only requires the type to be able to provide a base pointer to the
//! allocation. Dereferencing this base pointer is generally unsound, and only
//! some memory geometry paired with it can make it sound to dereference
//! *certain offsets* from it. For a type implementing both of these traits,
//! [`Storage::as_slice`] and [`RawStorage::as_ptr`] must return the same base
//! pointer.
//!
//! See the section on [Access Pointers](crate::intensity_array#access-pointers)
//! for how this is handled internally.
//!
//! ## Mutable
//!
//! The [`StorageMut`] and [`RawStorageMut`] traits are the mutable counterparts
//! to [`Storage`] and [`RawStorage`] respectively. They require slightly
//! different properties:
//!
//! The slice returned by [`StorageMut::as_mut_slice`] must have the same length
//! as the one that would have been returned by [`Storage::as_slice`]. In other
//! words, calling `as_mut_slice` (or `as_mut_ptr`) counts as a modification to
//! the storage, after which the returned base pointer needn't be the same as
//! returned by calls before this modification. However, the elements must still
//! compare equal to those accessible through `as_slice` or `as_ptr`. Further,
//! any writes through the returned slice or pointer must be observable from any
//! slice or pointer returned by the storage type.
//!
//! Other than that, what was previously described still applies.
//!
//! ## Access Pointers
//!
//! Suppose you have the following chunk of memory:
//!
//! ```text
//! ┌╌╌╌╌╌╌┬╌╌╌╌╌╌┬╌╌╌╌╌╌┬╌╌╌╌╌╌┐
//! │ 0x00 │ 0x01 │ 0x02 │ 0x03 │
//! └╌╌╌╌╌╌┴╌╌╌╌╌╌┴╌╌╌╌╌╌┴╌╌╌╌╌╌┘
//! ```
//!
//! We can treat this like a 2 x 2 matrix by, for example, treating the value at
//! `0x00` as the element at `(0, 0)`, `0x01` as the element at `(0, 1)`,
//! `0x02` as the element at `(1, 0)`, and `0x03` as the element at `(1, 1)`.
//! Here, the first index describes the row, the second describes the column:
//!
//! ```text
//!       0      1
//!   ┌╌╌╌╌╌╌┬╌╌╌╌╌╌┐
//! 0 │ 0x00 │ 0x01 │
//!   ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
//! 1 │ 0x02 │ 0x03 │
//!   └╌╌╌╌╌╌┴╌╌╌╌╌╌┘
//! ```
//!
//! Now, if we wanted to borrow the columns of this matrix mutably, we would
//! have a problem that cannot be expressed cleanly with slices. A column simply
//! isn't a contiguous run of elements, so no `&mut [T]` can name one without
//! also containing elements from the other column. The rows, by contrast, would
//! be very easy to produce, for example like in [`SplitMut`]. It would of
//! course be possible to borrow the entire buffer as a mutable slice and modify
//! the columns one at a time. That, however, still wouldn't give us a value to
//! yield from an iterator, and no way to send two disjoint columns across
//! different threads.
//!
//! [`SplitMut`]: std::slice::SplitMut
//!
//! The solution is to relax the notion of what we're borrowing from the
//! guarantees made by the slice types. This is the purpose of this module's
//! access pointers, [`Access`] and [`AccessMut`], which implement the
//! [`RawStorage`] (and [`RawStorageMut`]) trait(s).
//!
//! These pointer wrappers are essentially slices that do not make the claim
//! that it is sound to dereference every pointer offset within the allocation
//! they point to. They are always paired with some kind of memory geometry
//! that determines which part of the allocation they are allowed to observe or
//! mutate. In the example above, we could pair an access pointer with a base
//! offset, a stride, i.e., the number of elements to skip to find the next
//! element we're allowed to access, and the number of elements we can access
//! from it:
//!
//! ```text
//! col1 = { access: AccessMut(0x00), offset: 0, stride: 2, count: 2 }
//! col2 = { access: AccessMut(0x00), offset: 1, stride: 2, count: 2 }
//! ```
//!
//! In this way, `col1`, can only access `0x00` and `0x02`, while `col2`, can
//! only access `0x01` and `0x03`, meaning they can simultaneously provide
//! mutable references to the elements of columns without violating rust's
//! memory safety guarantees!
//!
//! # Layout
//!
//! WIP
//!
//! # Lanes
//!
//! WIP
//!
//! # Restrict & Crop
//!
//! WIP
//!
//! # NumPy Acknowledgement
//!
//! The multidimensional array implementation featured in this module is heavily
//! inspired by [NumPy]. While there are equivalent Rust crates, they, much like
//! numpy itself, support a lot of features we simply do not need and/or are
//! in their own unstable phase. Further, NMR and spectroscopy in general
//! have quite specific algorithmic needs, letting us aggressively optimize
//! the cases we care about while putting much less priority on some features
//! than a general purpose multidimensional array would have to.
//!
//! [NumPy]: https://numpy.org/

mod array;

pub use array::{
    Array, Array1D, Array2D, Array3D, ArrayArc, ArrayCow, ArrayDyn, ArrayOwned, ArrayRc, ArrayView,
    ArrayViewMut,
};

mod storage;
pub use storage::{
    Access, AccessMut, RawStorage, RawStorageMut, Storage, StorageMut, StorageOwned,
};

mod layout;
pub use layout::{
    ArrayIndex, DimOrder, DynArrayIndex, DynDimOrder, DynLayout, DynShape, DynStrides,
    LaneGeometry, Layout, Shape, StaticArrayIndex, StaticDimOrder, StaticLayout, StaticShape,
    StaticStrides, Strides, index, order, shape, strides,
};

mod lane;
pub use lane::{Lane, LaneMut};

pub mod diagnostic_1d;

pub mod iter;
