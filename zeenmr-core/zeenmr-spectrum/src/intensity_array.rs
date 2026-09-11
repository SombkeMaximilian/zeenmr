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
//! The second component of the array type is the information necessary for
//! properly interpreting the storage described above: its [`Layout`]. It
//! consists of a [`Shape`], the [`Strides`], and an offset.
//!
//! The layout's shape specifies how far the array extends in each dimension.
//! The 2 x 2 matrix above has shape the `[2, 2]`. Each row has two elements,
//! and so does each column. The product of the extents is the number of
//! elements the layout addresses; for a shape of `[2, 2]`, this would be four.
//! Should any extent be `0`, it would leave the layout with nothing to address
//! irrespective of its other extents. This is almost always not something we
//! want, and any attempt to create such a layout results in a `None` value.
//!
//! The strides describe the distance in the storage between adjacent elements
//! along a given dimension. Moving from one column to the other in the matrix
//! above means moving one element along in memory in the corresponding
//! direction. Moving from one row to the other requires skipping one element,
//! i.e., moving in steps of two. Its strides are therefore `[2, 1]`. A layout
//! whose rows are contiguous chunks of memory like this are commonly called
//! "row-major" or "C order", with its last dimension varying fastest.
//! Reversing the roles, `[1, 2]` would make the columns contiguous instead,
//! which is "column-major" or "FORTRAN order", with the first dimension varying
//! fastest.
//!
//! Using the shape, we can map the elements using multidimensional indices
//! ([`ArrayIndex`]), where each component must be less than the corresponding
//! extent. The position of an element in the array storage is the sum of
//! component-wise index-stride product. The element at the index `(1, 0)` of
//! our 2 x 2 matrix can therefore be found at `1 * 2 + 0 * 1 = 2`, or `0x02`.
//!
//! Finally, the offset completes the picture by marking where the layout
//! begins. It is added to every linear offset, which lets a layout map any
//! region that starts partway through the storage. For example, we could map
//! only the second row of the 2 x 2 matrix, using a shape of `[1, 2]`, the
//! same strides as before, and an offset of `2`. The resulting layout addresses
//! only `0x02` and `0x03` without reallocating, moving, or copying.
//!
//! # Lanes
//!
//! Element access by index, as described above, is straightforward but
//! potentially expensive. For an array with `n` dimensions (rank `n`),
//! reaching a single element means naming an index along every dimension and
//! computing a sum of `n` products. This is a lot of work for what is
//! ultimately one read or write, and constructing the index is not free either.
//! While this may be acceptable for individual reads and writes, it scales
//! very poorly for bulk access.
//!
//! The access pattern NMR data processing actually calls for is iteration
//! along a dimension, most often the direct one. Instead of asking for the
//! element at `(x, y, z)`, fix every dimension but one and traverse everything
//! along the one that remains: a [`Lane`]. This is familiar one-dimensional
//! iteration, and because the stride along a dimension is fixed, consecutive
//! elements of a lane are a constant distance apart. In the contiguous case,
//! they are directly adjacent, and the lane is a plain slice. Stepping to the
//! next lane costs work proportional to `n` once per lane, rather than per
//! element.
//!
//! Fixing all dimensions but `dim` leaves an index over the others, and that
//! index is what names a lane. Enumerating the lanes therefore means counting
//! through those remaining dimensions, and the [`DimOrder`] decides their
//! ordering. For an array of shape `[2, 3, 4]` with lanes along dimension `2`,
//! the remaining dimensions are `0` and `1`, yielding 6 lanes of length 4.
//! Ordering them `[0, 1, 2]` makes dimension `1` vary the fastest, so the
//! lanes are yielded in the order:
//!
//! ```text
//! L(0) = (0, 0, ·)
//! L(1) = (0, 1, ·)
//! L(2) = (0, 2, ·)
//! L(3) = (1, 0, ·)
//! L(4) = (1, 1, ·)
//! L(5) = (1, 2, ·)
//! ```
//!
//! while the ordering `[1, 0, 2]` makes dimension `0` vary the fastest:
//!
//! ```text
//! L(0) = (0, 0, ·)
//! L(1) = (1, 0, ·)
//! L(2) = (0, 1, ·)
//! L(3) = (1, 1, ·)
//! L(4) = (0, 2, ·)
//! L(5) = (1, 2, ·)
//! ```
//!
//! Note that `dim` still needs to be part of the dimension ordering to make it
//! a valid permutation, though its position in the permutation has no effect
//! on the result. `[0, 1, 2]`, `[0, 2, 1]`, and `[2, 0, 1]` all produce the
//! same lane ordering along dimension `2`.
//!
//! The following table provides an overview of the `lanes_*` methods of the
//! array type. Each method also has a mutable and a parallel counterpart.
//!
//! | Method                  | Returns the array's lanes along...                        |
//! |-------------------------|-----------------------------------------------------------|
//! | [`contiguous_lanes`]    | a dimension with stride `1`, or `None` if there is none   |
//! | [`lanes_memory_order`]  | `dim`, traversing the storage as sequentially as possible |
//! | [`lanes_lexicographic`] | `dim`, with the last dimension counting fastest           |
//! | [`lanes_with_order`]    | `dim`, in the provided `order`                            |
//!
//! [`lanes_with_order`]: Array::lanes_with_order
//! [`lanes_memory_order`]: Array::lanes_memory_order
//! [`lanes_lexicographic`]: Array::lanes_lexicographic
//! [`contiguous_lanes`]: Array::contiguous_lanes
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
    Array, Array1D, Array2D, Array3D, ArrayArc, ArrayCow, ArrayDyn, ArrayOwned, ArrayRc,
    ArraySliceView, ArraySliceViewMut, ArrayView, ArrayViewMut,
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
