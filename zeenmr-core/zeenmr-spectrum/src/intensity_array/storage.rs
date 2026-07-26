use std::borrow::Cow;
use std::rc::Rc;
use std::sync::Arc;

/// Immutable array storage.
///
/// The array types owns shape and indexing. Storage only provides access to the
/// elements.
///
/// # Safety
///
/// For any given value, `as_slice` must return a slice with the same base
/// pointer and length on every call, until that value is modified other than
/// through a slice it returned. Overwriting elements through a returned slice
/// is not such a modification. Operations that may resize or reallocate the
/// buffer, such as [`Vec::push`], are.
///
/// Arrays validate storage length against their shape at construction, so
/// violating this yields panics, incorrect results and possibly undefined
/// behavior.
pub unsafe trait Storage: Sized {
    /// Array element type.
    type Elem;

    /// Returns an immutable slice containing the entire storage.
    fn as_slice(&self) -> &[Self::Elem];
}

unsafe impl<T> Storage for Vec<T> {
    type Elem = T;

    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

unsafe impl<T> Storage for &[T] {
    type Elem = T;

    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

unsafe impl<T> Storage for &mut [T] {
    type Elem = T;

    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

unsafe impl<T> Storage for Box<[T]> {
    type Elem = T;

    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

unsafe impl<'s, T> Storage for Cow<'s, [T]>
where
    T: Clone,
{
    type Elem = T;

    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

unsafe impl<T> Storage for Rc<[T]> {
    type Elem = T;

    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

unsafe impl<T> Storage for Arc<[T]> {
    type Elem = T;

    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

/// Mutable array storage.
///
/// # Safety
///
/// `as_mut_slice` is subject to the same requirement as [`Storage::as_slice`],
/// and must return the same length as it.
///
/// Calling `as_mut_slice` counts as modifying the value for the purposes of
/// `as_slice`'s stability requirement. Implementations that unshare a buffer
/// may return a different base pointer than a prior call to either method. The
/// elements must nevertheless compare equal to those `as_slice` would have
/// returned. Writing through a returned slice is not such a modification,
/// though the writes must be observable through every subsequent call to
/// `as_slice` and `as_mut_slice`.
pub unsafe trait StorageMut: Storage {
    /// Returns a mutable slice containing the entire storage.
    fn as_mut_slice(&mut self) -> &mut [Self::Elem];
}

unsafe impl<T> StorageMut for Vec<T> {
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self
    }
}

unsafe impl<T> StorageMut for &mut [T] {
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self
    }
}

unsafe impl<T> StorageMut for Box<[T]> {
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self
    }
}

unsafe impl<'s, T> StorageMut for Cow<'s, [T]>
where
    T: Clone,
{
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self.to_mut()
    }
}

/// Owned storage for operations that allocate a result.
pub trait StorageOwned: Storage {
    /// Constructs `Self` from an owned `Vec`.
    ///
    /// The result must contain the same elements in the same memory order.
    fn from_vec(data: Vec<Self::Elem>) -> Self;
}

impl<T> StorageOwned for Vec<T> {
    fn from_vec(data: Vec<Self::Elem>) -> Self {
        data
    }
}

impl<T> StorageOwned for Box<[T]> {
    fn from_vec(data: Vec<Self::Elem>) -> Self {
        data.into_boxed_slice()
    }
}

impl<'s, T> StorageOwned for Cow<'s, [T]>
where
    T: Clone,
{
    fn from_vec(data: Vec<Self::Elem>) -> Self {
        data.into()
    }
}

impl<T> StorageOwned for Rc<[T]> {
    fn from_vec(data: Vec<Self::Elem>) -> Self {
        Rc::<[T]>::from(data)
    }
}

impl<T> StorageOwned for Arc<[T]> {
    fn from_vec(data: Vec<Self::Elem>) -> Self {
        Arc::<[T]>::from(data)
    }
}
