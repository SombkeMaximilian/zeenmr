use std::borrow::Cow;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;

/// Immutable array storage that only provides a base pointer to the allocation.
///
/// Dereferencing the pointer returned by `as_ptr` at any offset is potentially
/// unsound. What an implementation is paired with, such as an array layout,
/// determines which offsets may be read.
///
/// See the [`module`] level documentation for more information.
///
/// [`module`]: crate::intensity_array
///
/// # Safety
///
/// For any given value, `as_ptr` must return  the same base pointer on every
/// call, until that value is modified other than through a slice or pointer it
/// returned. Writing elements is not such a modification. Operations that may
/// resize or reallocate the buffer, such as [`Vec::push`] and [`Cow::to_mut`],
/// are.
pub unsafe trait RawStorage: Sized {
    /// Array element type.
    type Elem;

    /// Returns the base pointer of the allocation.
    fn as_ptr(&self) -> *const Self::Elem;
}

unsafe impl<T> RawStorage for Vec<T> {
    type Elem = T;

    fn as_ptr(&self) -> *const Self::Elem {
        <[T]>::as_ptr(self)
    }
}

unsafe impl<T> RawStorage for &[T] {
    type Elem = T;

    fn as_ptr(&self) -> *const Self::Elem {
        <[T]>::as_ptr(self)
    }
}

unsafe impl<T> RawStorage for &mut [T] {
    type Elem = T;

    fn as_ptr(&self) -> *const Self::Elem {
        <[T]>::as_ptr(self)
    }
}

unsafe impl<T> RawStorage for Box<[T]> {
    type Elem = T;

    fn as_ptr(&self) -> *const Self::Elem {
        <[T]>::as_ptr(self)
    }
}

unsafe impl<T> RawStorage for Cow<'_, [T]>
where
    T: Clone,
{
    type Elem = T;

    fn as_ptr(&self) -> *const Self::Elem {
        <[T]>::as_ptr(self)
    }
}

unsafe impl<T> RawStorage for Rc<[T]> {
    type Elem = T;

    fn as_ptr(&self) -> *const Self::Elem {
        <[T]>::as_ptr(self)
    }
}

unsafe impl<T> RawStorage for Arc<[T]> {
    type Elem = T;

    fn as_ptr(&self) -> *const Self::Elem {
        <[T]>::as_ptr(self)
    }
}

/// Mutable array storage that only provides a base pointer to the allocation.
///
/// This trait is the mutable analogue of [`RawStorage`].
///
/// See the [`module`] level documentation for more information.
///
/// [`module`]: crate::intensity_array
///
/// # Safety
///
/// `as_mut_ptr` is subject to the same requirement as [`RawStorage::as_ptr`].
///
/// Calling it counts as modifying the value for the purposes of that
/// requirement. Implementations that unshare a buffer when producing a mutable
/// reference or pointer may return a different base pointer than a prior call
/// to either method returned. The elements reachable from it must compare equal
/// to those the previous point would have made accessible. Writes through a
/// returned pointer must be observable through every subsequent call to either
/// method.
pub unsafe trait RawStorageMut: RawStorage {
    /// Returns the mutable base pointer of the allocation.
    fn as_mut_ptr(&mut self) -> *mut Self::Elem;
}

unsafe impl<T> RawStorageMut for Vec<T> {
    fn as_mut_ptr(&mut self) -> *mut Self::Elem {
        <[T]>::as_mut_ptr(self)
    }
}

unsafe impl<T> RawStorageMut for &mut [T] {
    fn as_mut_ptr(&mut self) -> *mut Self::Elem {
        <[T]>::as_mut_ptr(self)
    }
}

unsafe impl<T> RawStorageMut for Box<[T]> {
    fn as_mut_ptr(&mut self) -> *mut Self::Elem {
        <[T]>::as_mut_ptr(self)
    }
}

unsafe impl<T> RawStorageMut for Cow<'_, [T]>
where
    T: Clone,
{
    fn as_mut_ptr(&mut self) -> *mut Self::Elem {
        self.to_mut().as_mut_ptr()
    }
}

/// Immutable array storage.
///
/// Implementing this trait is only necessary to back an array with a custom
/// buffer type. The provided implementations cover the standard library types
/// for owned buffers, borrowed slices, and the shared and clone-on-write smart
/// pointers.
///
/// See the [`module`] level documentation for more information.
///
/// [`module`]: crate::intensity_array
///
/// # Safety
///
/// `as_slice` must return a slice whose base pointer is the one
/// [`RawStorage::as_ptr`] returns, and is subject to the same requirement,
/// extended to the length, which must also be identical on every call, until
/// modification.
///
/// Arrays validate storage length against their shape at construction, so
/// violating this yields panics, incorrect results and possibly undefined
/// behavior.
pub unsafe trait Storage: RawStorage {
    /// Returns an immutable slice containing the entire storage.
    fn as_slice(&self) -> &[Self::Elem];
}

unsafe impl<T> Storage for Vec<T> {
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

unsafe impl<T> Storage for &[T] {
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

unsafe impl<T> Storage for &mut [T] {
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

unsafe impl<T> Storage for Box<[T]> {
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

unsafe impl<T> Storage for Cow<'_, [T]>
where
    T: Clone,
{
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

unsafe impl<T> Storage for Rc<[T]> {
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

unsafe impl<T> Storage for Arc<[T]> {
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

/// Mutable array storage.
///
/// This trait is the mutable analogue of [`Storage`].
///
/// See the [`module`] level documentation for more information.
///
/// [`module`]: crate::intensity_array
///
/// # Safety
///
/// `as_mut_slice` must return a slice whose base pointer is hte one
/// [`RawStorageMut`] returns, and whose length is equal to that of
/// [`Storage::as_slice`]. Calling it counts as a modification in the same
/// way [`RawStorageMut::as_mut_ptr`] does.
pub unsafe trait StorageMut: Storage + RawStorageMut {
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

unsafe impl<T> StorageMut for Cow<'_, [T]>
where
    T: Clone,
{
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self.to_mut()
    }
}

/// Owned storage for operations that allocate a result.
pub trait StorageOwned: Storage + FromIterator<Self::Elem> {
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

impl<T> StorageOwned for Cow<'_, [T]>
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

/// Pointer type that provides immutable access to elements of an allocation.
///
/// The pointer this type contains is always non-null and aligned, and this
/// storage type can be thought of as a weaker `&[T]` in the sense that it
/// potentially isn't sound to dereference every offset within the allocation
/// from it.
///
/// See the [`module`] level documentation for more information.
///
/// [`module`]: crate::intensity_array#access-pointers
pub struct Access<'s, T> {
    /// Base pointer of the storage.
    ptr: NonNull<T>,
    /// Lifetime marker for the reference.
    lifetime: PhantomData<&'s T>,
}

impl<T> Copy for Access<'_, T> {}

impl<T> Clone for Access<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> std::fmt::Debug for Access<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_tuple("Access").field(&self.ptr).finish()
    }
}

// SAFETY: grants shared access to `T`, so it is Send when `T` is Sync.
unsafe impl<T> Send for Access<'_, T> where T: Sync {}

// SAFETY: grants shared access to `T`, so it is Sync when `T` is Sync.
unsafe impl<T> Sync for Access<'_, T> where T: Sync {}

unsafe impl<T> RawStorage for Access<'_, T> {
    type Elem = T;

    fn as_ptr(&self) -> *const Self::Elem {
        self.ptr.as_ptr()
    }
}

impl<'s, T> Access<'s, T> {
    /// Creates an access pointer to the allocation backing `storage`.
    pub fn from_slice(storage: &'s [T]) -> Self {
        Self {
            ptr: NonNull::from(storage).cast::<T>(),
            lifetime: PhantomData,
        }
    }
}

/// Pointer type that provides unique access to elements of an allocation.
///
/// The pointer this type contains is always non-null and aligned, and this
/// storage type can be thought of as a weaker `&mut [T]` in the sense that it
/// potentially isn't sound to dereference every offset within the allocation
/// from it.
///
/// See the [`module`] level documentation for more information.
///
/// [`module`]: crate::intensity_array#access-pointers
pub struct AccessMut<'s, T> {
    /// Base pointer of the storage.
    ptr: NonNull<T>,
    /// Lifetime marker for the reference.
    lifetime: PhantomData<&'s mut T>,
}

impl<T> std::fmt::Debug for AccessMut<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_tuple("AccessMut")
            .field(&self.ptr)
            .finish()
    }
}

// SAFETY: grants unique access to `T`, so it is Send when `T` is Send.
unsafe impl<T> Send for AccessMut<'_, T> where T: Send {}

// SAFETY: grants unique access to `T`, so it is Sync when `T` is Sync.
unsafe impl<T> Sync for AccessMut<'_, T> where T: Sync {}

unsafe impl<T> RawStorage for AccessMut<'_, T> {
    type Elem = T;

    fn as_ptr(&self) -> *const Self::Elem {
        self.ptr.as_ptr()
    }
}

unsafe impl<T> RawStorageMut for AccessMut<'_, T> {
    fn as_mut_ptr(&mut self) -> *mut Self::Elem {
        self.ptr.as_ptr()
    }
}

impl<'s, T> AccessMut<'s, T> {
    /// Creates a mutable access pointer to the allocation backing `storage`.
    pub fn from_slice(storage: &'s mut [T]) -> Self {
        Self {
            ptr: NonNull::from(storage).cast::<T>(),
            lifetime: PhantomData,
        }
    }

    /// Returns a mutable access pointer of the same storage borrowing `self`.
    pub fn reborrow(&mut self) -> AccessMut<'_, T> {
        Self {
            ptr: self.ptr,
            lifetime: PhantomData,
        }
    }

    /// Returns an immutable access pointer of the same storage borrowing
    /// `self`.
    pub fn as_access(&self) -> Access<'_, T> {
        Access {
            ptr: self.ptr,
            lifetime: PhantomData,
        }
    }

    /// Converts `self` into an immutable access pointer of the same storage.
    pub fn into_access(self) -> Access<'s, T> {
        Access {
            ptr: self.ptr,
            lifetime: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointer_size() {
        const fn assert_size_<T>() {
            assert!(size_of::<Access<'_, T>>() == size_of::<Option<Access<'_, T>>>());
            assert!(size_of::<Access<'_, T>>() == size_of::<*const T>());
            assert!(align_of::<Access<'_, T>>() == align_of::<Option<Access<'_, T>>>());
            assert!(align_of::<Access<'_, T>>() == align_of::<*const T>());
            assert!(size_of::<AccessMut<'_, T>>() == size_of::<Option<AccessMut<'_, T>>>());
            assert!(size_of::<AccessMut<'_, T>>() == size_of::<*mut T>());
            assert!(align_of::<AccessMut<'_, T>>() == align_of::<Option<AccessMut<'_, T>>>());
            assert!(align_of::<AccessMut<'_, T>>() == align_of::<*mut T>());
        }
        struct ZST;

        const _: () = {
            assert_size_::<u8>();
            assert_size_::<&u8>();
            assert_size_::<&mut u8>();
            assert_size_::<Box<[u8]>>();
            assert_size_::<ZST>();
        };
    }

    #[test]
    fn round_trip_to_slice() {
        let data = (0..10).collect::<Box<[u8]>>();
        let access = Access::from_slice(&data);
        // SAFETY: this is just a pointer round trip, passing the raw parts
        // as returned by `data` back into a slice, though going through
        // `Access` in between.
        let slice = unsafe { std::slice::from_raw_parts(access.as_ptr(), data.len()) };

        assert!(std::ptr::eq(data.as_ptr(), access.as_ptr()));
        assert_eq!(data.as_ref(), slice);
    }

    #[test]
    fn changes_observable() {
        let mut data = (0..10).collect::<Box<[u8]>>();
        let mut access = AccessMut::from_slice(&mut data);
        // SAFETY: trivially in bounds and no multiple aliasing.
        unsafe { *access.as_mut_ptr().add(2) = 100 };

        assert_eq!(data.get(2), Some(&100));
    }
}
