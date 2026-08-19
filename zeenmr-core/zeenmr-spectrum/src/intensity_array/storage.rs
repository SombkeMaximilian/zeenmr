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
/// For any given value, `as_ptr` must return the same, aligned, non-null base
/// pointer on every call, until that value is modified other than through a
/// slice or pointer it returned. Writing elements is not such a modification.
/// Operations that may resize or reallocate the buffer, such as [`Vec::push`]
/// and [`Cow::to_mut`], are.
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
/// Always non-null and aligned.
pub(crate) struct RawAccess<'s, T> {
    /// Base pointer of the storage.
    ptr: NonNull<T>,
    /// Lifetime marker for the reference.
    lifetime: PhantomData<&'s T>,
}

impl<T> Copy for RawAccess<'_, T> {}

impl<T> Clone for RawAccess<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> std::fmt::Debug for RawAccess<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_tuple("RawAccess")
            .field(&self.ptr)
            .finish()
    }
}

// SAFETY: grants shared access to `T`, so it is Send when `T` is Sync.
unsafe impl<T> Send for RawAccess<'_, T> where T: Sync {}

// SAFETY: grants shared access to `T`, so it is Sync when `T` is Sync.
unsafe impl<T> Sync for RawAccess<'_, T> where T: Sync {}

impl<'s, T> RawAccess<'s, T> {
    /// Creates a raw access pointer to the allocation backing `storage`.
    pub(crate) fn from_slice(storage: &'s [T]) -> Self {
        Self {
            ptr: NonNull::from(storage).cast::<T>(),
            lifetime: PhantomData,
        }
    }

    /// Creates a raw access pointer from a raw pointer.
    ///
    /// # Safety
    ///
    /// `base` must be non-null and aligned. Every offset addressed by a memory
    /// geometry this pointer is paired with must be a valid offset into the
    /// allocation, and its elements must be borrowed immutably for `'s`.
    pub(crate) unsafe fn from_raw(base: *const T) -> Self {
        // SAFETY: the caller guarantees that `base` is non-null and aligned.
        let ptr = unsafe { NonNull::new_unchecked(base.cast_mut()) };

        Self {
            ptr,
            lifetime: PhantomData,
        }
    }

    /// Returns an access pointer of the same storage borrowing `self`.
    pub(crate) fn reborrow(&self) -> RawAccess<'_, T> {
        Self {
            ptr: self.ptr,
            lifetime: PhantomData,
        }
    }

    /// Returns the base pointer of the allocation.
    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }
}

/// Pointer type that provides unique access to elements of an allocation.
///
/// Always non-null and aligned.
pub(crate) struct RawAccessMut<'s, T> {
    /// Base pointer of the storage.
    ptr: NonNull<T>,
    /// Lifetime marker for the reference.
    lifetime: PhantomData<&'s mut T>,
}

impl<T> Copy for RawAccessMut<'_, T> {}

impl<T> Clone for RawAccessMut<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> std::fmt::Debug for RawAccessMut<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_tuple("RawAccessMut")
            .field(&self.ptr)
            .finish()
    }
}

// SAFETY: grants unique access to `T`, so it is Send when `T` is Send.
unsafe impl<T> Send for RawAccessMut<'_, T> where T: Send {}

// SAFETY: grants unique access to `T`, so it is Sync when `T` is Sync.
unsafe impl<T> Sync for RawAccessMut<'_, T> where T: Sync {}

impl<'s, T> RawAccessMut<'s, T> {
    /// Creates a mutable, raw access pointer to the allocation backing
    /// `storage`.
    pub(crate) fn from_slice(storage: &'s mut [T]) -> Self {
        Self {
            ptr: NonNull::from(storage).cast::<T>(),
            lifetime: PhantomData,
        }
    }

    /// Creates a raw, mutable access pointer from a raw pointer.
    ///
    /// # Safety
    ///
    /// `base` must be non-null and aligned. Every offset addressed by a memory
    /// geometry this type is paired with must be a valid offset into the
    /// allocation, and its elements must be borrowed mutably for `'s`. No
    /// other live reference to them may exist. Any such geometry must further
    /// be injective, i.e., it must not create the same offset multiple times.
    pub(crate) unsafe fn from_raw(base: *mut T) -> Self {
        // SAFETY: the caller guarantees that `base` is non-null and aligned.
        let ptr = unsafe { NonNull::new_unchecked(base) };

        Self {
            ptr,
            lifetime: PhantomData,
        }
    }

    /// Returns a mutable access pointer of the same storage borrowing `self`.
    pub(crate) fn reborrow(&mut self) -> RawAccessMut<'_, T> {
        Self {
            ptr: self.ptr,
            lifetime: PhantomData,
        }
    }

    /// Returns an immutable access pointer of the same storage borrowing
    /// `self`.
    pub(crate) fn as_access(&self) -> RawAccess<'_, T> {
        RawAccess {
            ptr: self.ptr,
            lifetime: PhantomData,
        }
    }

    /// Converts `self` into an immutable access pointer of the same storage.
    pub(crate) fn into_access(self) -> RawAccess<'s, T> {
        RawAccess {
            ptr: self.ptr,
            lifetime: PhantomData,
        }
    }

    /// Returns the base pointer of the allocation.
    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns the mutable base pointer of the allocation.
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
}

/// Pointer wrapper that provides immutable access to elements of an allocation.
///
/// Creating an access pointer directly is almost never necessary. Prefer
/// letting an existing array derive a view, both of which handle the
/// requirements below. Since part of the soundness contract requires knowing
/// which geometries such a pointer will be paired with, it is not possible to
/// reason about it locally.
///
/// The pointer this type contains is always non-null and aligned, and this
/// storage type can be thought of as a weaker `&[T]` in the sense that it
/// potentially isn't sound to dereference every offset within the allocation
/// from it.
///
/// See the [`module`] level documentation for more information.
///
/// [`module`]: crate::intensity_array#access-pointers
pub struct Access<'s, T>(RawAccess<'s, T>);

impl<T> Copy for Access<'_, T> {}

impl<T> Clone for Access<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> std::fmt::Debug for Access<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_tuple("Access")
            .field(&self.as_ptr())
            .finish()
    }
}

unsafe impl<T> RawStorage for Access<'_, T> {
    type Elem = T;

    fn as_ptr(&self) -> *const Self::Elem {
        self.0.as_ptr()
    }
}

impl<'s, T> Access<'s, T> {
    /// Creates an access pointer to the allocation backing `storage`.
    pub fn from_slice(storage: &'s [T]) -> Self {
        Self(RawAccess::from_slice(storage))
    }

    /// Creates a raw access pointer from a raw pointer.
    ///
    /// # Safety
    ///
    /// `base` must be non-null and aligned. Every offset addressed by a memory
    /// geometry this pointer is paired with must be a valid offset into the
    /// allocation, and its elements must be borrowed immutably for `'s`.
    pub unsafe fn from_raw(base: *const T) -> Self {
        // SAFETY: the caller guarantees that `base` is non-null and aligned.
        Self(unsafe { RawAccess::from_raw(base) })
    }

    /// Returns an access pointer of the same storage borrowing `self`.
    pub fn reborrow(&mut self) -> Access<'_, T> {
        Access(self.0.reborrow())
    }
}

/// Pointer wrapper that provides unique access to elements of an allocation.
///
/// Creating an access pointer directly is almost never necessary. Prefer
/// letting an existing array derive a view, both of which handle the
/// requirements below. Since part of the soundness contract requires knowing
/// which geometries such a pointer will be paired with, it is not possible to
/// reason about it locally.
///
/// The pointer this type contains is always non-null and aligned, and this
/// storage type can be thought of as a weaker `&mut [T]` in the sense that it
/// potentially isn't sound to dereference every offset within the allocation
/// from it.
///
/// See the [`module`] level documentation for more information.
///
/// [`module`]: crate::intensity_array#access-pointers
pub struct AccessMut<'s, T>(RawAccessMut<'s, T>);

impl<T> std::fmt::Debug for AccessMut<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_tuple("AccessMut")
            .field(&self.as_ptr())
            .finish()
    }
}

unsafe impl<T> RawStorage for AccessMut<'_, T> {
    type Elem = T;

    fn as_ptr(&self) -> *const Self::Elem {
        self.0.as_ptr()
    }
}

unsafe impl<T> RawStorageMut for AccessMut<'_, T> {
    fn as_mut_ptr(&mut self) -> *mut Self::Elem {
        self.0.as_mut_ptr()
    }
}

impl<'s, T> AccessMut<'s, T> {
    /// Creates a mutable access pointer to the allocation backing `storage`.
    pub fn from_slice(storage: &'s mut [T]) -> Self {
        Self(RawAccessMut::from_slice(storage))
    }

    /// Creates a raw, mutable access pointer from a raw pointer.
    ///
    /// # Safety
    ///
    /// `base` must be non-null and aligned. Every offset addressed by a memory
    /// geometry this type is paired with must be a valid offset into the
    /// allocation, and its elements must be borrowed mutably for `'s`. No
    /// other live reference to them may exist. Any such geometry must further
    /// be injective, i.e., it must not create the same offset multiple times.
    pub unsafe fn from_raw(base: *mut T) -> Self {
        // SAFETY: the caller guarantees that `base` is non-null and aligned.
        Self(unsafe { RawAccessMut::from_raw(base) })
    }

    /// Returns a mutable access pointer of the same storage borrowing `self`.
    pub fn reborrow(&mut self) -> AccessMut<'_, T> {
        AccessMut(self.0.reborrow())
    }

    /// Returns an immutable access pointer of the same storage borrowing
    /// `self`.
    pub fn as_access(&self) -> Access<'_, T> {
        Access(self.0.as_access())
    }

    /// Converts `self` into an immutable access pointer of the same storage.
    pub fn into_access(self) -> Access<'s, T> {
        Access(self.0.into_access())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::assert_impl_all;

    #[test]
    fn thread_safety() {
        assert_impl_all!(Access<'_, u8>: Send, Sync);
        assert_impl_all!(AccessMut<'_, u8>: Send, Sync);
        assert_impl_all!(RawAccess<'_, u8>: Send, Sync);
        assert_impl_all!(RawAccessMut<'_, u8>: Send, Sync);
    }

    #[test]
    fn pointer_size() {
        const fn assert_size<T>() {
            const fn assert_block<A, P>() {
                assert!(size_of::<A>() == size_of::<Option<A>>());
                assert!(size_of::<A>() == size_of::<P>());
                assert!(align_of::<A>() == align_of::<Option<A>>());
                assert!(align_of::<A>() == align_of::<P>());
            }

            assert_block::<Access<'_, T>, *const T>();
            assert_block::<AccessMut<'_, T>, *mut T>();
            assert_block::<RawAccess<'_, T>, *const T>();
            assert_block::<RawAccessMut<'_, T>, *mut T>();
        }
        struct Zst;

        const _: () = {
            assert_size::<u8>();
            assert_size::<&u8>();
            assert_size::<&mut u8>();
            assert_size::<Box<[u8]>>();
            assert_size::<Zst>();
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
