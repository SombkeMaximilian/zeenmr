use crate::intensity_array::LaneGeometry;
use std::iter::FusedIterator;
use std::marker::PhantomData;

#[cfg(feature = "rayon")]
use rayon::iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Iterator over the buffer offsets of a lane along one dimension.
#[derive(Clone, Debug)]
pub struct LaneOffsets {
    /// Geometry of the lane.
    geometry: LaneGeometry,
    /// Next lane offset from the front.
    front: usize,
    /// Next lane offset from the back.
    back: usize,
}

impl Iterator for LaneOffsets {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let curr = self.geometry.offset_of_unvalidated(self.front);
            self.front += 1;

            Some(curr)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;

        (remaining, Some(remaining))
    }
}

impl DoubleEndedIterator for LaneOffsets {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            self.back -= 1;

            Some(self.geometry.offset_of_unvalidated(self.back))
        } else {
            None
        }
    }
}

impl ExactSizeIterator for LaneOffsets {}

impl FusedIterator for LaneOffsets {}

impl LaneOffsets {
    /// Creates an iterator over the offsets of a lane.
    pub fn new(geometry: LaneGeometry) -> Self {
        Self {
            geometry,
            front: 0,
            back: geometry.len(),
        }
    }

    /// Returns the stride of the iterator.
    pub fn stride(&self) -> usize {
        self.geometry.stride()
    }
}

/// Parallel iterator over the buffer offsets of a lane along one dimension.
#[cfg(feature = "rayon")]
#[derive(Clone, Debug)]
pub struct ParLaneOffsets(LaneOffsets);

#[cfg(feature = "rayon")]
impl ParallelIterator for ParLaneOffsets {
    type Item = usize;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.0.len())
    }
}

#[cfg(feature = "rayon")]
impl IndexedParallelIterator for ParLaneOffsets {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(LaneOffsetsProducer(self.0))
    }
}

#[cfg(feature = "rayon")]
impl ParLaneOffsets {
    /// Creates a parallel iterator over the offsets of a lane.
    pub fn new(geometry: LaneGeometry) -> Self {
        Self(LaneOffsets::new(geometry))
    }
}

/// Producer for [`ParLaneOffsets`]
#[cfg(feature = "rayon")]
struct LaneOffsetsProducer(LaneOffsets);

#[cfg(feature = "rayon")]
impl Producer for LaneOffsetsProducer {
    type Item = usize;
    type IntoIter = LaneOffsets;

    fn into_iter(self) -> Self::IntoIter {
        self.0
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let mid = self.0.front + index;
        let left = LaneOffsets {
            geometry: self.0.geometry,
            front: self.0.front,
            back: mid,
        };
        let right = LaneOffsets {
            geometry: self.0.geometry,
            front: mid,
            back: self.0.back,
        };

        (LaneOffsetsProducer(left), LaneOffsetsProducer(right))
    }
}

/// Strided iterator over the elements of a buffer.
#[derive(Debug)]
pub struct LaneElemStrided<'s, T> {
    /// Base pointer of the storage.
    base: *const T,
    /// Offsets to visit.
    ///
    /// # Safety
    ///
    /// All offsets returned by this iterator must be within bounds of the
    /// allocation `base` points into.
    offsets: LaneOffsets,
    /// Lifetime marker for the reference.
    ///
    /// # Safety
    ///
    /// This allows the borrow checker to ensure the pointer is never dangling.
    lifetime: PhantomData<&'s T>,
}

impl<T> Clone for LaneElemStrided<'_, T> {
    fn clone(&self) -> Self {
        Self {
            base: self.base,
            offsets: self.offsets.clone(),
            lifetime: PhantomData,
        }
    }
}

// SAFETY: grants shared access to `T`, so it is Send when `T` is Sync.
unsafe impl<T> Send for LaneElemStrided<'_, T> where T: Sync {}

// SAFETY: grants shared access to `T`, so it is Sync when `T` is Sync.
unsafe impl<T> Sync for LaneElemStrided<'_, T> where T: Sync {}

impl<'s, T> Iterator for LaneElemStrided<'s, T> {
    type Item = &'s T;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.offsets.next()?;

        // SAFETY: offsets are all guaranteed to be within bounds by
        // construction of this type
        Some(unsafe { &*self.base.add(next) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.offsets.size_hint()
    }
}

impl<'s, T> DoubleEndedIterator for LaneElemStrided<'s, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let next = self.offsets.next_back()?;

        // SAFETY: offsets are all guaranteed to be within bounds by
        // construction of this type
        Some(unsafe { &*self.base.add(next) })
    }
}

impl<'s, T> ExactSizeIterator for LaneElemStrided<'s, T> {}

impl<'s, T> FusedIterator for LaneElemStrided<'s, T> {}

impl<'s, T> LaneElemStrided<'s, T> {
    /// Creates a strided iterator over the elements of `base` according to
    /// `geometry`.
    ///
    /// Returns `None` if any offset generated by `geometry` is out of bounds
    /// or overflows.
    pub fn new(base: &'s [T], geometry: LaneGeometry) -> Option<Self> {
        geometry.fits_within(base.len()).then(|| {
            // SAFETY: `Geometry::fits_within` guarantees that every offset
            // it addresses is non-overflowing and within `len` and thereby
            // within bounds of `base`.
            unsafe { Self::from_raw(base.as_ptr(), geometry) }
        })
    }

    /// Creates a strided iterator over the elements addressed by `geometry`
    /// relative to `base`.
    ///
    /// # Safety
    ///
    /// Every offset `geometry` addresses must be a valid index into the
    /// allocation `base` points into, and those elements must be borrowed
    /// immutably for `'s`.
    pub(crate) unsafe fn from_raw(base: *const T, geometry: LaneGeometry) -> Self {
        Self {
            base,
            offsets: LaneOffsets::new(geometry),
            lifetime: PhantomData,
        }
    }

    /// Returns the stride of the iterator.
    pub fn stride(&self) -> usize {
        self.offsets.geometry.stride()
    }
}

/// Parallel, strided iterator over the elements of a buffer.
#[cfg(feature = "rayon")]
#[derive(Debug)]
pub struct ParLaneElemStrided<'s, T>(LaneElemStrided<'s, T>);

#[cfg(feature = "rayon")]
impl<T> Clone for ParLaneElemStrided<'_, T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

#[cfg(feature = "rayon")]
impl<'s, T> ParallelIterator for ParLaneElemStrided<'s, T>
where
    T: Sync,
{
    type Item = &'s T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.0.len())
    }
}

#[cfg(feature = "rayon")]
impl<'s, T> IndexedParallelIterator for ParLaneElemStrided<'s, T>
where
    T: Sync,
{
    fn len(&self) -> usize {
        self.0.len()
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(LaneElemStridedProducer {
            base: self.0.base,
            producer: LaneOffsetsProducer(self.0.offsets),
            lifetime: PhantomData,
        })
    }
}

#[cfg(feature = "rayon")]
impl<'s, T> ParLaneElemStrided<'s, T> {
    /// Creates a parallel, strided iterator over the elements of `base`
    /// according to `geometry`.
    ///
    /// Returns `None` if any offset generated by `geometry` is out of bounds
    /// or overflows.
    pub fn new(base: &'s [T], geometry: LaneGeometry) -> Option<Self> {
        Some(Self(LaneElemStrided::new(base, geometry)?))
    }

    /// Creates a parallel, strided iterator over the elements addressed by
    /// `geometry` relative to `base`.
    ///
    /// # Safety
    ///
    /// Every offset `geometry` addresses must be a valid index into the
    /// allocation `base` points into, and those elements must be borrowed
    /// immutably for `'s`.
    pub(crate) unsafe fn from_raw(base: *const T, geometry: LaneGeometry) -> Self {
        // SAFETY: requirements are the same as `LaneElemStrided::from_raw`, and
        // the caller guarantees that this is fine.
        unsafe { Self(LaneElemStrided::from_raw(base, geometry)) }
    }

    /// Returns the stride of the iterator.
    pub fn stride(&self) -> usize {
        self.0.stride()
    }
}

/// Producer for [`ParLaneElemStrided`].
#[cfg(feature = "rayon")]
struct LaneElemStridedProducer<'s, T> {
    /// Reference to the underlying buffer.
    base: *const T,
    /// Producer for [`ParLaneOffsets`].
    producer: LaneOffsetsProducer,
    /// Lifetime marker for the reference.
    lifetime: PhantomData<&'s T>,
}

// SAFETY: grants shared access to `T`, so it is Send when `T` is Sync.
#[cfg(feature = "rayon")]
unsafe impl<'s, T> Send for LaneElemStridedProducer<'s, T> where T: Sync {}

// SAFETY: grants shared access to `T`, so it is Sync when `T` is Sync.
#[cfg(feature = "rayon")]
unsafe impl<'s, T> Sync for LaneElemStridedProducer<'s, T> where T: Sync {}

#[cfg(feature = "rayon")]
impl<'s, T> Producer for LaneElemStridedProducer<'s, T>
where
    T: Sync,
{
    type Item = &'s T;
    type IntoIter = LaneElemStrided<'s, T>;

    fn into_iter(self) -> Self::IntoIter {
        LaneElemStrided {
            base: self.base,
            offsets: self.producer.into_iter(),
            lifetime: PhantomData,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.producer.split_at(index);

        (
            Self {
                base: self.base,
                producer: left,
                lifetime: PhantomData,
            },
            Self {
                base: self.base,
                producer: right,
                lifetime: PhantomData,
            },
        )
    }
}

/// Iterator over contiguous elements of a buffer.
pub type LaneElemContiguous<'s, T> = std::slice::Iter<'s, T>;

/// Parallel iterator over contiguous elements of a buffer.
#[cfg(feature = "rayon")]
pub type ParLaneElemContiguous<'s, T> = rayon::slice::Iter<'s, T>;

/// Strided iterator over mutable references of the elements of a buffer.
#[derive(Debug)]
pub struct LaneElemStridedMut<'s, T> {
    /// Base pointer of the storage.
    base: *mut T,
    /// Offsets to visit.
    ///
    /// # Safety
    ///
    /// All offsets returned by this iterator must be within bounds of the
    /// allocation `base` points into, and the geometry must be injective, so
    /// no offset is yielded twice.
    offsets: LaneOffsets,
    /// Lifetime marker for the mutable reference which is invariant in `T`.
    ///
    /// # Safety
    ///
    /// This allows the borrow checker to ensure the pointer is never dangling.
    lifetime: PhantomData<&'s mut T>,
}

// SAFETY: grants unique access to `T`, so it is Send when `T` is Send.
unsafe impl<T> Send for LaneElemStridedMut<'_, T> where T: Send {}

// SAFETY: grants unique access to `T`, so it is Sync when `T` is Sync.
unsafe impl<T> Sync for LaneElemStridedMut<'_, T> where T: Sync {}

impl<'s, T> Iterator for LaneElemStridedMut<'s, T> {
    type Item = &'s mut T;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.offsets.next()?;

        // SAFETY: the offset is in bounds by construction, and injectivity
        // guarantees it is never yielded twice, so the returned reference is
        // unique.
        Some(unsafe { &mut *self.base.add(next) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.offsets.size_hint()
    }
}

impl<'s, T> DoubleEndedIterator for LaneElemStridedMut<'s, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let next = self.offsets.next_back()?;

        // SAFETY: the offset is in bounds by construction, and injectivity
        // guarantees it is never yielded twice, so the returned reference is
        // unique.
        Some(unsafe { &mut *self.base.add(next) })
    }
}

impl<'s, T> ExactSizeIterator for LaneElemStridedMut<'s, T> {}

impl<'s, T> FusedIterator for LaneElemStridedMut<'s, T> {}

impl<'s, T> LaneElemStridedMut<'s, T> {
    /// Creates a strided iterator over mutable references of the elements of
    /// `base` according to `geometry`.
    ///
    /// Returns `None` if any offset generated by `geometry` is out of bounds or
    /// overflows, or if `geometry` is not injective.
    pub fn new(base: &'s mut [T], geometry: LaneGeometry) -> Option<Self> {
        if !geometry.is_injective() {
            return None;
        }

        geometry.fits_within(base.len()).then(|| {
            // SAFETY: `Geometry::fits_within` guarantees that every offset
            // it addresses is non-overflowing and within `len` and thereby
            // within bounds of `base`. `Geometry::is_injective` guarantees
            // that every offset addressed by it is unique.
            unsafe { Self::from_raw(base.as_mut_ptr(), geometry) }
        })
    }

    /// Creates a strided iterator over mutable references to the elements
    /// addressed by `geometry` relative to `base`.
    ///
    /// # Safety
    ///
    /// Every offset `geometry` addresses must be a valid index into the
    /// allocation `base` points into, those elements must be borrowed mutably
    /// for `'s`. No other live reference to them must exist, and `geometry`
    /// must be injective so no offset is yielded twice.
    pub(crate) unsafe fn from_raw(base: *mut T, geometry: LaneGeometry) -> Self {
        debug_assert!(geometry.is_injective());

        Self {
            base,
            offsets: LaneOffsets::new(geometry),
            lifetime: PhantomData,
        }
    }

    /// Returns the stride of the iterator.
    pub fn stride(&self) -> usize {
        self.offsets.geometry.stride()
    }
}

/// Parallel, strided iterator over mutable references of the elements of a
/// buffer.
#[cfg(feature = "rayon")]
#[derive(Debug)]
pub struct ParLaneElemStridedMut<'s, T>(LaneElemStridedMut<'s, T>);

#[cfg(feature = "rayon")]
impl<'s, T> ParallelIterator for ParLaneElemStridedMut<'s, T>
where
    T: Send,
{
    type Item = &'s mut T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.0.len())
    }
}

#[cfg(feature = "rayon")]
impl<'s, T> IndexedParallelIterator for ParLaneElemStridedMut<'s, T>
where
    T: Send,
{
    fn len(&self) -> usize {
        self.0.len()
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(LaneElemStridedMutProducer {
            base: self.0.base,
            producer: LaneOffsetsProducer(self.0.offsets),
            lifetime: PhantomData,
        })
    }
}

#[cfg(feature = "rayon")]
impl<'s, T> ParLaneElemStridedMut<'s, T> {
    /// Creates a parallel, strided iterator over mutable references of the
    /// elements of `base` according to `geometry`.
    ///
    /// Returns `None` if any offset generated by `geometry` is out of bounds or
    /// overflows, or if `geometry` is not injective.
    pub fn new(base: &'s mut [T], geometry: LaneGeometry) -> Option<Self> {
        Some(Self(LaneElemStridedMut::new(base, geometry)?))
    }

    /// Creates a parallel, strided iterator over mutable references to the
    /// elements addressed by `geometry` relative to `base`.
    ///
    /// # Safety
    ///
    /// Every offset `geometry` addresses must be a valid index into the
    /// allocation `base` points into, those elements must be borrowed mutably
    /// for `'s`. No other live reference to them must exist, and `geometry`
    /// must be injective so no offset is yielded twice.
    pub(crate) unsafe fn from_raw(base: *mut T, geometry: LaneGeometry) -> Self {
        // SAFETY: requirements are the same as `LaneElemStridedMut::from_raw`,
        // and the caller guarantees that this is fine.
        unsafe { Self(LaneElemStridedMut::from_raw(base, geometry)) }
    }

    /// Returns the stride of the iterator.
    pub fn stride(&self) -> usize {
        self.0.stride()
    }
}

/// Producer for [`ParLaneElemStridedMut`].
#[cfg(feature = "rayon")]
struct LaneElemStridedMutProducer<'s, T> {
    /// Base pointer of the storage.
    base: *mut T,
    /// Producer for [`ParLaneOffsets`].
    producer: LaneOffsetsProducer,
    /// Lifetime marker for the mutable reference which is invariant in `T`.
    lifetime: PhantomData<&'s mut T>,
}

// SAFETY: grants unique access to `T`, so it is Send when `T` is Send.
#[cfg(feature = "rayon")]
unsafe impl<'s, T> Send for LaneElemStridedMutProducer<'s, T> where T: Send {}

// SAFETY: grants unique access to `T`, so it is Sync when `T` is Sync.
#[cfg(feature = "rayon")]
unsafe impl<'s, T> Sync for LaneElemStridedMutProducer<'s, T> where T: Sync {}

#[cfg(feature = "rayon")]
impl<'s, T> Producer for LaneElemStridedMutProducer<'s, T>
where
    T: Send,
{
    type Item = &'s mut T;
    type IntoIter = LaneElemStridedMut<'s, T>;

    fn into_iter(self) -> Self::IntoIter {
        LaneElemStridedMut {
            base: self.base,
            offsets: self.producer.into_iter(),
            lifetime: PhantomData,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.producer.split_at(index);

        // SAFETY: `LaneElemStridedMut` guarantees that any reference returned
        // from it is unique, so splitting it in two without overlap can also
        // only produce unique mutable references.
        (
            Self {
                base: self.base,
                producer: left,
                lifetime: PhantomData,
            },
            Self {
                base: self.base,
                producer: right,
                lifetime: PhantomData,
            },
        )
    }
}

/// Iterator over mutable references of contiguous elements of a buffer.
pub type LaneElemContiguousMut<'s, T> = std::slice::IterMut<'s, T>;

/// Parallel iterator over mutable references of contiguous elements of a
/// buffer.
#[cfg(feature = "rayon")]
pub type ParLaneElemContiguousMut<'s, T> = rayon::slice::IterMut<'s, T>;
