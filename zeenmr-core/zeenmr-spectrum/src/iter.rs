//! Types and traits for iterators in this crate.

#[cfg(feature = "rayon")]
use rayon::iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Iterators that can be split into two halves at an index.
///
/// # Safety
///
/// `split_at` must yield at most `index` items on the left and the remainder,
/// if any, on the right, in the original order. Left and right must be
/// disjoint at the boundary in terms of yielded element enumeration. That is,
/// if we split at index `k`, left must only contain elements `[start, k)` and
/// right must only contain elements `[k, len)`.
pub unsafe trait SplitAt: Sized {
    /// Splits the iterator at the provided `index`.
    fn split_at(self, index: usize) -> (Self, Self);
}

/// Generalized parallel wrapper for the iterators of this module.
#[cfg(feature = "rayon")]
#[derive(Clone, Debug)]
pub struct Par<I>(I);

#[cfg(feature = "rayon")]
impl<I> ParallelIterator for Par<I>
where
    I: SplitAt + DoubleEndedIterator + ExactSizeIterator + Send,
    I::Item: Send,
{
    type Item = I::Item;

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
impl<I> IndexedParallelIterator for Par<I>
where
    I: SplitAt + DoubleEndedIterator + ExactSizeIterator + Send,
    I::Item: Send,
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
        callback.callback(ParProducer(self.0))
    }
}

#[cfg(feature = "rayon")]
impl<I> Par<I> {
    /// Wraps `iter`, producing a parallel version of it.
    ///
    /// Prefer the `par_` methods on the respective types of this module, though
    /// this can be used to convert an already acquired iterator.
    pub fn new(iter: I) -> Self {
        Par(iter)
    }
}

/// Producer for [`Par`].
#[cfg(feature = "rayon")]
struct ParProducer<I>(I);

#[cfg(feature = "rayon")]
impl<I> Producer for ParProducer<I>
where
    I: SplitAt + DoubleEndedIterator + ExactSizeIterator + Send,
    I::Item: Send,
{
    type Item = I::Item;
    type IntoIter = I;

    fn into_iter(self) -> Self::IntoIter {
        self.0
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.0.split_at(index);

        (Self(left), Self(right))
    }
}
