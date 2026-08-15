use crate::intensity_array::iter::SplitAt;
use rayon::iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge};
use rayon::prelude::*;

/// Generalized parallel wrapper for the iterators of this module.
#[derive(Clone, Debug)]
pub struct Par<I>(I);

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
struct ParProducer<I>(I);

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
