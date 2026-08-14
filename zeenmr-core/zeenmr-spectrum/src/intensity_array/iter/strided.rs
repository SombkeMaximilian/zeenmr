use std::iter::FusedIterator;

#[cfg(feature = "rayon")]
use rayon::iter::plumbing::{Consumer, ProducerCallback, UnindexedConsumer};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Sum type containing iterators over strided elements.
///
/// Splitting iteration with strides into the `stride = 1` and `stride = n > 1`
/// case allows for aggressive optimizations on the contiguous, fast path, while
/// also allowing for mitigation of the downsides in the cases with larger
/// strides.
#[derive(Clone, Debug)]
pub enum StridedIterKind<S1, SN> {
    /// Contiguous, fast path.
    Contiguous(S1),
    /// Strided, slow path.
    Strided(SN),
}

/// Delegates the implementation to `S1` and `SN` without having to write
/// infinitely many match statements.
macro_rules! delegate {
    ($self:ident, $method:ident($($arg:expr),*)) => {
        match $self {
            Self::Contiguous(iter) => iter.$method($($arg),*),
            Self::Strided(iter) => iter.$method($($arg),*),
        }
    };
}

// for the future: the `try` methods currently require the unstable trait `Try`
// to be named. we can delegate those methods once it becomes stable.
impl<S1, SN> Iterator for StridedIterKind<S1, SN>
where
    S1: Iterator,
    SN: Iterator<Item = S1::Item>,
{
    type Item = S1::Item;

    fn next(&mut self) -> Option<Self::Item> {
        delegate!(self, next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        delegate!(self, size_hint())
    }

    fn count(self) -> usize {
        delegate!(self, count())
    }

    fn last(self) -> Option<Self::Item> {
        delegate!(self, last())
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        delegate!(self, nth(n))
    }

    fn for_each<F>(self, f: F)
    where
        F: FnMut(Self::Item),
    {
        delegate!(self, for_each(f))
    }

    fn collect<B: FromIterator<Self::Item>>(self) -> B {
        delegate!(self, collect())
    }

    fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        delegate!(self, fold(init, f))
    }

    fn all<F>(&mut self, f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        delegate!(self, all(f))
    }

    fn any<F>(&mut self, f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        delegate!(self, any(f))
    }

    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        delegate!(self, find(predicate))
    }

    fn find_map<B, F>(&mut self, f: F) -> Option<B>
    where
        F: FnMut(Self::Item) -> Option<B>,
    {
        delegate!(self, find_map(f))
    }

    fn position<P>(&mut self, predicate: P) -> Option<usize>
    where
        P: FnMut(Self::Item) -> bool,
    {
        delegate!(self, position(predicate))
    }
}

impl<S1, SN> DoubleEndedIterator for StridedIterKind<S1, SN>
where
    S1: DoubleEndedIterator,
    SN: DoubleEndedIterator<Item = S1::Item>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        delegate!(self, next_back())
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        delegate!(self, nth_back(n))
    }

    fn rfold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        delegate!(self, rfold(init, f))
    }

    fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        delegate!(self, rfind(predicate))
    }
}

impl<S1, SN> ExactSizeIterator for StridedIterKind<S1, SN>
where
    S1: ExactSizeIterator,
    SN: ExactSizeIterator<Item = S1::Item>,
{
}

impl<S1, SN> FusedIterator for StridedIterKind<S1, SN>
where
    S1: FusedIterator,
    SN: FusedIterator<Item = S1::Item>,
{
}

#[cfg(feature = "rayon")]
impl<S1, SN> ParallelIterator for StridedIterKind<S1, SN>
where
    S1: ParallelIterator,
    SN: ParallelIterator<Item = S1::Item>,
{
    type Item = S1::Item;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        match self {
            Self::Contiguous(iter) => iter.drive_unindexed(consumer),
            Self::Strided(iter) => iter.drive_unindexed(consumer),
        }
    }

    fn opt_len(&self) -> Option<usize> {
        match self {
            Self::Contiguous(iter) => iter.opt_len(),
            Self::Strided(iter) => iter.opt_len(),
        }
    }
}

#[cfg(feature = "rayon")]
impl<S1, SN> IndexedParallelIterator for StridedIterKind<S1, SN>
where
    S1: IndexedParallelIterator,
    SN: IndexedParallelIterator<Item = S1::Item>,
{
    fn len(&self) -> usize {
        match self {
            Self::Contiguous(iter) => iter.len(),
            Self::Strided(iter) => iter.len(),
        }
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        match self {
            Self::Contiguous(iter) => iter.drive(consumer),
            Self::Strided(iter) => iter.drive(consumer),
        }
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        match self {
            Self::Contiguous(iter) => iter.with_producer(callback),
            Self::Strided(iter) => iter.with_producer(callback),
        }
    }
}

impl<S1, SN> StridedIterKind<S1, SN> {
    /// Returns the contained contiguous iterator, or `None` if `self` is
    /// `Strided`.
    pub fn try_into_contiguous(self) -> Option<S1> {
        match self {
            StridedIterKind::Contiguous(iter) => Some(iter),
            StridedIterKind::Strided(_) => None,
        }
    }

    /// Returns the contained strided iterator, or `None` if `self` is
    /// `Contiguous`.
    pub fn try_into_strided(self) -> Option<SN> {
        match self {
            StridedIterKind::Contiguous(_) => None,
            StridedIterKind::Strided(iter) => Some(iter),
        }
    }
}
