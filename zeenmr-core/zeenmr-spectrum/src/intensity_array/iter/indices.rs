use crate::dimension::Dimension;
use crate::intensity_array::{ArrayIndex, Shape};
use crate::iter::SplitAt;
use std::iter::FusedIterator;

/// Iterator over the multidimensional indices of a shape.
///
/// Yields indices in lexicographic order: the last dimension varies fastest.
#[derive(Clone, Debug)]
pub struct Indices<D> {
    /// Underlying shape.
    shape: Shape<D>,
    /// Next linear index from the front.
    front: usize,
    /// Next linear index from the back.
    back: usize,
    /// Index at `front`, if already computed.
    front_cached: Option<ArrayIndex<D>>,
    /// Index at `back - 1`, if already computed.
    back_cached: Option<ArrayIndex<D>>,
}

impl<D> Iterator for Indices<D>
where
    D: Dimension<Elem = usize>,
{
    type Item = ArrayIndex<D>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let index = self
                .front_cached
                .get_or_insert_with(|| ArrayIndex::linear_in_shape(self.front, &self.shape));
            let curr = index.clone();
            index.increment_lexicographic(self.shape.as_slice());
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

impl<D> DoubleEndedIterator for Indices<D>
where
    D: Dimension<Elem = usize>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            self.back -= 1;
            let index = self
                .back_cached
                .get_or_insert_with(|| ArrayIndex::linear_in_shape(self.back, &self.shape));
            let curr = index.clone();
            // wraps to zero on the last element, which `front == back` then
            // makes unreachable
            index.decrement_lexicographic(self.shape.as_slice());

            Some(curr)
        } else {
            None
        }
    }
}

impl<D> ExactSizeIterator for Indices<D> where D: Dimension<Elem = usize> {}

impl<D> FusedIterator for Indices<D> where D: Dimension<Elem = usize> {}

// SAFETY: setting left.back and right.front to `self.front + index` or
// `self.back`, whichever is smaller, achieves the disjoint split required by
// `SplitAt`.
unsafe impl<D> SplitAt for Indices<D>
where
    D: Dimension<Elem = usize>,
{
    fn split_at(self, index: usize) -> (Self, Self) {
        let mid = (self.front + index).max(self.back);
        let left = Self {
            shape: self.shape.clone(),
            front: self.front,
            back: mid,
            front_cached: self.front_cached,
            back_cached: None,
        };
        let right = Self {
            shape: self.shape,
            front: mid,
            back: self.back,
            front_cached: None,
            back_cached: self.back_cached,
        };

        (left, right)
    }
}

impl<D> Indices<D>
where
    D: Dimension<Elem = usize>,
{
    /// Creates a lexicographic index iterator.
    ///
    /// Returns `None` in the same situations that [`Shape::product_checked`]
    /// returns `None`.
    ///
    /// Prefer [`Shape::indices_lexicographic`].
    pub fn new(shape: Shape<D>) -> Option<Self> {
        let back = shape.product_checked()?;

        Some(Self {
            shape,
            front: 0,
            back,
            front_cached: None,
            back_cached: None,
        })
    }
}
