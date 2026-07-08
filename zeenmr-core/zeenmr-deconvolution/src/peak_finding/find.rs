use crate::error::Result;
use std::ops::Range;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Trait for finding peaks in a smoothed signal.
///
/// # Note for Implementors
///
/// A peak finding algorithm should specify in its documentation what the bounds
/// of a peak represent (e.g., inflection points).
pub trait Find<T>: Send + Sync {
    /// Finds peaks in the given smoothed signal within the specified bounds,
    /// optionally ignoring certain regions.
    fn find(
        &self,
        smoothed: &[T],
        signal: &Range<usize>,
        ignore: &[Range<usize>],
    ) -> Result<Vec<Peak>>;
}

/// A 1D NMR peak, represented by its left bound, center, and right bound.
///
/// It must hold that `left <= center <= right`.
///
/// The center point should always be the maximum, but what the left and right
/// bounds represent may change between different peak finding algorithms.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Peak {
    /// Left bound index of the peak.
    pub left: usize,
    /// Center index of the peak (the maximum).
    pub center: usize,
    /// Right bound index of the peak.
    pub right: usize,
}

impl Peak {
    /// Creates a new `Peak` from the given indices.
    pub fn new(left: usize, center: usize, right: usize) -> Self {
        debug_assert!(left < center && center < right);

        Self {
            left,
            center,
            right,
        }
    }
}
