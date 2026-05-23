#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A 1D NMR peak, represented by its left bound, center, and right bound.
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
