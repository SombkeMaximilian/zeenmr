use crate::error::Result;
use crate::peak_finding::Peak;
use std::ops::Range;

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
