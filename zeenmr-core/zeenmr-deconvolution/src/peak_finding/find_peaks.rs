use crate::error::Result;
use crate::peak_finding::Peak;
use zeenmr_spectrum::IndexRange;

/// Trait for finding peaks in a smoothed signal.
///
/// # Note for Implementors
///
/// A peak finding algorithm should specify in its documentation what the bounds
/// of a peak represent (e.g., inflection points).
pub trait FindPeaks: Send + Sync {
    /// Finds peaks in the given smoothed signal within the specified bounds,
    /// optionally ignoring certain regions.
    fn find_peaks(
        &self,
        smoothed: &[f64],
        signal: IndexRange,
        ignore: Option<&[IndexRange]>,
    ) -> Result<Vec<Peak>>;
}
