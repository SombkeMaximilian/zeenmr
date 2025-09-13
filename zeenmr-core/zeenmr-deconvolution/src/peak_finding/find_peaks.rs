use crate::error::Result;
use crate::peak_finding::Peak;
use zeenmr_spectrum::IndexRange;

/// Trait for finding peaks in a smoothed signal.
pub trait FindPeaks {
    /// Finds peaks in the given smoothed signal within the specified bounds,
    /// optionally ignoring certain regions.
    fn find_peaks(
        &self,
        smoothed: Vec<f64>,
        signal: IndexRange,
        ignore: Option<&[IndexRange]>,
    ) -> Result<Vec<Peak>>;
}
