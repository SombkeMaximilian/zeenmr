use crate::error::Result;
use crate::peak_finding::Peak;
use num_traits::Float;
use std::iter::Sum;
use zeenmr_spectrum::IndexRange;

/// Trait for finding peaks in a smoothed signal.
pub trait FindPeaks<F>
where
    F: Float + Sum,
{
    /// Finds peaks in the given smoothed signal within the specified bounds,
    /// optionally ignoring certain regions.
    fn find_peaks(
        &self,
        smoothed: Vec<F>,
        signal: IndexRange,
        ignore: Option<&[IndexRange]>,
    ) -> Result<Vec<Peak>>;
}
