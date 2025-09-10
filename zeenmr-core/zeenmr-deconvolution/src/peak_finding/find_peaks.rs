use crate::peak_finding::Peak;
use num_traits::Float;
use zeenmr_spectrum::IndexRange;

/// Trait for finding peaks in a smoothed signal.
pub trait FindPeaks<F>
where
    F: Float,
{
    /// Finds peaks in the given smoothed signal within the specified bounds,
    /// optionally ignoring certain regions.
    fn find_peaks(
        &self,
        smoothed: Vec<F>,
        bounds: IndexRange,
        ignore: Option<&[IndexRange]>,
    ) -> impl Iterator<Item = Peak>;
}
