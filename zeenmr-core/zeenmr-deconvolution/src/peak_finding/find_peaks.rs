use crate::peak_finding::Peak;
use num_traits::Float;
use zeenmr_spectrum::IndexRange;

pub trait FindPeaks<F>
where
    F: Float,
{
    fn find_peaks<I>(
        &self,
        smoothed: I,
        bounds: IndexRange,
        ignore: Option<&[IndexRange]>,
    ) -> impl Iterator<Item = Peak>
    where
        I: IntoIterator<Item = F>;
}
