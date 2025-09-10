use num_traits::Float;
use zeenmr_peakshape::PeakShape;
use zeenmr_spectrum::Spectrum;

#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub trait FitPeakShapes<F, P>
where
    F: Float,
    P: PeakShape,
{
    fn fit_peak_shapes<I>(&self, spectrum: &Spectrum, peaks: I) -> impl Iterator<Item = P>
    where
        I: Iterator<Item = usize>;

    #[cfg(feature = "rayon")]
    fn par_fit_peak_shapes<I>(&self, spectrum: &Spectrum, peaks: I) -> impl ParallelIterator<Item = P>
    where
        I: IntoParallelIterator<Item = usize>;
}
