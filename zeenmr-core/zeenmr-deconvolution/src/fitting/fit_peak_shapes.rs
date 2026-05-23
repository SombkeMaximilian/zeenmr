use crate::peak_finding::Peak;
use zeenmr_peakshape::PeakShape;
use zeenmr_spectrum::Spectrum;

#[cfg(feature = "rayon")]
use rayon::iter::IndexedParallelIterator;

/// Trait for fitting peak shapes to a spectrum given a set of detected peaks.
pub trait FitPeakShapes<P>: Send + Sync
where
    P: PeakShape + Send + Sync,
{
    /// Fits peak shapes to the spectrum using the provided peaks.
    fn fit_peak_shapes<I>(&self, spectrum: &Spectrum, peaks: I) -> impl Iterator<Item = P>
    where
        I: IntoIterator<Item = Peak>;

    /// Fits peak shapes to the spectrum in parallel using the provided peaks.
    #[cfg(feature = "rayon")]
    fn par_fit_peak_shapes<I>(
        &self,
        spectrum: &Spectrum,
        peaks: I,
    ) -> impl IndexedParallelIterator<Item = P>
    where
        I: IntoIterator<Item = Peak>;
}
