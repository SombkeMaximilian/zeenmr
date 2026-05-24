use crate::peak_finding::Peak;
use zeenmr_peakshape::PeakShape;
use zeenmr_spectrum::Spectrum;

/// Trait for fitting peak shapes to a spectrum given a set of detected peaks.
pub trait Fit<P>: Send + Sync
where
    P: PeakShape + Send + Sync,
{
    /// Fits peak shapes to the spectrum using the provided peaks.
    fn fit(&self, spectrum: &Spectrum, peaks: &[Peak]) -> Vec<P>;

    /// Fits peak shapes to the spectrum in parallel using the provided peaks.
    #[cfg(feature = "rayon")]
    fn par_fit(&self, spectrum: &Spectrum, peaks: &[Peak]) -> Vec<P>;
}
