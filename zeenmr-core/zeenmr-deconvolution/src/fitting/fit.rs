use crate::peak_finding::Peak;
use zeenmr_peakshape::PeakShape;
use zeenmr_spectrum::Spectrum;

/// Trait for fitting peak shapes to a spectrum, given a set of detected peaks.
pub trait Fit<P>: Send + Sync
where
    P: PeakShape + Send + Sync,
{
    /// Fits peak shapes to the spectrum using the provided peaks.
    fn fit(&self, spectrum: &Spectrum, peaks: &[Peak]) -> Vec<P>;
}

/// Trait for fitting peak shapes to a spectrum in parallel, given a set of
/// detected peaks.
///
/// Some fitting algorithms can be efficiently parallelized for certain input
/// sizes.
#[cfg(feature = "rayon")]
pub trait ParFit<P>: Fit<P>
where
    P: PeakShape + Send + Sync,
{
    /// Fits peak shapes to the spectrum in parallel using the provided peaks.
    fn par_fit(&self, spectrum: &Spectrum, peaks: &[Peak]) -> Vec<P>;
}
