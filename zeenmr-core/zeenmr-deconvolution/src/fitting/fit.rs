use crate::peak_finding::Peak;
use zeenmr_peakshape::PeakShape;
use zeenmr_spectrum::SpectrumView1D;

/// Trait for fitting peak shapes to a spectrum, given a set of detected peaks.
pub trait Fit<T, P>: Send + Sync
where
    P: PeakShape<T>,
{
    /// Fits peak shapes to the spectrum using the provided peaks.
    fn fit(&self, spectrum: SpectrumView1D<T, T>, peaks: &[Peak]) -> Vec<P>;
}

/// Trait for fitting peak shapes to a spectrum in parallel, given a set of
/// detected peaks.
///
/// Some fitting algorithms can be efficiently parallelized for certain input
/// sizes.
#[cfg(feature = "rayon")]
pub trait ParFit<T, P>
where
    P: PeakShape<T> + Send + Sync,
{
    /// Fits peak shapes to the spectrum in parallel using the provided peaks.
    fn par_fit(&self, spectrum: SpectrumView1D<T, T>, peaks: &[Peak]) -> Vec<P>;
}
