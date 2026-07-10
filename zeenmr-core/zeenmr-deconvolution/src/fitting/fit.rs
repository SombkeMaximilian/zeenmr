use crate::peak_finding::Peak;
use zeenmr_peakshape::PeakShape;
use zeenmr_spectrum::Spectrum1D;
use zeenmr_spectrum::intensity_array::Storage;

/// Trait for fitting peak shapes to a spectrum, given a set of detected peaks.
pub trait Fit<T, P>: Send + Sync
where
    P: PeakShape<T>,
{
    /// Fits peak shapes to the spectrum using the provided peaks.
    fn fit<S>(&self, spectrum: &Spectrum1D<T, S>, peaks: &[Peak]) -> Vec<P>
    where
        S: Storage<Elem = T>;
}

/// Trait for fitting peak shapes to a spectrum in parallel, given a set of
/// detected peaks.
///
/// Some fitting algorithms can be efficiently parallelized for certain input
/// sizes.
#[cfg(feature = "rayon")]
pub trait ParFit<T, P>: Fit<T, P>
where
    P: PeakShape<T> + Send + Sync,
{
    /// Fits peak shapes to the spectrum in parallel using the provided peaks.
    fn par_fit<S>(&self, spectrum: &Spectrum1D<T, S>, peaks: &[Peak]) -> Vec<P>
    where
        S: Storage<Elem = T>;
}
