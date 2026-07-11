use crate::peak_finding::Peak;
use zeenmr_peakshape::PeakShape;
use zeenmr_spectrum::SpectrumView1D;

/// Trait for fitting peak shapes to a spectrum, given a set of detected peaks.
///
/// # Notes for Implementors
///
/// It can be assumed that the `spectrum` contains no floating point `NaN` or
/// either infinity, and that `left < center < right` holds for all `peaks`.
///
/// It can be useful for type inferences to include a `PhantomData<fn() -> P>`
/// or similar in the type implementing this trait in order to guide type
/// inference while constructing a deconvoluter.
pub trait Fit<T, P>
where
    P: PeakShape<T>,
{
    /// Error type when an error occurs during fitting.
    type Error;

    /// Fits peak shapes to the spectrum using the provided peaks.
    fn fit(&self, spectrum: SpectrumView1D<T, T>, peaks: &[Peak]) -> Result<Vec<P>, Self::Error>;
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
    /// Error type when an error occurs during fitting.
    type Error;

    /// Fits peak shapes to the spectrum in parallel using the provided peaks.
    fn par_fit(
        &self,
        spectrum: SpectrumView1D<T, T>,
        peaks: &[Peak],
    ) -> Result<Vec<P>, Self::Error>;
}
