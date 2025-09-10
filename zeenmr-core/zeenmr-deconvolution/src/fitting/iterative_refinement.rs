use crate::peak_finding::Peak;
use num_traits::Float;
use zeenmr_spectrum::Spectrum;

/// A reduced representation of a spectrum that only contains the data points
/// that are part of peaks.
#[derive(Debug)]
struct ReducedSpectrum<F> {
    /// Peaks found in the original spectrum.
    peaks: Vec<Peak>,
    /// Intensity values that are part of the peaks.
    intensities: Vec<(F, F, F)>,
}

impl<F> ReducedSpectrum<F>
where
    F: Float,
{
    /// Extracts the positions and intensities of the peaks from the spectrum
    /// and constructs a `ReducedSpectrum` from them.
    fn new<I>(spectrum: &Spectrum, peaks: Vec<Peak>) -> Self {
        let intensities = peaks
            .iter()
            .map(|peak| {
                (
                    // this always gets optimized away because spectrum only
                    // supports f64 at this time, so none of the deconvoluter
                    // components can be anything else but f64. once spectrum
                    // is generic over a float type, these conversions can be
                    // removed.
                    F::from(spectrum.intensities()[peak.left]).unwrap(),
                    F::from(spectrum.intensities()[peak.center]).unwrap(),
                    F::from(spectrum.intensities()[peak.right]).unwrap(),
                )
            })
            .collect();

        Self {
            peaks,
            intensities,
        }
    }

    /// Returns the positions of the intensities in the original [`Spectrum`].
    fn peaks(&self) -> &[Peak] {
        &self.peaks
    }

    /// Returns the intensities of the original [`Spectrum`] that are part of
    /// peaks.
    fn intensities(&self) -> &[(F, F, F)] {
        &self.intensities
    }
}
