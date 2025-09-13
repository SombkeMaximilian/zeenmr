use crate::peak_finding::Peak;
use zeenmr_spectrum::Spectrum;

/// A reduced representation of a spectrum that only contains the data points
/// that are part of peaks.
#[derive(Debug)]
struct ReducedSpectrum {
    /// Peaks found in the original spectrum.
    peaks: Vec<Peak>,
    /// Intensity values that are part of the peaks.
    intensities: Vec<(f64, f64, f64)>,
}

impl ReducedSpectrum {
    /// Extracts the positions and intensities of the peaks from the spectrum
    /// and constructs a `ReducedSpectrum` from them.
    fn new<I>(spectrum: &Spectrum, peaks: Vec<Peak>) -> Self {
        let intensities = peaks
            .iter()
            .map(|peak| {
                (
                    spectrum.intensities()[peak.left],
                    spectrum.intensities()[peak.center],
                    spectrum.intensities()[peak.right],
                )
            })
            .collect();

        Self { peaks, intensities }
    }

    /// Returns the positions of the intensities in the original [`Spectrum`].
    fn peaks(&self) -> &[Peak] {
        &self.peaks
    }

    /// Returns the intensities of the original [`Spectrum`] that are part of
    /// peaks.
    fn intensities(&self) -> &[(f64, f64, f64)] {
        &self.intensities
    }
}
