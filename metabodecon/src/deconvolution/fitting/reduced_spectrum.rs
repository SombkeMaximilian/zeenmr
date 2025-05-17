use crate::deconvolution::peak_selection::Peak;
use crate::spectrum::Spectrum;

/// Data structure that contains a subset of the data from a spectrum.
#[derive(Debug)]
pub(crate) struct ReducedSpectrum {
    /// The chemical shifts of the reduced spectrum in ppm.
    chemical_shifts: Box<[f64]>,
    /// The intensities of the reduced spectrum in arbitrary units.
    intensities: Box<[f64]>,
}

impl ReducedSpectrum {
    /// Extracts the chemical shifts and intensities of the peaks from the
    /// spectrum and constructs a `ReducedSpectrum` from them.
    pub(crate) fn new(spectrum: &Spectrum, peaks: &[Peak]) -> Self {
        let chemical_shifts = peaks
            .iter()
            .flat_map(|peak| {
                vec![
                    spectrum.chemical_shifts()[peak.left()],
                    spectrum.chemical_shifts()[peak.center()],
                    spectrum.chemical_shifts()[peak.right()],
                ]
            })
            .collect();
        let intensities = peaks
            .iter()
            .flat_map(|peak| {
                vec![
                    spectrum.intensities()[peak.left()],
                    spectrum.intensities()[peak.center()],
                    spectrum.intensities()[peak.right()],
                ]
            })
            .collect();

        Self {
            chemical_shifts,
            intensities,
        }
    }

    /// Returns the chemical shifts as a slice.
    pub(crate) fn chemical_shifts(&self) -> &[f64] {
        &self.chemical_shifts
    }

    /// Returns the intensities as a slice.
    pub(crate) fn intensities(&self) -> &[f64] {
        &self.intensities
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn new() {
        let spectrum = Spectrum::new(
            (0..10).map(|x| 1.0 + x as f64).collect(),
            (0..10).map(|x| 10.0 - x as f64).collect(),
            (2.0, 9.0),
        )
        .unwrap();
        let peaks = vec![Peak::new(2, 3, 4), Peak::new(4, 5, 6), Peak::new(6, 7, 8)];
        let reduced_spectrum = ReducedSpectrum::new(&spectrum, &peaks);
        let expected_chemical_shifts = [3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 9.0];
        let expected_intensities = [8.0, 7.0, 6.0, 6.0, 5.0, 4.0, 4.0, 3.0, 2.0];
        reduced_spectrum
            .chemical_shifts()
            .iter()
            .zip(expected_chemical_shifts.iter())
            .for_each(|(&x, &xe)| {
                assert_approx_eq!(f64, x, xe);
            });
        reduced_spectrum
            .intensities()
            .iter()
            .zip(expected_intensities.iter())
            .for_each(|(&y, &ye)| {
                assert_approx_eq!(f64, y, ye);
            });
    }
}
