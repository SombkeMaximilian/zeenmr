use crate::spectrum::Spectrum;
use crate::spectrum::meta::{Nucleus, ReferenceCompound};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};

/// Form of [`Spectrum`] used for serialization/deserialization.
///
/// The reason for this transformation is that the regular [`Spectrum`] stores a
/// lot of information that is trivially derived from the data itself. One of
/// the invariants of [`Spectrum`] is that the chemical shifts are evenly spaced
/// within the spectrum boundaries. This means that the start, end and size of
/// the spectrum are enough to reconstruct the chemical shifts. Storing them is
/// still useful as they frequently need to be accessed. This form only contains
/// the essential information needed to reconstruct the original [`Spectrum`].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename = "Spectrum", rename_all = "camelCase")]
pub(crate) struct SerializedSpectrum {
    /// Spectrum boundaries in ppm.
    spectrum_boundaries: (f64, f64),
    /// Boundaries of the signal region in ppm.
    signal_boundaries: (f64, f64),
    /// Number of data points in the spectrum.
    size: usize,
    /// Nucleus observed in the NMR experiment.
    nucleus: Nucleus,
    /// Spectrometer frequency in MHz.
    frequency: f64,
    /// Chemical shift reference.
    reference_compound: ReferenceCompound,
    /// Intensities in arbitrary units.
    intensities: Vec<f64>,
}

impl<S: AsRef<Spectrum>> From<S> for SerializedSpectrum {
    fn from(value: S) -> Self {
        let spectrum = value.as_ref();

        Self {
            intensities: spectrum.intensities().to_vec(),
            size: spectrum.len(),
            spectrum_boundaries: spectrum.range(),
            signal_boundaries: spectrum.signal_boundaries(),
            nucleus: spectrum.nucleus(),
            frequency: spectrum.frequency(),
            reference_compound: spectrum.reference_compound().clone(),
        }
    }
}

impl TryFrom<SerializedSpectrum> for Spectrum {
    type Error = Error;

    fn try_from(value: SerializedSpectrum) -> Result<Spectrum> {
        let intensities = value.intensities;
        let size = value.size;
        let (start, end) = value.spectrum_boundaries;
        let signal_boundaries = value.signal_boundaries;
        let step = (end - start) / (size as f64 - 1.0);
        let chemical_shifts = (0..size)
            .map(|index| start + index as f64 * step)
            .collect();
        let mut spectrum = Spectrum::new(chemical_shifts, intensities, signal_boundaries)?;
        spectrum.set_nucleus(value.nucleus);
        spectrum.set_frequency(value.frequency);
        spectrum.set_reference_compound(value.reference_compound);

        Ok(spectrum)
    }
}

#[cfg(test)]
impl PartialEq for SerializedSpectrum {
    fn eq(&self, other: &Self) -> bool {
        use float_cmp::approx_eq;
        self.size == other.size
            && approx_eq!(f64, self.spectrum_boundaries.0, other.spectrum_boundaries.0)
            && approx_eq!(f64, self.spectrum_boundaries.1, other.spectrum_boundaries.1)
            && approx_eq!(f64, self.signal_boundaries.0, other.signal_boundaries.0)
            && approx_eq!(f64, self.signal_boundaries.1, other.signal_boundaries.1)
            && self
                .intensities
                .iter()
                .zip(other.intensities.iter())
                .all(|(a, b)| approx_eq!(f64, *a, *b))
    }
}

#[cfg(test)]
impl SerializedSpectrum {
    /// Creates a valid `SerializedSpectrum` with 2^n resolution for testing.
    fn valid(resolution: u32) -> Self {
        let intensities = (0..2_u32.pow(resolution))
            .map(|i| i as f64 * 10.0 / (2_f64.powi(resolution as i32) - 1.0))
            .map(|x| {
                1.0 * 0.25 / (0.25_f64.powi(2) + (x - 3.0).powi(2))
                    + 1.0 * 0.25 / (0.25_f64.powi(2) + (x - 7.0).powi(2))
            })
            .collect();

        Self {
            intensities,
            size: 2_usize.pow(resolution),
            spectrum_boundaries: (0.0, 10.0),
            signal_boundaries: (1.0, 9.0),
            nucleus: Nucleus::Hydrogen1,
            frequency: 400.0,
            reference_compound: ReferenceCompound::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectrum::error::Kind;
    use float_cmp::assert_approx_eq;

    #[test]
    fn spectrum_conversion_forward() {
        let initial = SerializedSpectrum::valid(8);
        let recovered = SerializedSpectrum::from(Spectrum::try_from(initial.clone()).unwrap());
        assert_eq!(initial, recovered);
    }

    #[test]
    fn spectrum_conversion_backward() {
        let chemical_shifts = (0..2_u32.pow(8))
            .map(|i| i as f64 * 10.0 / (2_f64.powi(8) - 1.0))
            .collect::<Vec<f64>>();
        let intensities = chemical_shifts
            .iter()
            .map(|x| {
                1.0 * 0.25 / (0.25_f64.powi(2) + (x - 3.0).powi(2))
                    + 1.0 * 0.25 / (0.25_f64.powi(2) + (x - 7.0).powi(2))
            })
            .collect::<Vec<f64>>();
        let signal_boundaries = (1.0, 9.0);
        let initial = Spectrum::new(chemical_shifts, intensities, signal_boundaries).unwrap();
        let recovered = Spectrum::try_from(SerializedSpectrum::from(&initial)).unwrap();
        initial
            .chemical_shifts()
            .iter()
            .zip(recovered.chemical_shifts())
            .for_each(|(init, rec)| {
                assert_approx_eq!(f64, *init, *rec);
            });
        initial
            .intensities()
            .iter()
            .zip(recovered.intensities())
            .for_each(|(init, rec)| {
                assert_approx_eq!(f64, *init, *rec);
            });
        assert_approx_eq!(
            f64,
            initial.signal_boundaries().0,
            recovered.signal_boundaries().0
        );
        assert_approx_eq!(
            f64,
            initial.signal_boundaries().1,
            recovered.signal_boundaries().1
        );
    }

    #[test]
    fn empty_data() {
        let mut invalid = [
            SerializedSpectrum::valid(2),
            SerializedSpectrum::valid(2),
            SerializedSpectrum::valid(2),
        ];
        invalid[0].size = 0;
        invalid[1].intensities.clear();
        invalid[2].size = 0;
        invalid[2].intensities.clear();
        let errors = [
            Spectrum::try_from(invalid[0].clone()).unwrap_err(),
            Spectrum::try_from(invalid[1].clone()).unwrap_err(),
            Spectrum::try_from(invalid[2].clone()).unwrap_err(),
        ];
        let expected_context = [(0, 4), (4, 0), (0, 0)];
        errors
            .into_iter()
            .zip(expected_context)
            .for_each(|(error, context)| match error {
                Error::Spectrum(inner) => match inner.kind() {
                    Kind::EmptyData {
                        chemical_shifts,
                        intensities,
                    } => {
                        assert_eq!(*chemical_shifts, context.0);
                        assert_eq!(*intensities, context.1);
                    }
                    _ => panic!("unexpected kind: {:?}", inner),
                },
                _ => panic!("unexpected error: {:?}", error),
            });
    }

    #[test]
    fn data_length_mismatch() {
        let mut invalid = [SerializedSpectrum::valid(2), SerializedSpectrum::valid(2)];
        invalid[0].intensities.pop();
        invalid[1].size -= 1;
        let errors = [
            Spectrum::try_from(invalid[0].clone()).unwrap_err(),
            Spectrum::try_from(invalid[1].clone()).unwrap_err(),
        ];
        let expected_context = [(4, 3), (3, 4)];
        errors
            .into_iter()
            .zip(expected_context)
            .for_each(|(error, context)| match error {
                Error::Spectrum(inner) => match inner.kind() {
                    Kind::DataLengthMismatch {
                        chemical_shifts,
                        intensities,
                    } => {
                        assert_eq!(*chemical_shifts, context.0);
                        assert_eq!(*intensities, context.1);
                    }
                    _ => panic!("unexpected kind: {:?}", inner),
                },
                _ => panic!("unexpected error: {:?}", error),
            });
    }

    #[test]
    fn non_uniform_spacing() {
        let mut invalid = SerializedSpectrum::valid(2);
        invalid.spectrum_boundaries.0 = f64::NAN;
        let error = Spectrum::try_from(invalid.clone()).unwrap_err();
        let expected_context = (f64::NAN, (0, 1));
        match error {
            Error::Spectrum(inner) => match inner.kind() {
                Kind::NonUniformSpacing {
                    step_size,
                    positions,
                } => {
                    assert_eq!(step_size.is_nan(), expected_context.0.is_nan());
                    assert_eq!(step_size.is_infinite(), expected_context.0.is_infinite());
                    assert_eq!(
                        step_size.is_sign_positive(),
                        expected_context.0.is_sign_positive()
                    );
                    if step_size.is_finite() && expected_context.0.is_finite() {
                        assert_approx_eq!(f64, *step_size, expected_context.0);
                    };
                    assert_eq!(*positions, expected_context.1);
                }
                _ => panic!("unexpected kind: {:?}", inner),
            },
            _ => panic!("unexpected error: {:?}", error),
        };
    }

    #[test]
    fn invalid_intensities() {
        let mut invalid = [
            SerializedSpectrum::valid(4),
            SerializedSpectrum::valid(4),
            SerializedSpectrum::valid(4),
        ];
        invalid[0].intensities.push(f64::NAN);
        invalid[0].size += 1;
        invalid[1].intensities.extend([f64::INFINITY; 5]);
        invalid[1].size += 5;
        invalid[2]
            .intensities
            .extend([f64::NEG_INFINITY; 10]);
        invalid[2].size += 10;
        let errors = [
            Spectrum::try_from(invalid[0].clone()).unwrap_err(),
            Spectrum::try_from(invalid[1].clone()).unwrap_err(),
            Spectrum::try_from(invalid[2].clone()).unwrap_err(),
        ];
        let expected_context = [
            (vec![16], 1),
            ((16..21).collect(), 5),
            ((16..26).collect(), 10),
        ];
        errors
            .into_iter()
            .zip(expected_context)
            .for_each(|(error, context)| {
                match error {
                    Error::Spectrum(inner) => match inner.kind() {
                        Kind::InvalidIntensities { positions } => {
                            assert_eq!(*positions, context.0);
                            assert_eq!(positions.len(), context.1);
                        }
                        _ => panic!("unexpected kind: {:?}", inner),
                    },
                    _ => panic!("unexpected error: {:?}", error),
                };
            });
    }

    #[test]
    fn invalid_signal_boundaries() {
        let r = (0.0, 10.0);
        let mut invalid = [
            SerializedSpectrum::valid(4),
            SerializedSpectrum::valid(4),
            SerializedSpectrum::valid(4),
            SerializedSpectrum::valid(4),
            SerializedSpectrum::valid(4),
        ];
        invalid[0].signal_boundaries.0 = f64::NAN;
        invalid[1].signal_boundaries.0 = -1.0;
        invalid[2].signal_boundaries.1 = 11.0;
        invalid[3].signal_boundaries = (7.0, 7.0);
        invalid[4].signal_boundaries = (11.0, 11.0);
        let errors = [
            Spectrum::try_from(invalid[0].clone()).unwrap_err(),
            Spectrum::try_from(invalid[1].clone()).unwrap_err(),
            Spectrum::try_from(invalid[2].clone()).unwrap_err(),
            Spectrum::try_from(invalid[3].clone()).unwrap_err(),
            Spectrum::try_from(invalid[4].clone()).unwrap_err(),
        ];
        let expected_context = [
            ((f64::NAN, 9.0), r),
            ((-1.0, 9.0), r),
            ((1.0, 11.0), r),
            ((7.0, 7.0), r),
            ((11.0, 11.0), r),
        ];
        errors
            .into_iter()
            .zip(expected_context)
            .for_each(|(error, context)| {
                match error {
                    Error::Spectrum(inner) => match inner.kind() {
                        Kind::InvalidSignalBoundaries {
                            signal_boundaries,
                            chemical_shifts_range,
                        } => {
                            assert_approx_eq!(f64, signal_boundaries.0, context.0.0);
                            assert_approx_eq!(f64, signal_boundaries.1, context.0.1);
                            assert_approx_eq!(f64, chemical_shifts_range.0, context.1.0);
                            assert_approx_eq!(f64, chemical_shifts_range.1, context.1.1);
                        }
                        _ => panic!("unexpected kind: {:?}", inner),
                    },
                    _ => panic!("unexpected error: {:?}", error),
                };
            });
    }
}
