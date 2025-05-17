use crate::deconvolution::Deconvolution;
use crate::deconvolution::fitting::FittingSettings;
use crate::deconvolution::lorentzian::Lorentzian;
use crate::deconvolution::peak_selection::SelectionSettings;
use crate::deconvolution::smoothing::SmoothingSettings;
use crate::{Error, Result, Settings};
use serde::{Deserialize, Serialize};

/// Form of [`Deconvolution`] used for serialization/deserialization.
///
/// [`Arc`] is used to store the [`Lorentzian`]s within [`Deconvolution`] to
/// allow for efficient cloning and sharing of the deconvoluted signals for the
/// alignment process. [`Arc`] can cause issues with serialization, so the
/// [`Deconvolution`] struct is converted to this form, where the `Lorentzian`s
/// are stored as a `Vec` instead.
///
/// [`Arc`]: std::sync::Arc
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename = "Deconvolution", rename_all = "camelCase")]
pub(crate) struct SerializedDeconvolution {
    /// The smoothing parameters used.
    smoothing_settings: SmoothingSettings,
    /// The peak selection parameters used.
    selection_settings: SelectionSettings,
    /// The fitting parameters used.
    fitting_settings: FittingSettings,
    /// The mean squared error of the deconvolution.
    mse: f64,
    /// The deconvoluted signals.
    lorentzians: Vec<Lorentzian>,
}

impl<D: AsRef<Deconvolution>> From<D> for SerializedDeconvolution {
    fn from(value: D) -> Self {
        let deconvolution = value.as_ref();

        Self {
            smoothing_settings: deconvolution.smoothing_settings(),
            selection_settings: deconvolution.selection_settings(),
            fitting_settings: deconvolution.fitting_settings(),
            mse: deconvolution.mse(),
            lorentzians: deconvolution.lorentzians().to_vec(),
        }
    }
}

impl TryFrom<SerializedDeconvolution> for Deconvolution {
    type Error = Error;

    fn try_from(value: SerializedDeconvolution) -> Result<Deconvolution> {
        value.smoothing_settings.validate()?;
        value.selection_settings.validate()?;
        value.fitting_settings.validate()?;

        Ok(Deconvolution::new(
            value.lorentzians,
            value.smoothing_settings,
            value.selection_settings,
            value.fitting_settings,
            value.mse,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deconvolution::peak_selection::ScoringMethod;
    use float_cmp::assert_approx_eq;

    #[test]
    fn deconvolution_conversion_forward() {
        let lorentzians = vec![
            Lorentzian::new(5.5, 0.25, 3.0),
            Lorentzian::new(7.0, 0.16, 5.0),
            Lorentzian::new(5.5, 0.25, 7.0),
        ];
        let initial = SerializedDeconvolution {
            lorentzians,
            smoothing_settings: SmoothingSettings::default(),
            selection_settings: SelectionSettings::default(),
            fitting_settings: FittingSettings::default(),
            mse: 0.5,
        };
        let recovered =
            SerializedDeconvolution::from(Deconvolution::try_from(initial.clone()).unwrap());
        initial
            .lorentzians
            .iter()
            .zip(recovered.lorentzians.iter())
            .for_each(|(initial, recovered)| {
                assert_approx_eq!(f64, initial.sfhw(), recovered.sfhw());
                assert_approx_eq!(f64, initial.hw2(), recovered.hw2());
                assert_approx_eq!(f64, initial.maxp(), recovered.maxp());
            });
        match recovered.smoothing_settings {
            SmoothingSettings::Identity => unreachable!(),
            SmoothingSettings::MovingAverage {
                iterations,
                window_size,
            } => {
                assert_eq!(iterations, 3);
                assert_eq!(window_size, 3);
            }
        };
        match recovered.selection_settings {
            SelectionSettings::DetectorOnly => unreachable!("expected noise score filter"),
            SelectionSettings::NoiseScoreFilter {
                scoring_method,
                threshold,
            } => {
                match scoring_method {
                    ScoringMethod::MinimumSum => {}
                }
                assert_approx_eq!(f64, threshold, 5.0);
            }
        };
        match recovered.fitting_settings {
            FittingSettings::Analytical { iterations } => {
                assert_eq!(iterations, 10);
            }
        };
    }

    #[test]
    fn deconvolution_conversion_backward() {
        let lorentzians = vec![
            Lorentzian::new(5.5, 0.25, 3.0),
            Lorentzian::new(7.0, 0.16, 5.0),
            Lorentzian::new(5.5, 0.25, 7.0),
        ];
        let initial = Deconvolution::new(
            lorentzians.clone(),
            SmoothingSettings::default(),
            SelectionSettings::default(),
            FittingSettings::default(),
            0.5,
        );
        let recovered =
            Deconvolution::try_from(SerializedDeconvolution::from(initial.clone())).unwrap();
        initial
            .lorentzians()
            .iter()
            .zip(recovered.lorentzians().iter())
            .for_each(|(init, rec)| {
                assert_approx_eq!(f64, init.sfhw(), rec.sfhw());
                assert_approx_eq!(f64, init.hw2(), rec.hw2());
                assert_approx_eq!(f64, init.maxp(), rec.maxp());
            });
        match recovered.smoothing_settings() {
            SmoothingSettings::Identity => panic!("expected moving average"),
            SmoothingSettings::MovingAverage {
                iterations,
                window_size,
            } => {
                assert_eq!(iterations, 3);
                assert_eq!(window_size, 3);
            }
        };
        match recovered.selection_settings() {
            SelectionSettings::DetectorOnly => panic!("expected noise score filter"),
            SelectionSettings::NoiseScoreFilter {
                scoring_method,
                threshold,
            } => {
                match scoring_method {
                    ScoringMethod::MinimumSum => {}
                }
                assert_approx_eq!(f64, threshold, 5.0);
            }
        };
        match recovered.fitting_settings() {
            FittingSettings::Analytical { iterations } => {
                assert_eq!(iterations, 10);
            }
        };
    }
}
