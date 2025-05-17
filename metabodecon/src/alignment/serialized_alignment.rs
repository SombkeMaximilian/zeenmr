use crate::alignment::{Alignment, AlignmentStrategy, FilteringSettings, SolvingSettings};
use crate::deconvolution::Deconvolution;
use crate::{Error, Result, Settings};
use serde::{Deserialize, Serialize};

/// Form of [`Alignment`] used for serialization/deserialization.
///
/// [`Arc`] is used to store the [`Deconvolution`]s within [`Alignment`] to
/// allow for efficient cloning and sharing of the aligned deconvolutions.
/// [`Arc`] can cause issues with serialization, so the [`Alignment`] struct is
/// converted to this form, where the `Deconvolution`s are stored as a `Vec`
/// instead.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename = "Alignment", rename_all = "camelCase")]
pub(crate) struct SerializedAlignment {
    /// Strategy used for aligning the deconvolutions.
    alignment_strategy: AlignmentStrategy,
    /// Settings for filtering the feature maps.
    filtering_settings: FilteringSettings,
    /// Settings for solving the assignment problem.
    solving_settings: SolvingSettings,
    /// Aligned `Deconvolution`s.
    deconvolutions: Vec<Deconvolution>,
}

impl<A: AsRef<Alignment>> From<A> for SerializedAlignment {
    fn from(value: A) -> Self {
        let alignment = value.as_ref();

        Self {
            alignment_strategy: alignment.alignment_strategy(),
            filtering_settings: alignment.filtering_settings(),
            solving_settings: alignment.solving_settings(),
            deconvolutions: alignment.deconvolutions().to_vec(),
        }
    }
}

impl TryFrom<SerializedAlignment> for Alignment {
    type Error = Error;

    fn try_from(value: SerializedAlignment) -> Result<Alignment> {
        value.alignment_strategy.validate()?;
        value.filtering_settings.validate()?;
        value.solving_settings.validate()?;

        Ok(Alignment::new(
            value.deconvolutions,
            value.alignment_strategy,
            value.filtering_settings,
            value.solving_settings,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deconvolution::{FittingSettings, Lorentzian, SelectionSettings, SmoothingSettings};
    use float_cmp::assert_approx_eq;

    #[test]
    fn alignment_conversion_forward() {
        let deconvolutions = vec![
            Deconvolution::new(
                vec![
                    Lorentzian::new(5.5, 0.25, 3.0),
                    Lorentzian::new(7.0, 0.16, 5.0),
                    Lorentzian::new(5.5, 0.25, 7.0),
                ],
                SmoothingSettings::default(),
                SelectionSettings::default(),
                FittingSettings::default(),
                0.5,
            ),
            Deconvolution::new(
                vec![
                    Lorentzian::new(5.3, 0.30, 3.0),
                    Lorentzian::new(6.8, 0.18, 5.0),
                    Lorentzian::new(5.7, 0.22, 7.0),
                ],
                SmoothingSettings::default(),
                SelectionSettings::default(),
                FittingSettings::default(),
                0.6,
            ),
        ];
        let initial = SerializedAlignment {
            deconvolutions,
            alignment_strategy: AlignmentStrategy::default(),
            filtering_settings: FilteringSettings::default(),
            solving_settings: SolvingSettings::default(),
        };
        let recovered = SerializedAlignment::from(Alignment::try_from(initial.clone()).unwrap());
        initial
            .deconvolutions
            .iter()
            .zip(recovered.deconvolutions.iter())
            .for_each(|(init_deconvolution, rec_deconvolution)| {
                init_deconvolution
                    .lorentzians()
                    .iter()
                    .zip(rec_deconvolution.lorentzians().iter())
                    .for_each(|(init, rec)| {
                        assert_approx_eq!(f64, init.sfhw(), rec.sfhw());
                        assert_approx_eq!(f64, init.hw2(), rec.hw2());
                        assert_approx_eq!(f64, init.maxp(), rec.maxp());
                    });
                assert!(SmoothingSettings::compare(
                    &init_deconvolution.smoothing_settings(),
                    &rec_deconvolution.smoothing_settings()
                ));
                assert!(SelectionSettings::compare(
                    &init_deconvolution.selection_settings(),
                    &rec_deconvolution.selection_settings()
                ));
                assert!(FittingSettings::compare(
                    &init_deconvolution.fitting_settings(),
                    &rec_deconvolution.fitting_settings()
                ));
            });
        assert!(AlignmentStrategy::compare(
            &initial.alignment_strategy,
            &recovered.alignment_strategy
        ));
        assert!(FilteringSettings::compare(
            &initial.filtering_settings,
            &recovered.filtering_settings
        ));
        assert!(SolvingSettings::compare(
            &initial.solving_settings,
            &recovered.solving_settings
        ));
    }

    #[test]
    fn alignment_conversion_backward() {
        let deconvolutions = vec![
            Deconvolution::new(
                vec![
                    Lorentzian::new(5.5, 0.25, 3.0),
                    Lorentzian::new(7.0, 0.16, 5.0),
                    Lorentzian::new(5.5, 0.25, 7.0),
                ],
                SmoothingSettings::default(),
                SelectionSettings::default(),
                FittingSettings::default(),
                0.5,
            ),
            Deconvolution::new(
                vec![
                    Lorentzian::new(5.3, 0.30, 3.0),
                    Lorentzian::new(6.8, 0.18, 5.0),
                    Lorentzian::new(5.7, 0.22, 7.0),
                ],
                SmoothingSettings::default(),
                SelectionSettings::default(),
                FittingSettings::default(),
                0.6,
            ),
        ];
        let initial = Alignment::new(
            deconvolutions.clone(),
            AlignmentStrategy::default(),
            FilteringSettings::default(),
            SolvingSettings::default(),
        );
        let recovered = Alignment::try_from(SerializedAlignment::from(initial.clone())).unwrap();
        initial
            .deconvolutions()
            .iter()
            .zip(recovered.deconvolutions().iter())
            .for_each(|(init_deconvolution, rec_deconvolution)| {
                init_deconvolution
                    .lorentzians()
                    .iter()
                    .zip(rec_deconvolution.lorentzians().iter())
                    .for_each(|(init, rec)| {
                        assert_approx_eq!(f64, init.sfhw(), rec.sfhw());
                        assert_approx_eq!(f64, init.hw2(), rec.hw2());
                        assert_approx_eq!(f64, init.maxp(), rec.maxp());
                    });
                assert!(SmoothingSettings::compare(
                    &init_deconvolution.smoothing_settings(),
                    &rec_deconvolution.smoothing_settings()
                ));
                assert!(SelectionSettings::compare(
                    &init_deconvolution.selection_settings(),
                    &rec_deconvolution.selection_settings()
                ));
                assert!(FittingSettings::compare(
                    &init_deconvolution.fitting_settings(),
                    &rec_deconvolution.fitting_settings()
                ));
            });
        assert!(AlignmentStrategy::compare(
            &initial.alignment_strategy(),
            &recovered.alignment_strategy()
        ));
        assert!(FilteringSettings::compare(
            &initial.filtering_settings(),
            &recovered.filtering_settings()
        ));
        assert!(SolvingSettings::compare(
            &initial.solving_settings(),
            &recovered.solving_settings()
        ));
    }
}
