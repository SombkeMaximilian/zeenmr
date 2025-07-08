use crate::deconvolution::error::{Error, Kind};
use crate::deconvolution::peak_selection::{Peak, ScoringMethod};
use crate::{Result, Settings};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Trait interface for peak selection algorithms.
pub(crate) trait Selector: Send + Sync + std::fmt::Debug {
    /// Detects peaks in a spectrum and returns the ones that pass a filter.
    fn select_peaks(
        &self,
        intensities: &[f64],
        signal_boundaries: (usize, usize),
        ignore_regions: Option<&[(usize, usize)]>,
    ) -> Result<Vec<Peak>>;

    /// Returns the settings of the trait object.
    fn settings(&self) -> SelectionSettings;
}

/// Peak selection settings for configuring the [`Deconvoluter`].
///
/// [`Deconvoluter`]: crate::deconvolution::Deconvoluter
#[non_exhaustive]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(tag = "method", rename_all_fields = "camelCase")
)]
pub enum SelectionSettings {
    /// Only use the peak detector without any filtering.
    ///
    /// Finds peaks in the spectrum by analyzing the curvature of the signal
    /// through the second derivative. This option disables any filtering of the
    /// detected peaks. May be useful when no noise is expected.
    DetectorOnly,
    /// Filter based on the score of peaks found in the signal free region.
    ///
    /// Finds peaks in the spectrum by analyzing the curvature of the signal
    /// through the second derivative and scores them based on the selected
    /// scoring method. Mean and standard deviation are calculated for the
    /// scores of peaks in the signal free region (where only noise is present).
    /// Finally, peaks in the signal region are filtered according to the
    /// following criterion:
    ///
    /// ```text
    /// score > mean + threshold * std_dev
    /// ```
    NoiseScoreFilter {
        /// Scoring method to use.
        scoring_method: ScoringMethod,
        /// Score threshold to use for filtering peaks.
        threshold: f64,
    },
}

impl Default for SelectionSettings {
    fn default() -> Self {
        SelectionSettings::NoiseScoreFilter {
            scoring_method: ScoringMethod::default(),
            threshold: 5.0,
        }
    }
}

impl std::fmt::Display for SelectionSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectionSettings::DetectorOnly => write!(f, "Detector Only"),
            SelectionSettings::NoiseScoreFilter {
                scoring_method,
                threshold,
            } => write!(
                f,
                "Noise Score Filter [scoring method: {}, score threshold: {}]",
                scoring_method, threshold
            ),
        }
    }
}

impl Settings for SelectionSettings {
    fn validate(&self) -> Result<()> {
        match self {
            SelectionSettings::DetectorOnly => {}
            SelectionSettings::NoiseScoreFilter { threshold, .. } => {
                if *threshold <= 0.0 || !threshold.is_finite() {
                    return Err(
                        Error::new(Kind::InvalidSelectionSettings { settings: *self }).into(),
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(test)]
    fn compare(&self, other: &Self) -> bool {
        match (self, other) {
            (SelectionSettings::DetectorOnly, SelectionSettings::DetectorOnly) => true,
            (
                SelectionSettings::NoiseScoreFilter {
                    scoring_method: scoring_method1,
                    threshold: threshold1,
                },
                SelectionSettings::NoiseScoreFilter {
                    scoring_method: scoring_method2,
                    threshold: threshold2,
                },
            ) => {
                ScoringMethod::compare(scoring_method1, scoring_method2)
                    && float_cmp::approx_eq!(f64, *threshold1, *threshold2)
            }
            _ => false,
        }
    }
}
