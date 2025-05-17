use crate::deconvolution::peak_selection::Peak;
use crate::{Result, Settings};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Trait interface for peak scoring methods.
pub(crate) trait Scorer {
    /// Scores the given peak.
    fn score_peak(&self, peak: &Peak) -> f64;
}

/// Scoring methods for the peaks.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(tag = "method", rename_all_fields = "camelCase")
)]
pub enum ScoringMethod {
    /// Minimum Sum of the absolute second derivative.
    ///
    /// The minimum sum scorer computes the sum of the absolute second
    /// derivative values of the peak at the left and right sides of the
    /// peak center. The smaller of the two sums is the score of the peak.
    #[default]
    MinimumSum,
}

impl std::fmt::Display for ScoringMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScoringMethod::MinimumSum => write!(f, "Minimum Sum"),
        }
    }
}

impl Settings for ScoringMethod {
    fn validate(&self) -> Result<()> {
        match self {
            ScoringMethod::MinimumSum => Ok(()),
        }
    }

    #[cfg(test)]
    fn compare(&self, other: &Self) -> bool {
        match (self, other) {
            (ScoringMethod::MinimumSum, ScoringMethod::MinimumSum) => true,
        }
    }
}

/// Peak scoring method based on the sum of second derivative values.
#[derive(Debug)]
pub(crate) struct ScorerMinimumSum<'a> {
    /// The absolute second derivative of the signal as a reference.
    abs_second_derivative: &'a [f64],
}

impl Scorer for ScorerMinimumSum<'_> {
    /// Scores the given peak by computing the sum of the absolute second
    /// derivative values at the left and right sides of the peak center.
    /// Returns the smaller of the two sums as the score of the peak.
    fn score_peak(&self, peak: &Peak) -> f64 {
        f64::min(
            self.abs_second_derivative[peak.left() - 1..peak.center()]
                .iter()
                .sum(),
            self.abs_second_derivative[peak.center() - 1..peak.right()]
                .iter()
                .sum(),
        )
    }
}

impl<'a> ScorerMinimumSum<'a> {
    /// Creates a new `ScorerMinimumSum` with the given absolute second
    /// derivative.
    pub(crate) fn new(abs_second_derivative: &'a [f64]) -> Self {
        ScorerMinimumSum {
            abs_second_derivative,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deconvolution::peak_selection::Peak;
    use float_cmp::assert_approx_eq;

    #[test]
    fn minimum_sum() {
        let peaks = [Peak::new(1, 3, 4), Peak::new(5, 6, 9)];
        let abs_second_derivative = vec![1.0, 2.0, 4.0, 2.0, 2.0, 5.0, 4.0, 3.0, 2.0];
        let scorer = ScorerMinimumSum::new(&abs_second_derivative);
        let expected_scores = [6.0, 7.0];
        let computed_scores = peaks
            .iter()
            .map(|peak| scorer.score_peak(peak))
            .collect::<Vec<f64>>();
        computed_scores
            .iter()
            .zip(expected_scores.iter())
            .for_each(|(&cs, &es)| assert_approx_eq!(f64, cs, es));
    }
}
