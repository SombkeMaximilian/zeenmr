use crate::Result;
use crate::deconvolution::error::{Error, Kind};
use crate::deconvolution::peak_selection::{
    Detector, Peak, Scorer, ScorerMinimumSum, ScoringMethod, SelectionSettings, Selector,
    peak_region_boundaries, second_derivative,
};

/// Detects peaks in a spectrum and returns the ones that pass a filter.
///
/// Peaks are detected using the curvature of the signal through the second
/// derivative. The scores of the peaks are computed using the selected
/// scoring algorithm. The mean and standard deviation of the scores in the
/// signal free region (where only noise is present) are calculated, and
/// peaks in the signal region are filtered according to the following
/// criterion:
///
/// ```text
/// score > mean + threshold * std_dev
/// ```
///
/// Optionally, regions to be ignored can be provided. Peaks within these
/// regions are also filtered out.
#[derive(Debug)]
pub(crate) struct NoiseScoreFilter {
    /// The scoring method to use.
    scoring_method: ScoringMethod,
    /// The threshold for filtering peaks.
    threshold: f64,
}

impl Selector for NoiseScoreFilter {
    fn select_peaks(
        &self,
        intensities: &[f64],
        signal_boundaries: (usize, usize),
        ignore_regions: Option<&[(usize, usize)]>,
    ) -> Result<Vec<Peak>> {
        let mut second_derivative = second_derivative(intensities);
        let detector = Detector::new(&second_derivative);
        let mut peaks = detector.detect_peaks()?;
        if let Some(ignore_regions) = ignore_regions {
            peaks.retain(|peak| {
                !ignore_regions.iter().any(|(start, end)| {
                    (peak.left() >= *start && peak.left() < *end)
                        || (peak.right() >= *start && peak.right() < *end)
                })
            });
        }
        second_derivative
            .iter_mut()
            .for_each(|d| *d = d.abs());

        self.filter_peaks(peaks, &second_derivative, signal_boundaries)
    }

    fn settings(&self) -> SelectionSettings {
        SelectionSettings::NoiseScoreFilter {
            scoring_method: self.scoring_method,
            threshold: self.threshold,
        }
    }
}

impl NoiseScoreFilter {
    /// Creates a new `NoiseScoreFilter` with the given scoring algorithm and
    /// threshold.
    pub(crate) fn new(scoring_method: ScoringMethod, threshold: f64) -> Self {
        Self {
            scoring_method,
            threshold,
        }
    }

    /// Filters peaks based on their scores.
    ///
    /// The scores are computed using the selected scoring algorithm, and then
    /// divided into signal free region (SFR) and signal region. The mean and
    /// standard deviation of the scores in the SFR, where only noise is
    /// present, are calculated, and peaks in the signal region are filtered
    /// according to the following criterion:
    ///
    /// ```text
    /// score > mean + threshold * std_dev
    /// ```
    ///
    /// # Errors
    ///
    /// The following errors are possible:
    /// - [`EmptySignalRegion`](Kind::EmptySignalRegion)
    /// - [`EmptySignalFreeRegion`](Kind::EmptySignalFreeRegion)
    fn filter_peaks(
        &self,
        mut peaks: Vec<Peak>,
        abs_second_derivative: &[f64],
        signal_boundaries: (usize, usize),
    ) -> Result<Vec<Peak>> {
        let scorer = match self.scoring_method {
            ScoringMethod::MinimumSum => ScorerMinimumSum::new(abs_second_derivative),
        };
        let boundaries = peak_region_boundaries(&peaks, signal_boundaries);

        if peaks[..boundaries.0].is_empty() && peaks[boundaries.1..].is_empty() {
            return Err(Error::new(Kind::EmptySignalFreeRegion).into());
        }
        if peaks[boundaries.0..boundaries.1].is_empty() {
            return Err(Error::new(Kind::EmptySignalRegion).into());
        }

        let scores_sfr = peaks[0..boundaries.0]
            .iter()
            .chain(peaks[boundaries.1..].iter())
            .map(|peak| scorer.score_peak(peak))
            .collect();
        let (mean, sd) = Self::mean_sd_scores(scores_sfr);

        peaks = peaks
            .drain(boundaries.0..boundaries.1)
            .filter(|peak| scorer.score_peak(peak) >= mean + self.threshold * sd)
            .collect();

        if peaks.is_empty() {
            return Err(Error::new(Kind::EmptySignalRegion).into());
        }

        Ok(peaks)
    }

    /// Computes the mean and standard deviation of a vector of scores.
    fn mean_sd_scores(scores: Vec<f64>) -> (f64, f64) {
        let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance: f64 = scores
            .iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f64>()
            / scores.len() as f64;

        (mean, variance.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_send, assert_sync};
    use float_cmp::assert_approx_eq;

    #[test]
    fn thread_safety() {
        assert_send!(NoiseScoreFilter);
        assert_sync!(NoiseScoreFilter);
    }

    #[test]
    fn mean_sd_scores() {
        let peaks: Vec<Peak> = vec![2, 4, 5, 8]
            .into_iter()
            .map(|i| Peak::new(i - 1, i, i + 1))
            .collect();
        let abs_second_derivative = vec![1.0, 2.0, 4.0, 2.0, 2.0, 5.0, 4.0, 3.0, 2.0];
        let scorer = ScorerMinimumSum::new(&abs_second_derivative);
        let peak_region_boundaries = (1, 3);
        let scores_sfr: Vec<f64> = peaks[0..peak_region_boundaries.0]
            .iter()
            .chain(peaks[peak_region_boundaries.1..].iter())
            .map(|peak| scorer.score_peak(peak))
            .collect();
        let (mean, sd) = NoiseScoreFilter::mean_sd_scores(scores_sfr);
        assert_approx_eq!(f64, mean, 4.0);
        assert_approx_eq!(f64, sd, 1.0);
    }
}
