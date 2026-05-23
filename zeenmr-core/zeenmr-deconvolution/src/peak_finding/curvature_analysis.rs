use crate::error::{Error, Result};
use crate::peak_finding::{FindPeaks, Peak};
use std::ops::RangeBounds;
use zeenmr_spectrum::IndexRange;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Peak detection algorithm based on the second derivative of the signal.
///
/// In the tables below, `x` marks the position of the peak center or border
/// under consideration, and `-` represents adjacent data points included in
/// the sliding window.
///
/// # Centers
///
/// A peak center is defined as a point where the second derivative is negative
/// and where the second derivative is at a local minimum. This means that the
/// signal is concave down (curving downwards).
///
/// The indices are offset by +2 to align with the corresponding signal data
/// points, as the second derivative can only be computed for inner points of
/// the signal intensities, and the conditions for a peak center can only be
/// checked for a second derivative data point with neighbors on both sides. For
/// example, with 10 signal intensity data points, the second derivative is
/// computed for the 8 inner points, and the peak centers can be found at
/// indices 2 to 7. The peak centers are then scanned with a sliding window.
///
/// | Signal Intensities Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
/// | ------------------------ | - | - | - | - | - | - | - | - | - | - |
/// | Second Derivative Index  |   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |   |
/// | Potential Centers Index  |   |   | 0 | 1 | 2 | 3 | 4 | 5 |   |   |
/// | Step 1                   |   | - | x | - |   |   |   |   |   |   |
/// | Step 2                   |   |   | - | x | - |   |   |   |   |   |
/// | Step 3                   |   |   |   | - | x | - |   |   |   |   |
/// | Step 4                   |   |   |   |   | - | x | - |   |   |   |
/// | Step 5                   |   |   |   |   |   | - | x | - |   |   |
/// | Step 6                   |   |   |   |   |   |   | - | x | - |   |
///
/// # Bounds
///
/// ## Right
///
/// A right bound is a point where the second derivative is increasing relative
/// to the previous point and either remains greater than or equal to the next
/// point, or transitions from negative to non-negative curvature at the
/// subsequent point. This behavior in the second derivative reflects the
/// transition of the signal's curvature from concave down to flat or concave
/// up, marking the boundary of the peak region. The index is computed relative
/// to the peak center, i.e. to get the global index, the peak center index must
/// be added. In other words, the peak center is at index 0 for this reference
/// frame.
///
/// | Index relative to center | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
/// | ------------------------ | - | - | - | - | - | - | - |
/// | Step 1                   | - | x | - |   |   |   |   |
/// | Step 2                   |   | - | x | - |   |   |   |
/// | Step 3                   |   |   | - | x | - |   |   |
/// | Step 4                   |   |   |   | - | x | - |   |
/// | Step 5                   |   |   |   |   | - | x | - |
///
/// ## Left
///
/// A left bound is defined analogously to a right bound. Likewise, the index
/// is computed relative to the peak center. However, the indices have to be
/// interpreted as negative indices. Using negative indices provides a
/// consistent way to analyze points relative to the peak center. The peak
/// center is at index 0 for this reference frame again.
///
/// | Index relative to center | -6 | -5 | -4 | -3 | -2 | -1 | 0 |
/// | ------------------------ | -- | -- | -- | -- | -- | -- | - |
/// | Step 1                   |    |    |    |    | -  | x  | - |
/// | Step 2                   |    |    |    | -  | x  | -  |   |
/// | Step 3                   |    |    | -  | x  | -  |    |   |
/// | Step 4                   |    | -  | x  | -  |    |    |   |
/// | Step 5                   | -  | x  | -  |    |    |    |   |
#[derive(Debug)]
struct CurvatureDetector<'a>(&'a [f64]);

impl CurvatureDetector<'_> {
    /// Detects peaks in the signal based on the second derivative.
    ///
    /// # Errors
    ///
    /// The following errors can occur:
    /// - [`NoPeaksDetected`](crate::error::Kind::NoPeaksDetected)
    fn detect_peaks(&self) -> Result<Vec<Peak>> {
        let peaks = self
            .0
            .windows(3)
            .enumerate()
            .filter_map(|(i, w)| {
                if w[1] < 0.0 && w[1] < w[0] && w[1] < w[2] {
                    let center = i + 2;
                    let left = center - self.find_left_offset(&self.0[..center]);
                    let right = center + self.find_right_offset(&self.0[center - 1..]);

                    if left != 0 && right != self.0.len() + 1 {
                        Some(Peak {
                            left,
                            center,
                            right,
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<Peak>>();

        match peaks.is_empty() {
            false => Ok(peaks),
            true => Err(Error::no_peaks_detected()),
        }
    }

    /// Finds the left bound's offset from the peak center.
    ///
    /// See also: [Left Bound](CurvatureDetector#left)
    fn find_left_offset(&self, second_derivative_left: &[f64]) -> usize {
        second_derivative_left
            .windows(3)
            .rev()
            .position(|w| w[1] > w[2] && (w[1] >= w[0] || (w[1] < 0.0 && w[0] >= 0.0)))
            .map_or(second_derivative_left.len(), |left| left + 1)
    }

    /// Finds the right bound's offset from the peak center.
    ///
    /// See also: [Right Bound](CurvatureDetector#right)
    fn find_right_offset(&self, second_derivative_right: &[f64]) -> usize {
        second_derivative_right
            .windows(3)
            .position(|w| w[1] > w[0] && (w[1] >= w[2] || (w[1] < 0.0 && w[2] >= 0.0)))
            .map_or(second_derivative_right.len(), |right| right + 1)
    }
}

/// Scores peaks based on the sum of absolute second derivative values on
/// either side of the peak center.
///
/// Peaks are scored based on how sharp they are, i.e., how quickly the signal's
/// curvature changes around the peak. A higher score indicates a sharper peak.
/// The score is computed as the minimum of the sums of the absolute second
/// derivative values within bounds on both sides of the peak center.
#[derive(Debug)]
struct CurvatureScore<'a>(&'a [f64]);

impl CurvatureScore<'_> {
    /// Scores the given peak.
    fn score_peak(&self, peak: &Peak) -> f64 {
        let left_sum = self.0[peak.left - 1..peak.center]
            .iter()
            .sum::<f64>();
        let right_sum = self.0[peak.center - 1..peak.right]
            .iter()
            .sum::<f64>();

        left_sum.min(right_sum)
    }
}

/// Detects peaks in a spectrum using the signal curvature and returns ones with
/// a score that passes a filter based on the threshold.
///
/// # Detection
///
/// Peak centers are searched for by looking for points where the second
/// derivative is negative and at a local minimum. The left and right bounds of
/// the peak are determined by finding points where the curvature changes from
/// concave down to flat or concave up, i.e., where the second derivative
/// starts increasing and either remains greater than or equal to the next
/// point, or transitions from negative to non-negative curvature at the
/// subsequent point.
///
/// # Filtering
///
/// Peaks are scored based on how sharp they are, i.e., how quickly the signal's
/// curvature changes around the peak. A higher score indicates a sharper peak.
/// The score is computed as the minimum of the sums of the absolute second
/// derivative values within bounds on both sides of the peak center.
#[derive(Copy, Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CurvatureAnalysis {
    /// Score threshold for peak filtering.
    pub threshold: Option<f64>,
}

impl FindPeaks for CurvatureAnalysis {
    fn find_peaks(
        &self,
        smoothed: Vec<f64>,
        signal: IndexRange,
        ignore: Option<Vec<IndexRange>>,
    ) -> Result<Vec<Peak>> {
        let mut second_derivative = smoothed
            .windows(3)
            .map(|w| w[0] - 2.0 * w[1] + w[2])
            .collect::<Vec<f64>>();
        let mut peaks = CurvatureDetector(&second_derivative).detect_peaks()?;
        if let Some(ignore) = ignore {
            peaks.retain(|peak| {
                !ignore
                    .iter()
                    .any(|range| range.contains(&peak.left) && range.contains(&peak.right))
            });
        }
        for value in second_derivative.iter_mut() {
            *value = value.abs();
        }
        let scorer = CurvatureScore(&second_derivative);
        let bounds = Self::peak_region_boundaries(&peaks, signal);

        if peaks[..bounds.start].is_empty() && peaks[bounds.end..].is_empty() {
            return Err(Error::empty_signal_free_region());
        }
        if peaks[bounds.start..bounds.end].is_empty() {
            return Err(Error::empty_signal_region());
        }

        let scores_sfr = peaks[0..bounds.start]
            .iter()
            .chain(peaks[bounds.end..].iter())
            .map(|peak| scorer.score_peak(peak))
            .collect::<Vec<f64>>();
        let (mean, std_dev) = Self::mean_sd_scores(scores_sfr);
        if let Some(threshold) = self.threshold {
            peaks = peaks
                .drain(bounds.start..bounds.end)
                .filter(|peak| scorer.score_peak(peak) >= mean + threshold * std_dev)
                .collect();
        } else {
            peaks = peaks.drain(bounds.start..bounds.end).collect();
        }

        match peaks.is_empty() {
            false => Ok(peaks),
            true => Err(Error::no_peaks_detected()),
        }
    }
}

impl Default for CurvatureAnalysis {
    fn default() -> Self {
        Self {
            threshold: Some(5.0),
        }
    }
}

impl CurvatureAnalysis {
    /// Creates a new `CurvatureAnalysis` with the given score threshold.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_deconvolution::peak_finding::CurvatureAnalysis;
    ///
    /// let finder_with_filter = CurvatureAnalysis::new(Some(5.0));
    /// let finder_without_filter = CurvatureAnalysis::new(None);
    /// ```
    pub fn new(threshold: Option<f64>) -> Self {
        Self { threshold }
    }

    /// Determines the range of peaks that fall within the specified signal
    /// boundaries.
    fn peak_region_boundaries(peaks: &[Peak], signal: IndexRange) -> IndexRange {
        let left = peaks
            .iter()
            .position(|peak| peak.center > signal.start)
            .unwrap_or(0);
        let right = peaks[left..]
            .iter()
            .position(|peak| peak.center > signal.end)
            .map_or(peaks.len() - 1, |offset| left + offset);

        (left, right).into()
    }

    /// Computes the mean and standard deviation of a vector of scores.
    fn mean_sd_scores(scores: Vec<f64>) -> (f64, f64) {
        let mean = scores.iter().sum::<f64>() / (scores.len() as f64);
        let variance = scores
            .iter()
            .map(|score| (*score - mean).powi(2))
            .sum::<f64>()
            / (scores.len() as f64);

        (mean, variance.sqrt())
    }
}
