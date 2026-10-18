use crate::peak_finding::{Find, Peak};
use crate::smoothing::{CoefficientWindow, MirrorCorrection};
use crate::util::signal_boundaries::SignalBoundaries;
use num_traits::Float;
use std::ops::Range;
use zeenmr_spectrum::SpectrumView1D;
use zeenmr_spectrum::axis::range::RelativeRange;
use zeenmr_spectrum::dimension::DimIndex;
use zeenmr_spectrum::intensity_array::{ArrayView1D, index};

/// Estimate for how many points to expect per peak to avoid reallocations.
const POINTS_PER_PEAK_ESTIMATE: usize = 8;

/// Chunk size in elements.
const CHUNK: usize = 128;

/// Maximum number of points to search for a boundary.
const SEARCH_LIMIT: usize = 512;

/// An error that occurred during the curvature analysis based peak finding.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct CurvatureError {
    /// Kind of error that occurred.
    kind: CurvatureErrorKind,
}

/// Kind of error that can occur during curvature analysis based peak finding.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub enum CurvatureErrorKind {
    /// No peaks were detected in the input data.
    ///
    /// Indicates that intensities may have been read incorrectly or something
    /// went wrong during the smoothing process.
    NoPeaksDetected,
    /// No peaks were found in the part of the spectrum where signals would be
    /// expected.
    ///
    /// This may happen if there are no signals in the signal region, either
    /// due to incorrectly setting its boundaries or if all the signals were
    /// filtered out by the peak finding algorithm.
    EmptySignalRegion,
    /// No peaks were found in the part of the spectrum where random signals due
    /// to noise would be expected.
    ///
    /// Some peak finding algorithms use the estimated noise from regions of a
    /// spectrum where no signals are expected filter out peaks in the signal
    /// region. If there is no noise, i.e., no peaks can be found, this process
    /// might silently fail. This error avoids that issue.
    EmptySignalFreeRegion,
}

impl std::error::Error for CurvatureError {}

impl std::fmt::Display for CurvatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let description = match self.kind {
            CurvatureErrorKind::NoPeaksDetected => "no peaks detected",
            CurvatureErrorKind::EmptySignalRegion => "no peaks found in signal region",
            CurvatureErrorKind::EmptySignalFreeRegion => "no peaks found in signal-free region",
        };

        write!(f, "{description}")
    }
}

impl CurvatureError {
    /// Returns the kind of error that occurred.
    pub fn kind(&self) -> CurvatureErrorKind {
        self.kind
    }

    /// Creates a [`NoPeaksDetected`] error.
    ///
    /// [`NoPeaksDetected`]: CurvatureErrorKind::NoPeaksDetected
    fn no_peaks_detected() -> Self {
        Self {
            kind: CurvatureErrorKind::NoPeaksDetected,
        }
    }

    /// Creates an [`EmptySignalRegion`] error.
    ///
    /// [`EmptySignalRegion`]: CurvatureErrorKind::EmptySignalRegion
    fn empty_signal_region() -> Self {
        Self {
            kind: CurvatureErrorKind::EmptySignalRegion,
        }
    }

    /// Creates an [`EmptySignalFreeRegion`] error.
    ///
    /// [`EmptySignalFreeRegion`]: CurvatureErrorKind::EmptySignalFreeRegion
    fn empty_signal_free_region() -> Self {
        Self {
            kind: CurvatureErrorKind::EmptySignalFreeRegion,
        }
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
pub struct CurvatureAnalysis<T, S> {
    /// Score threshold for peak filtering.
    threshold: Option<T>,
    /// Type determining the signal boundaries of the input spectrum.
    signal_finder: S,
}

impl<T, S> Find<T> for CurvatureAnalysis<T, S>
where
    T: Float,
    S: SignalBoundaries<T>,
{
    type Error = CurvatureError;

    fn find(
        &self,
        spectrum: SpectrumView1D<T, T>,
        smoothed: ArrayView1D<T>,
        ignore: &[Range<usize>],
    ) -> Result<Vec<Peak>, Self::Error> {
        let signal = self.signal_finder.signal_boundaries(spectrum);
        let second_derivative = second_derivative(smoothed);
        let mut peaks = detect_peaks(&second_derivative);
        let bounds = peak_region_boundaries(&peaks, &signal);

        if peaks[..bounds.start].is_empty() && peaks[bounds.end..].is_empty() {
            return Err(CurvatureError::empty_signal_free_region());
        }
        if peaks[bounds.start..bounds.end].is_empty() {
            return Err(CurvatureError::empty_signal_region());
        }

        if let Some(threshold) = self.threshold {
            let sfr_scores = peaks[0..bounds.start]
                .iter()
                .chain(peaks[bounds.end..].iter())
                .map(|peak| score_peak(peak, &second_derivative))
                .collect();
            let (mean, std_dev) = mean_sd_scores(sfr_scores);
            let filter = mean + threshold * std_dev;
            peaks = peaks
                .drain(bounds)
                .filter(|peak| score_peak(peak, &second_derivative) >= filter)
                .collect();
        } else {
            peaks = peaks.drain(bounds).collect();
        }
        peaks.retain(|peak| {
            !ignore
                .iter()
                .any(|range| range.contains(&peak.left) && range.contains(&peak.right))
        });

        match peaks.is_empty() {
            false => Ok(peaks),
            true => Err(CurvatureError::no_peaks_detected()),
        }
    }
}

impl<T> Default for CurvatureAnalysis<T, RelativeRange<T>>
where
    T: Float,
{
    fn default() -> Self {
        Self {
            threshold: Some(T::from(5_u8).expect("conversion from u8 to T must never fail")),
            signal_finder: RelativeRange::new(
                T::from(0.2).expect("conversion from {float} to T must never fail"),
                T::from(0.8).expect("conversion from {float} to T must never fail"),
            )
            .expect("bounds are in [0, 1]"),
        }
    }
}

impl<T, S> CurvatureAnalysis<T, S> {
    /// Creates a new curvature-analysis-based peak finder.
    pub fn new(threshold: Option<T>, signal_finder: S) -> Self {
        Self {
            threshold,
            signal_finder,
        }
    }
}

/// Computes the second derivative of smoothed data.
fn second_derivative<T>(smoothed: ArrayView1D<T>) -> Vec<T>
where
    T: Float,
{
    let lane = smoothed
        .lane_at(DimIndex(0), &index([0]))
        .expect("1D array always has a first dimension");

    match lane.as_slice() {
        Some(lane) => {
            let two = T::one() + T::one();
            let len = lane.len();
            let mut acc = vec![T::zero(); len];
            acc[1..(len - 1)]
                .iter_mut()
                .zip(lane.windows(3))
                .for_each(|(value, w)| {
                    *value = w[0] - two * w[1] + w[2];
                });
            acc[0] = two * lane[1] - two * lane[0];
            acc[len - 1] = two * acc[len - 2] - two * lane[len - 1];

            acc
        }
        None => {
            let coefficients = Box::from([1_f64, -2_f64, 1_f64]);
            let convolution = CoefficientWindow::new(coefficients, MirrorCorrection);
            let mut acc = lane.iter().copied().collect::<Vec<T>>();
            convolution.apply_slice(&mut acc);

            acc
        }
    }
}

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
/// points. This is because the second derivative can only be computed for inner
/// points of the signal intensities, and the conditions for a peak center can
/// only be checked for a second derivative data point with neighbors on both
/// sides. For example, with 10 signal intensity data points, the second
/// derivative is computed for the 8 inner points, and the peak centers can be
/// found at indices 2 to 7. The peak centers are then scanned with a sliding
/// window.
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
fn detect_peaks<T>(second_derivative: &[T]) -> Vec<Peak>
where
    T: Float,
{
    let len = second_derivative.len();
    let mut peaks = Vec::with_capacity(len / POINTS_PER_PEAK_ESTIMATE);
    let mut scratch = [0_usize; CHUNK];
    let mut start = 1;
    while start + 1 < len {
        let end = (start + CHUNK).min(len - 1);
        let mut passed = 0;
        for center in start..end {
            scratch[passed] = center;
            let center_test = (second_derivative[center] < T::zero())
                && (second_derivative[center] < second_derivative[center - 1])
                && (second_derivative[center] <= second_derivative[center + 1]);
            passed += center_test as usize;
        }
        peaks.extend(scratch[..passed].iter().filter_map(|&center| {
            let left = center
                - second_derivative[center.saturating_sub(SEARCH_LIMIT)..(center + 1)]
                    .windows(3)
                    .rev()
                    .position(|w| {
                        w[1] > w[2] && (w[1] >= w[0] || (w[1] < T::zero() && w[0] >= T::zero()))
                    })?;
            let right = center
                + second_derivative[center..center.saturating_add(SEARCH_LIMIT).min(len)]
                    .windows(3)
                    .position(|w| {
                        w[1] > w[0] && (w[1] >= w[2] || (w[1] < T::zero() && w[2] >= T::zero()))
                    })?;

            Some(Peak {
                left,
                center,
                right,
            })
        }));
        start = end;
    }

    peaks
}

/// Determines the range of peaks that fall within the specified signal
/// boundaries.
fn peak_region_boundaries(peaks: &[Peak], signal: &Range<usize>) -> Range<usize> {
    let left = peaks
        .iter()
        .position(|peak| peak.center > signal.start)
        .unwrap_or(0);
    let right = peaks[left..]
        .iter()
        .position(|peak| peak.center > signal.end)
        .map_or(peaks.len(), |offset| left + offset);

    left..right
}

/// Scores peaks based on the sum of absolute second derivative values on
/// either side of the peak center.
///
/// Peaks are scored based on how sharp they are, i.e., how quickly the signal's
/// curvature changes around the peak. A higher score indicates a sharper peak.
/// The score is computed as the minimum of the sums of the absolute second
/// derivative values within bounds on both sides of the peak center.
fn score_peak<T>(peak: &Peak, second_derivative: &[T]) -> T
where
    T: Float,
{
    let left_sum = second_derivative[peak.left..(peak.center + 1)]
        .iter()
        .fold(T::zero(), |acc, &x| acc + x.abs());
    let right_sum = second_derivative[peak.center..(peak.right + 1)]
        .iter()
        .fold(T::zero(), |acc, &x| acc + x.abs());

    left_sum.min(right_sum)
}

/// Computes the mean and standard deviation of a collection of scores.
fn mean_sd_scores<T>(scores: Vec<T>) -> (T, T)
where
    T: Float,
{
    let len = T::from(scores.len()).expect("conversion from usize to T must never fail");
    let mean = scores.iter().fold(T::zero(), |acc, &x| acc + x) / len;
    let variance = scores
        .iter()
        .map(|score| (*score - mean).powi(2))
        .fold(T::zero(), |acc, x| acc + x)
        / len;

    (mean, variance.sqrt())
}
