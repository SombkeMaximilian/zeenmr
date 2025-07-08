use crate::deconvolution::error::{Error, Kind};
use crate::deconvolution::peak_selection::Peak;
use crate::error::Result;

/// Peak detection algorithm based on the second derivative of the signal.
///
/// This algorithm detects peaks in a signal by analyzing its second derivative,
/// which captures changes in curvature and transitions in signal behavior, such
/// as local extrema and changes in concavity
///
/// In the tables below, `x` marks the position of the peak center or border
/// under consideration, and `-` represents adjacent data points included in
/// the sliding window.
///
/// In the future, this may be extended into a trait to allow for customizable
/// peak detection strategies.
///
/// # Peak Centers
///
/// A peak center is defined as a point where the second derivative is negative
/// and where the second derivative is at a local minimum. This means that the
/// signal is concave down (curving upwards), and the curvature is decreasing in
/// both directions.
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
/// # Peak Borders
///
/// ## Right
///
/// A right border is a point where the second derivative is increasing relative
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
/// A left border is defined analogously to a right border. Likewise, the index
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
pub(crate) struct Detector<'a> {
    /// The second derivative of the signal.
    second_derivative: &'a [f64],
}

impl<'a> Detector<'a> {
    /// Creates a new `Detector` with the given second derivative.
    pub(crate) fn new(second_derivative: &'a [f64]) -> Self {
        Detector { second_derivative }
    }

    /// Detects the peaks in the signal.
    ///
    /// # Errors
    ///
    /// The following errors are possible:
    /// - [`NoPeaksDetected`](Kind::NoPeaksDetected)
    pub(crate) fn detect_peaks(&self) -> Result<Vec<Peak>> {
        let peak_centers = self.find_peak_centers();
        let peak_borders = self.find_peak_borders(&peak_centers);
        let peaks = peak_centers
            .into_iter()
            .zip(peak_borders)
            .filter(|(_, (left, right))| *left != 0 && *right != self.second_derivative.len() + 1)
            .map(|(center, (left, right))| Peak::new(left, center, right))
            .collect::<Vec<Peak>>();
        if peaks.is_empty() {
            return Err(Error::new(Kind::NoPeaksDetected).into());
        }

        Ok(peaks)
    }

    /// Internal helper function to finds the indices of the peak centers.
    /// Indices are offset by +2 to account for the fact that the second
    /// derivative can only be computed for the inner points of the signal
    /// intensities, and the sliding window approach requires neighbors on
    /// both sides.
    fn find_peak_centers(&self) -> Vec<usize> {
        self.second_derivative
            .windows(3)
            .enumerate()
            .filter(|(_, w)| w[1] < 0. && w[1] < w[0] && w[1] < w[2])
            .map(|(i, _)| i + 2)
            .collect()
    }

    /// Internal helper function to find the borders of the peaks. The indices
    /// of the borders are relative to the peak center, so they get added to the
    /// peak center index to get the global index (left border index has to be
    /// interpreted as negative index).
    fn find_peak_borders(&self, peak_centers: &[usize]) -> Vec<(usize, usize)> {
        peak_centers
            .iter()
            .map(|&i| {
                (
                    i - Self::find_left_border(&self.second_derivative[..i]),
                    i + Self::find_right_border(&self.second_derivative[i - 1..]),
                )
            })
            .collect()
    }

    /// Internal helper function to finds the right border of a peak. Returns
    /// the length of the slice to the right of the peak center if no right
    /// border can be found.
    fn find_right_border(second_derivative_right: &[f64]) -> usize {
        second_derivative_right
            .windows(3)
            .position(|w| w[1] > w[0] && (w[1] >= w[2] || (w[1] < 0. && w[2] >= 0.)))
            .map_or(second_derivative_right.len(), |i| i + 1)
    }

    /// Internal helper function to finds the left border of a peak. Returns the
    /// length of the slice to the left of the peak center if no left border can
    /// be found. Note that the indices are reversed.
    fn find_left_border(second_derivative_left: &[f64]) -> usize {
        second_derivative_left
            .windows(3)
            .rev()
            .position(|w| w[1] > w[2] && (w[1] >= w[0] || (w[1] < 0. && w[0] >= 0.)))
            .map_or(second_derivative_left.len(), |i| i + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_peak_centers() {
        let second_derivative = vec![0.0, -2.0, 0.0];
        let detector = Detector::new(&second_derivative);
        let peak_centers = detector.find_peak_centers();
        assert_eq!(peak_centers, vec![2]);
    }

    #[test]
    fn find_peak_borders() {
        // indices are offset by 1, as second derivative is computed for central points
        // only
        let mut second_derivative = vec![0.5, -0.5, -1.0, 0.0, 0.5, 0.0];
        {
            let detector = Detector::new(&second_derivative);
            assert_eq!(detector.find_peak_borders(&[3]), vec![(2, 5)]);
        }
        second_derivative = vec![0.0, 0.5, 0.0, -1.0, -0.5, 0.5];
        {
            let detector = Detector::new(&second_derivative);
            assert_eq!(detector.find_peak_borders(&[4]), vec![(2, 5)]);
        }
        second_derivative = vec![1.0, 1.0, 1.0, 1.5, 1.0];
        {
            let detector = Detector::new(&second_derivative);
            assert_eq!(detector.find_peak_borders(&[3]), vec![(0, 4)]);
        }
        second_derivative = vec![1.0, 1.5, 1.0, 1.0, 1.0];
        {
            let detector = Detector::new(&second_derivative);
            assert_eq!(detector.find_peak_borders(&[3]), vec![(2, 6)]);
        }
        second_derivative = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        {
            let detector = Detector::new(&second_derivative);
            assert_eq!(detector.find_peak_borders(&[3]), vec![(0, 6)]);
        }
    }

    #[test]
    fn find_right_border() {
        let mut second_derivative = vec![0.0, -2.0, -1.0, -0.5, 0.5];
        assert_eq!(Detector::find_right_border(&second_derivative[2..]), 1);
        second_derivative = vec![0.0, -2.0, -1.0, 0.0, 0.5, 0.0];
        assert_eq!(Detector::find_right_border(&second_derivative[2..]), 2);
        second_derivative = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(Detector::find_right_border(&second_derivative[2..]), 3);
    }

    #[test]
    fn find_left_border() {
        let mut second_derivative = vec![0.5, -0.5, -1.0, -2.0, 0.0];
        assert_eq!(Detector::find_left_border(&second_derivative[0..=2]), 1);
        second_derivative = vec![0.0, 0.5, 0.0, -1.0, -2.0, 0.0];
        assert_eq!(Detector::find_left_border(&second_derivative[0..=3]), 2);
        second_derivative = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(Detector::find_left_border(&second_derivative[0..=2]), 3);
    }
}
