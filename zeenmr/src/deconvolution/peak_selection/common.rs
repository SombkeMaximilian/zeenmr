use crate::deconvolution::peak_selection::Peak;

/// Computes the second derivative of a signal with 3-point finite
/// differences.
pub(crate) fn second_derivative(intensities: &[f64]) -> Vec<f64> {
    intensities
        .windows(3)
        .map(|w| w[0] - 2.0 * w[1] + w[2])
        .collect()
}

/// Computes the indices in the slice of `Peak`s that delimit the signal
/// region.
///
/// Peaks are ordered by their center due to how the peak detection
/// algorithm works. Therefore, there is a first and a last peak that belong
/// to the signal region. Note that the lower boundary is included in and
/// upper boundary is excluded from the signal region.
///
/// |  Index              | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
/// | ------------------- | - | - | - | - | - | - | - | - | - | - |
/// | Signal Boundaries   |   |   | x |   |   |   |   | x |   |   |
/// | Peak Centers        |   | 0 |   | 1 |   | 2 |   |   | 3 |   |
/// | Signal Region Peaks |   |   |   | x |   | x |   |   |   |   |
/// | Noise Peaks         |   | x |   |   |   |   |   |   | x |   |
pub(crate) fn peak_region_boundaries(
    peaks: &[Peak],
    signal_boundaries: (usize, usize),
) -> (usize, usize) {
    let left = peaks
        .iter()
        .position(|peak| peak.center() > signal_boundaries.0)
        .map_or(0, |i| i);
    let right = peaks[left..]
        .iter()
        .position(|peak| peak.center() > signal_boundaries.1)
        .map_or(peaks.len() - 1, |i| left + i);

    (left, right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deconvolution::peak_selection::common::peak_region_boundaries as prb;
    use crate::deconvolution::peak_selection::common::second_derivative as sd;
    use float_cmp::assert_approx_eq;

    #[test]
    fn second_derivative() {
        let intensities = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let expected_second_derivative = [0.0, -2.0, 0.0];
        let computed_second_derivative = sd(&intensities);
        computed_second_derivative
            .iter()
            .zip(expected_second_derivative.iter())
            .for_each(|(&sdc, &sde)| assert_approx_eq!(f64, sdc, sde));
    }

    #[test]
    fn peak_region_boundaries() {
        let signal_region_boundaries = (3, 7);
        let peaks = vec![2, 4, 5, 8]
            .into_iter()
            .map(|i| Peak::new(i - 1, i, i + 1))
            .collect::<Vec<Peak>>();
        assert_eq!(prb(&peaks, signal_region_boundaries), (1, 3));
    }
}
