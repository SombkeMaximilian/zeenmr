use crate::smoothing::{CoefficientWindow, MirrorCorrection, Smooth};
use num_traits::Float;
use zeenmr_spectrum::intensity_array::ArrayViewMut1D;

const PRECISION: f64 = 100_f64 * f64::EPSILON;

/// Least squares filter.
///
/// Smooths a sequence of values by fitting polynomials to a sliding window and
/// replacing its central value.
///
/// Also known as Savitzky-Golay, Polynomial or DISPO filter.
///
/// Reference: Numerical Recipes The Art of Scientific Computing (3rd Edition)
/// p. 766 f.
#[derive(Clone, PartialEq, Debug)]
pub struct LeastSquares {
    /// Number of iterations to apply the filter.
    iterations: usize,
    /// Convoluted coefficients.
    convolution: CoefficientWindow<MirrorCorrection>,
}

impl<T> Smooth<T> for LeastSquares
where
    T: Float,
{
    type Error = std::convert::Infallible;

    fn smooth_in_place(&self, mut data: ArrayViewMut1D<T>) -> Result<(), Self::Error> {
        for _ in 0..self.iterations {
            self.convolution
                .smooth_in_place(data.view_mut())
                .expect("coefficient convolution is infallible");
        }

        Ok(())
    }
}

impl Default for LeastSquares {
    fn default() -> Self {
        let iterations = 2;
        let window = 5;
        let order = 2;
        let coefficients = Self::coefficients(window, order, 0);

        if let Some(coefficients) =
            Self::convolve(&coefficients, iterations, 100_f64 * f64::EPSILON)
        {
            Self {
                iterations: 1,
                convolution: CoefficientWindow::new(coefficients, MirrorCorrection),
            }
        } else {
            Self {
                iterations,
                convolution: CoefficientWindow::new(coefficients, MirrorCorrection),
            }
        }
    }
}

impl LeastSquares {
    /// Creates a new least squares filter.
    ///
    /// Returns `None` if
    /// - `iterations` is `0`, or
    /// - `window < 3`, or
    /// - `window = 0 mod 2`, or
    /// - `window <= order`
    pub fn new(iterations: usize, window: usize, order: u32) -> Option<Self> {
        if iterations == 0 || window < 3 || window % 2 != 1 || window <= order as usize {
            return None;
        }

        let coefficients = Self::coefficients(window, order, 0);

        if let Some(coefficients) = Self::convolve(&coefficients, iterations, PRECISION) {
            Some(Self {
                iterations: 1,
                convolution: CoefficientWindow::new(coefficients, MirrorCorrection),
            })
        } else {
            Some(Self {
                iterations,
                convolution: CoefficientWindow::new(coefficients, MirrorCorrection),
            })
        }
    }

    /// Computes the SG coefficients using L D L^T decomposition.
    fn coefficients(window: usize, order: u32, unit: usize) -> Box<[f64]> {
        let half_window = window as i32 / 2;
        let window_range = -half_window..half_window + 1;
        let n = order as usize + 1;
        let order_range = 0..n as i32;

        // matrix A^T A
        let mut m = order_range
            .clone()
            .flat_map(|i| order_range.clone().map(move |j| (i, j)))
            .map(|(i, j)| {
                window_range
                    .clone()
                    .map(move |k| ((k as f64) / (half_window as f64)).powi(i + j))
                    .fold(0_f64, |acc, x| acc + x)
            })
            .collect::<Box<[f64]>>();

        // L D L^T decomposition in-place, lower triangle only
        for j in 0..n {
            // D[j] = M[j][j] - sum_(k < j) L[j][k]^2 * D[k]
            m[j * n + j] = (0..j)
                .map(|k| -m[j * n + k].powi(2) * m[k * n + k])
                .fold(m[j * n + j], |acc, x| acc + x);

            for i in (j + 1)..n {
                // L[i][j] = (M[i][j] - sum_(k < j) L[i][k] * L[j][k] * D[k]) / D[j]
                m[i * n + j] = (0..j)
                    .map(|k| -m[i * n + k] * m[j * n + k] * m[k * n + k])
                    .fold(m[i * n + j], |acc, x| acc + x)
                    / m[j * n + j];
            }
        }

        // forward substitution L y = e_0
        let mut y = vec![0_f64; n];
        y[unit] = 1_f64;
        for i in 0..n {
            if i == unit {
                continue;
            }

            y[i] = (0..i)
                .map(|k| -m[i * n + k] * y[k])
                .fold(0_f64, |acc, x| acc + x);
        }

        // diagonal solve D w = y
        let w = (0..n)
            .map(|i| y[i] / m[i * n + i])
            .collect::<Vec<f64>>();

        // back substitution L^T z = w
        let mut z = vec![0_f64; n];
        for i in (0..n).rev() {
            z[i] = ((i + 1)..n)
                .map(|k| -m[k * n + i] * z[k])
                .fold(w[i], |acc, x| acc + x);
        }

        // scale correction for derivatives
        let scale = ((1..=unit).product::<usize>() as f64) / (half_window.pow(unit as u32) as f64);

        // project c_k = sum_(j = 0)^n (half_window - k)^j * z[j]
        window_range
            .map(|k| {
                scale
                    * order_range
                        .clone()
                        .map(|j| ((k as f64) / (half_window as f64)).powi(j) * z[j as usize])
                        .fold(0_f64, |acc, x| acc + x)
            })
            .collect()
    }

    /// Computes the convolution of `coefficients` with itself `num - 1` times.
    ///
    /// Returns `None` if the estimated rounding error exceeds `precision`.
    fn convolve(coefficients: &[f64], num: usize, precision: f64) -> Option<Box<[f64]>> {
        if coefficients.is_empty() || num == 0 {
            return None;
        }

        let l1 = coefficients.iter().map(|c| c.abs()).sum::<f64>();
        let bound = f64::EPSILON * l1.powi(num as i32) * coefficients.len() as f64;
        if !bound.is_finite() || bound > precision {
            return None;
        }

        let init_len = coefficients.len();
        let full_len = num * (init_len - 1) + 1;
        let mut curr_len = init_len;
        let mut scratch = vec![0_f64; 2 * full_len];
        let (curr, next) = scratch.split_at_mut(full_len);
        curr[..init_len].copy_from_slice(coefficients);
        for _ in 1..num.saturating_sub(1) {
            for (i, &a) in curr[..curr_len].iter().enumerate() {
                for (j, &b) in coefficients.iter().enumerate() {
                    next[i + j] = a.mul_add(b, next[i + j]);
                }
            }

            curr_len += init_len - 1;
            curr[..curr_len].copy_from_slice(&next[..curr_len]);
            next[..curr_len].fill(0_f64);
        }
        scratch.truncate(full_len);

        Some(scratch.into_boxed_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;
    use zeenmr_spectrum::dimension::DimIndex;
    use zeenmr_spectrum::intensity_array::{ArrayOwned, shape};

    fn compare<const N: usize>(computed: [Box<[f64]>; N], expected: [Box<[f64]>; N]) {
        computed
            .iter()
            .zip(expected.iter())
            .for_each(|(computed, expected)| {
                computed
                    .iter()
                    .zip(expected.iter())
                    .for_each(|(&c, &e)| {
                        assert_approx_eq!(f64, c, e, epsilon = 1e-12);
                    })
            })
    }

    #[test]
    fn smoothing_coefficients() {
        let computed = [
            LeastSquares::coefficients(5, 2, 0),
            LeastSquares::coefficients(7, 2, 0),
            LeastSquares::coefficients(9, 2, 0),
            LeastSquares::coefficients(7, 4, 0),
            LeastSquares::coefficients(9, 4, 0),
        ];
        let expected: [Box<[f64]>; 5] = [
            [-3.0, 12.0, 17.0, 12.0, -3.0]
                .into_iter()
                .map(|c| c / 35.0)
                .collect(),
            [-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0]
                .into_iter()
                .map(|c| c / 21.0)
                .collect(),
            [-21.0, 14.0, 39.0, 54.0, 59.0, 54.0, 39.0, 14.0, -21.0]
                .into_iter()
                .map(|c| c / 231.0)
                .collect(),
            [5.0, -30.0, 75.0, 131.0, 75.0, -30.0, 5.0]
                .into_iter()
                .map(|c| c / 231.0)
                .collect(),
            [15.0, -55.0, 30.0, 135.0, 179.0, 135.0, 30.0, -55.0, 15.0]
                .into_iter()
                .map(|c| c / 429.0)
                .collect(),
        ];

        compare(computed, expected);
    }

    #[test]
    fn first_derivative_coefficients() {
        let computed = [
            LeastSquares::coefficients(3, 1, 1),
            LeastSquares::coefficients(5, 1, 1),
            LeastSquares::coefficients(7, 1, 1),
            LeastSquares::coefficients(9, 1, 1),
            LeastSquares::coefficients(5, 3, 1),
            LeastSquares::coefficients(7, 3, 1),
            LeastSquares::coefficients(9, 3, 1),
        ];
        let expected: [Box<[f64]>; 7] = [
            [-1.0, 0.0, 1.0]
                .into_iter()
                .map(|c| c / 2.0)
                .collect(),
            [-2.0, -1.0, 0.0, 1.0, 2.0]
                .into_iter()
                .map(|c| c / 10.0)
                .collect(),
            [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
                .into_iter()
                .map(|c| c / 28.0)
                .collect(),
            [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
                .into_iter()
                .map(|c| c / 60.0)
                .collect(),
            [1.0, -8.0, 0.0, 8.0, -1.0]
                .into_iter()
                .map(|c| c / 12.0)
                .collect(),
            [22.0, -67.0, -58.0, 0.0, 58.0, 67.0, -22.0]
                .into_iter()
                .map(|c| c / 252.0)
                .collect(),
            [
                86.0, -142.0, -193.0, -126.0, 0.0, 126.0, 193.0, 142.0, -86.0,
            ]
            .into_iter()
            .map(|c| c / 1188.0)
            .collect(),
        ];

        compare(computed, expected);
    }

    #[test]
    fn second_derivative_coefficients() {
        let computed = [
            LeastSquares::coefficients(5, 2, 2),
            LeastSquares::coefficients(7, 2, 2),
            LeastSquares::coefficients(9, 2, 2),
            LeastSquares::coefficients(5, 4, 2),
            LeastSquares::coefficients(7, 4, 2),
            LeastSquares::coefficients(9, 4, 2),
        ];
        let expected: [Box<[f64]>; 6] = [
            [2.0, -1.0, -2.0, -1.0, 2.0]
                .into_iter()
                .map(|c| c / 7.0)
                .collect(),
            [5.0, 0.0, -3.0, -4.0, -3.0, 0.0, 5.0]
                .into_iter()
                .map(|c| c / 42.0)
                .collect(),
            [28.0, 7.0, -8.0, -17.0, -20.0, -17.0, -8.0, 7.0, 28.0]
                .into_iter()
                .map(|c| c / 462.0)
                .collect(),
            [-1.0, 16.0, -30.0, 16.0, -1.0]
                .into_iter()
                .map(|c| c / 12.0)
                .collect(),
            [-13.0, 67.0, -19.0, -70.0, -19.0, 67.0, -13.0]
                .into_iter()
                .map(|c| c / 132.0)
                .collect(),
            [
                -126.0, 371.0, 151.0, -211.0, -370.0, -211.0, 151.0, 371.0, -126.0,
            ]
            .into_iter()
            .map(|c| c / 1716.0)
            .collect(),
        ];

        compare(computed, expected);
    }

    #[test]
    fn constant_signal() {
        let window = 11;
        let data = ArrayOwned::new(
            std::iter::repeat_n(69.0, 2_usize.pow(8)).collect::<Box<[f64]>>(),
            shape([2_usize.pow(8)]),
        )
        .expect("hand verified");
        let smoother = LeastSquares::new(1, window, 4).unwrap();
        let smoothed = smoother.smooth(data.view()).expect("infallible");

        data.elem()
            .zip(smoothed.elem())
            .for_each(|(&raw, &smoothed)| assert_approx_eq!(f64, raw, smoothed));
    }

    #[test]
    fn linear_ramp() {
        let window = 11;
        let data = ArrayOwned::new(
            (0..2_usize.pow(8))
                .map(|x| x as f64)
                .collect::<Box<[f64]>>(),
            shape([2_usize.pow(8)]),
        )
        .expect("hand verified");
        let smoother = LeastSquares::new(1, window, 4).unwrap();
        let smoothed = smoother
            .smooth(data.view())
            .ok()
            .and_then(|array| array.cropped(DimIndex(0), (window / 2)..(data.len() - window / 2)))
            .expect("infallible");
        let data = data
            .cropped(DimIndex(0), (window / 2)..(data.len() - window / 2))
            .expect("hand verified");

        data.elem()
            .zip(smoothed.elem())
            .for_each(|(&raw, &smoothed)| assert_approx_eq!(f64, raw, smoothed));
    }
}
