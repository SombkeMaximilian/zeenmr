use crate::smoothing::Smooth;
use num_traits::Float;
use std::mem::MaybeUninit;
use zeenmr_spectrum::dimension::DimIndex;
use zeenmr_spectrum::intensity_array::{ArrayViewMut1D, index};

/// Least squares filter.
///
/// Smooths a sequence of values by fitting polynomials to a sliding window and
/// replacing its central value.
///
/// Also known as Savitzky-Golay, Polynomial or DISPO filter.
///
/// Reference: Numerical Recipes The Art of Scientific Computing (3rd Edition)
/// p. 766 f.
///
/// # Edge Handling
///
/// To handle edges, the filter uses an asymmetric moving average until the full
/// window width is available.
#[derive(Clone, PartialEq, Debug)]
pub struct LeastSquares {
    /// Number of iterations to apply the filter.
    iterations: usize,
    /// Coefficient vector for the window.
    coefficients: Box<[f64]>,
}

impl<T> Smooth<T> for LeastSquares
where
    T: Float,
{
    type Error = std::convert::Infallible;

    fn smooth_in_place(&self, mut data: ArrayViewMut1D<T>) -> Result<(), Self::Error> {
        if data.len() < 2 || data.len() < self.coefficients.len() || self.iterations == 0 {
            return Ok(());
        }

        let coefficients = self
            .coefficients
            .iter()
            .map(|&weight| T::from(weight).expect("conversion from f64 to T must never fail"))
            .collect::<Vec<T>>();

        let half_window = self.coefficients.len() / 2;
        let len = data.len();
        let mut lane = data
            .lane_at_mut(DimIndex(0), &index([0]))
            .expect("1D array always has a first dimension");
        let mut gathered = MaybeUninit::<Vec<T>>::uninit();
        let (values, is_gathered) = match lane.as_mut_slice() {
            Some(lane) => (lane, false),
            None => (
                gathered
                    .write(lane.iter().copied().collect::<Vec<T>>())
                    .as_mut(),
                true,
            ),
        };
        let mut scratch = vec![T::zero(); len];
        for _ in 0..self.iterations {
            for curr in 0..half_window {
                // asymmetric moving average at the edges
                let div = T::from(curr + half_window + 1)
                    .expect("conversion from usize to T must never fail");
                scratch[curr] = values[0..curr + half_window + 1]
                    .iter()
                    .fold(T::zero(), |acc, &x| acc + x)
                    / div;
            }
            for curr in half_window..(len - half_window) {
                scratch[curr] = values[curr - half_window..curr + half_window + 1]
                    .iter()
                    .zip(coefficients.iter())
                    .map(|(&x, &c)| x * c)
                    .fold(T::zero(), |acc, x| acc + x);
            }
            for curr in (len - half_window)..len {
                // asymmetric moving average at the edges
                let div = T::from(len - curr + half_window)
                    .expect("conversion from usize to T must never fail");
                scratch[curr] = values[curr - half_window..]
                    .iter()
                    .fold(T::zero(), |acc, &x| acc + x)
                    / div;
            }
            values.swap_with_slice(&mut scratch);
        }
        if is_gathered {
            // SAFETY: this can only be reached if the lane was gathered
            let gathered = unsafe { gathered.assume_init() };
            lane.iter_mut()
                .zip(gathered)
                .for_each(|(value, new)| *value = new);
        }

        Ok(())
    }
}

impl Default for LeastSquares {
    fn default() -> Self {
        Self {
            iterations: 2,
            coefficients: coefficients(7, 2, 0),
        }
    }
}

impl LeastSquares {
    /// Creates a new `Polynomial` filter based on Savitzky-Golay coefficients.
    ///
    /// Returns `None` if
    /// - `iterations` is `0`, or
    /// - `window < 3`, or
    /// - `window = 0 mod 2`, or
    /// - `window <= order`
    pub fn new(iterations: usize, window: usize, order: u32) -> Option<Self> {
        if iterations == 0 || window < 3 || window % 2 != 1 || window <= order as usize {
            None
        } else {
            Some(Self {
                iterations,
                coefficients: coefficients(window, order, 0),
            })
        }
    }
}

/// Computes the SG coefficients using L D L^T decomposition.
pub(crate) fn coefficients(window: usize, order: u32, unit: usize) -> Box<[f64]> {
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
            coefficients(5, 2, 0),
            coefficients(7, 2, 0),
            coefficients(9, 2, 0),
            coefficients(7, 4, 0),
            coefficients(9, 4, 0),
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
            coefficients(3, 1, 1),
            coefficients(5, 1, 1),
            coefficients(7, 1, 1),
            coefficients(9, 1, 1),
            coefficients(5, 3, 1),
            coefficients(7, 3, 1),
            coefficients(9, 3, 1),
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
            coefficients(5, 2, 2),
            coefficients(7, 2, 2),
            coefficients(9, 2, 2),
            coefficients(5, 4, 2),
            coefficients(7, 4, 2),
            coefficients(9, 4, 2),
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
