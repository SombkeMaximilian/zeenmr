//! Numerically stable grid type.

use num_traits::Float;

/// Grid type that guarantees no rounding errors up to its length.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct StableGrid<T> {
    /// Step size of the grid.
    step: T,
    /// Number of error-free points in the grid.
    len: usize,
}

impl<T> StableGrid<T>
where
    T: Float,
{
    /// Creates a stable grid of length `len`.
    pub fn new(len: usize) -> Self {
        let span = len.saturating_sub(1).max(1);
        let target = len.clamp(1 << 10, 1 << 17);
        let exp = if span <= target {
            (target / span).ilog2() as i32
        } else {
            -(span.div_ceil(target).next_power_of_two().ilog2() as i32)
        };

        Self {
            step: (T::one() + T::one()).powi(exp),
            len,
        }
    }

    /// Returns the step size of the grid.
    #[inline(always)]
    pub fn step(&self) -> T {
        self.step
    }

    /// Returns the length of the grid.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the grid is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the value at `index`.
    ///
    /// There is no bounds check, so if `index` is out of bounds, the error
    /// guarantee no longer applies.
    #[inline(always)]
    pub fn get(&self, index: usize) -> T {
        self.step * T::from(index).expect("conversion from usize to T must never fail")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! stable_grid_tests {
        ($name: ident, $t:ty, max_exp = $max_exp:expr) => {
            mod $name {
                use super::*;

                const MAX_CHECK: usize = 1 << 17;
                const EXPONENTS: std::ops::Range<u32> = 1..($max_exp + 1);

                fn grids() -> impl Iterator<Item = StableGrid<$t>> {
                    EXPONENTS.map(|e| StableGrid::<$t>::new(2_usize.pow(e)))
                }

                #[test]
                fn cumulative_error() {
                    for grid in grids() {
                        let cumulative_error = (0..(grid.len.min(MAX_CHECK) - 1))
                            .zip(1..grid.len.min(MAX_CHECK))
                            .map(|(i, j)| (grid.get(i) - grid.get(j)).abs())
                            .map(|e| (e - grid.step).abs())
                            .sum::<$t>();

                        assert_eq!(cumulative_error, 0_u8 as $t);
                    }
                }

                #[test]
                fn recover_index() {
                    for grid in grids() {
                        let cumulative_error = (0..(grid.len.min(MAX_CHECK)))
                            .map(|i| ((i as $t) - (grid.get(i) / grid.step)).abs())
                            .sum::<$t>();

                        assert_eq!(cumulative_error, 0_u8 as $t);
                    }
                }
            }
        };
    }

    stable_grid_tests!(f32, f32, max_exp = 24);
    stable_grid_tests!(f64, f64, max_exp = 53);
}
