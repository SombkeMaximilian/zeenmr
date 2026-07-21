//! Superposition kernels for large data.
//!
//! # Formulation
//!
//! Let `x` be the `n`-dimensional vector of points to compute the superposition
//! at, and `f₁, …, fₘ` be the functions. Further, let `M` be the `n x m` matrix
//! of function evaluations:
//!
//! ```text
//! Mᵢⱼ = fⱼ(xᵢ)
//! ```
//!
//! The evaluation of the superposition is then `y = M 1` where `1` is the
//! column vector filled with the multiplicative identity:
//!
//! ```text
//! yᵢ = F(xᵢ) = f₁(xᵢ) + f₂(xᵢ) + ... + fₘ(xᵢ)
//! ```

use crate::iter::EvaluateMap;
use crate::util::fuse_fold;
use crate::{Evaluate, EvaluateParts};
use num_traits::{Float, Zero};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

mod cache_topology {
    //! Contains cache size constants in bytes.

    include!(concat!(env!("OUT_DIR"), "/cache.rs"));
}

/// Points streamed in multiples of a cache line at a time.
const POINT_LINES: usize = 8;

/// Problem size (in bytes) threshold below which parallelization does not make
/// sense.
#[cfg(feature = "rayon")]
const PAR_THRESHOLD: usize = 2_usize * cache_topology::L1D;

/// Tasks per thread scale.
///
/// Used to divide the points array into work chunks.
#[cfg(feature = "rayon")]
const TASKS_PER_THREAD: usize = 4;

/// Superposition implementation.
pub trait SuperpositionKernel<T, E> {
    /// Accumulates the superposition.
    ///
    /// Only the first `min(at.len(), acc.len())` points are computed. If the
    /// lengths differ the result is a partial superposition. Callers are
    /// responsible for matching them.
    fn accumulate(&self, functions: &[E], at: &[T], acc: &mut [T]);

    /// Computes the superposition and returns it as an owned `Vec<T>`.
    ///
    /// # Note for Implementors
    ///
    /// This method should almost always use the provided implementation,
    /// unless there is a provably more efficient version.
    fn superposition(&self, functions: &[E], at: &[T]) -> Vec<T>
    where
        T: Copy + Zero,
    {
        let mut acc = vec![T::zero(); at.len()];
        self.accumulate(functions, at, &mut acc);

        acc
    }
}

impl<T, E, K> SuperpositionKernel<T, E> for &K
where
    K: SuperpositionKernel<T, E> + ?Sized,
{
    fn accumulate(&self, functions: &[E], at: &[T], acc: &mut [T]) {
        (**self).accumulate(functions, at, acc);
    }

    fn superposition(&self, functions: &[E], at: &[T]) -> Vec<T>
    where
        T: Copy + Zero,
    {
        (**self).superposition(functions, at)
    }
}

impl<T, E, K> SuperpositionKernel<T, E> for &mut K
where
    K: SuperpositionKernel<T, E> + ?Sized,
{
    fn accumulate(&self, functions: &[E], at: &[T], acc: &mut [T]) {
        (**self).accumulate(functions, at, acc);
    }

    fn superposition(&self, functions: &[E], at: &[T]) -> Vec<T>
    where
        T: Copy + Zero,
    {
        (**self).superposition(functions, at)
    }
}

/// Parallel superposition implementation.
#[cfg(feature = "rayon")]
pub trait ParSuperpositionKernel<T, E> {
    /// Accumulates the superposition in parallel.
    ///
    /// Only the first `min(at.len(), acc.len())` points are computed. If the
    /// lengths differ the result is a partial superposition. Callers are
    /// responsible for matching them.
    fn par_accumulate(&self, functions: &[E], at: &[T], acc: &mut [T]);

    /// Computes the superposition in parallel and returns it as an owned
    /// `Vec<T>`.
    ///
    /// # Note for Implementors
    ///
    /// This method should almost always use the provided implementation,
    /// unless there is a provably more efficient version.
    fn par_superposition(&self, functions: &[E], at: &[T]) -> Vec<T>
    where
        T: Copy + Zero,
    {
        let mut acc = vec![T::zero(); at.len()];
        self.par_accumulate(functions, at, &mut acc);

        acc
    }
}

#[cfg(feature = "rayon")]
impl<T, E, K> ParSuperpositionKernel<T, E> for &K
where
    K: ParSuperpositionKernel<T, E> + ?Sized,
{
    fn par_accumulate(&self, functions: &[E], at: &[T], acc: &mut [T]) {
        (**self).par_accumulate(functions, at, acc);
    }

    fn par_superposition(&self, functions: &[E], at: &[T]) -> Vec<T>
    where
        T: Copy + Zero,
    {
        (**self).par_superposition(functions, at)
    }
}

#[cfg(feature = "rayon")]
impl<T, E, K> ParSuperpositionKernel<T, E> for &mut K
where
    K: ParSuperpositionKernel<T, E> + ?Sized,
{
    fn par_accumulate(&self, functions: &[E], at: &[T], acc: &mut [T]) {
        (**self).par_accumulate(functions, at, acc);
    }

    fn par_superposition(&self, functions: &[E], at: &[T]) -> Vec<T>
    where
        T: Copy + Zero,
    {
        (**self).par_superposition(functions, at)
    }
}

/// Superposition strategy for computing the sum of many functions at many
/// points.
#[non_exhaustive]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub enum Strategy {
    /// Uses a heuristic to pick the best computation order.
    ///
    /// This option should be used unless you have found a reason not to.
    #[default]
    Auto,
    /// Computes `y` by processing the columns of `M` one by one.
    FunctionsOuter,
    /// Computes `y` in chunks of subcolumns of `M`.
    Subvectors {
        /// Number of points to process in each chunk.
        p: usize,
    },
    /// Computes `y` in chunks of submatrices of `M`.
    Submatrices {
        /// Number of points to process in each chunk.
        p: usize,
        /// Number of functions to process in each chunk.
        f: usize,
    },
}

impl Strategy {
    /// Resolve to `(rows, cols)`.
    fn resolve<T, E>(self, functions: &[E], at: &[T], parallel: bool) -> (usize, usize) {
        let (n, m) = (at.len(), functions.len());
        let (rows, cols) = match self {
            // current best from benchmarks
            Strategy::Auto => match parallel {
                false => serial_submatrix::<T, E>(),
                true => parallel_submatrix::<T, E>(),
            },
            Strategy::FunctionsOuter => (n, m),
            Strategy::Subvectors { p } => (p, m),
            Strategy::Submatrices { p, f } => (p, f),
        };

        (rows.min(n).max(1), cols.min(m).max(1))
    }
}

/// Standard formulation of the superposition problem.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub struct Standard {
    /// Scheduling strategy used for computing the superposition.
    strategy: Strategy,
}

impl<T, E> SuperpositionKernel<T, E> for Standard
where
    T: Copy + Zero,
    E: Evaluate<T>,
{
    fn accumulate(&self, functions: &[E], at: &[T], acc: &mut [T]) {
        let (rows, cols) = self.strategy.resolve(functions, at, false);
        schedule(functions, at, acc, rows, cols);
    }
}

#[cfg(feature = "rayon")]
impl<T, E> ParSuperpositionKernel<T, E> for Standard
where
    T: Copy + Send + Sync + Zero,
    E: Evaluate<T> + Sync,
{
    fn par_accumulate(&self, functions: &[E], at: &[T], acc: &mut [T]) {
        let parallel = working_set(functions, at) >= PAR_THRESHOLD;
        let (rows, cols) = self.strategy.resolve(functions, at, parallel);
        if !parallel {
            schedule(functions, at, acc, rows, cols);
        } else {
            let task_size = task_size::<T>(at.len(), rows);
            acc.par_chunks_mut(task_size)
                .zip(at.par_chunks(task_size))
                .for_each(|(acc, at)| schedule(functions, at, acc, rows, cols))
        }
    }
}

impl Standard {
    /// Creates a new `Standard` superposition kernel.
    pub const fn new() -> Self {
        Self {
            strategy: Strategy::Auto,
        }
    }

    /// Sets the scheduling strategy.
    pub const fn with_strategy(self, strategy: Strategy) -> Self {
        Self { strategy }
    }
}

/// Number of evaluations to fuse.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
#[non_exhaustive]
#[repr(u8)]
pub enum FuseWidth {
    /// Automatically picks the highest width that carries no risk of over- or
    /// underflow.
    #[default]
    PickBest = 0,
    /// No fusing, equivalent to regular evaluation.
    One = 1,
    /// Fuses two evaluations.
    Two = 2,
    /// Fuses four evaluations.
    Four = 4,
    /// Fuses eight evaluations.
    Eight = 8,
}

impl FuseWidth {
    /// Returns the scheduling function for the fuse width.
    fn resolve<T, E>(self, functions: &[E], at: &[T]) -> fn(&[E], &[T], &mut [T], usize, usize)
    where
        T: Float,
        E: EvaluateParts<T>,
    {
        if functions.is_empty() || at.is_empty() {
            return schedule::<T, E>;
        }
        let width = match self {
            FuseWidth::PickBest => Self::pick_best(functions, at),
            _ => self,
        };

        match width {
            FuseWidth::Eight => schedule_fused::<T, E, 8>,
            FuseWidth::Four => schedule_fused::<T, E, 4>,
            FuseWidth::Two => schedule_fused::<T, E, 2>,
            _ => schedule::<T, E>,
        }
    }

    /// Checks against the data and returns the optimal, safe fuse width.
    ///
    /// Largest `K ∈ {8, 4, 2}` for which the fused kernel cannot overflow or
    /// underflow on this data. Returns `1` if no fusion is safe.
    fn pick_best<T, E>(functions: &[E], at: &[T]) -> Self
    where
        T: Float,
        E: EvaluateParts<T>,
    {
        let (lo, hi) = at
            .iter()
            .fold((T::infinity(), T::neg_infinity()), |(lo, hi), &x| {
                (lo.min(x), hi.max(x))
            });
        let (d_min, d_max, n_max) = functions.iter().fold(
            (T::infinity(), T::neg_infinity(), T::zero()),
            |(d_min, d_max, n_max), f| {
                let (d_lo, d_hi) = f.den_bounds(lo, hi);
                let (_, n_hi) = f.num_bounds(lo, hi);

                (d_min.min(d_lo), d_max.max(d_hi), n_max.max(n_hi))
            },
        );
        let widest_safe = [FuseWidth::Eight, FuseWidth::Four, FuseWidth::Two]
            .into_iter()
            .find(|&k| k.is_safe((d_min, d_max), n_max))
            .unwrap_or(FuseWidth::One);

        match size_of::<T>() {
            // override for f32 from benchmark
            4 if widest_safe as u8 >= 4 => FuseWidth::Four,
            // override for f64 from benchmark
            8 if widest_safe as u8 >= 8 => FuseWidth::Eight,
            _ => widest_safe,
        }
    }

    /// Returns `true` if this fuse width does not cause over- or underflow.
    fn is_safe<T>(self, (d_min, d_max): (T, T), n_max: T) -> bool
    where
        T: Float,
    {
        let k = if let Self::PickBest = self {
            return false;
        } else {
            self as i32
        };

        let margin = T::from(1e3_f64).expect("conversion from f64 to T must never fail");
        let max = T::max_value() / margin;
        let min = T::min_positive_value() * margin;
        let k_as_t = T::from(k).expect("conversion from i32 to T must never fail");

        // fused denominator multiplies K such values, which can overflow
        d_max.powi(k) < max
            // or underflow
            && d_min.powi(k) > min
            // numerator sums K mixed products with K terms each
            && k_as_t * n_max * d_max.powi(k - 1) < max
    }
}

/// Fused formulation of the superposition problem.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub struct Fused {
    /// Scheduling strategy used for computing the superposition.
    strategy: Strategy,
    /// Number of evaluations to fuse.
    width: FuseWidth,
}

impl<T, E> SuperpositionKernel<T, E> for Fused
where
    T: Float,
    E: EvaluateParts<T>,
{
    fn accumulate(&self, functions: &[E], at: &[T], acc: &mut [T]) {
        let (rows, cols) = self.strategy.resolve(functions, at, false);
        self.width.resolve(functions, at)(functions, at, acc, rows, cols);
    }
}

#[cfg(feature = "rayon")]
impl<T, E> ParSuperpositionKernel<T, E> for Fused
where
    T: Float + Send + Sync,
    E: EvaluateParts<T> + Sync,
{
    fn par_accumulate(&self, functions: &[E], at: &[T], acc: &mut [T]) {
        let parallel = working_set(functions, at) >= PAR_THRESHOLD;
        let (rows, cols) = self.strategy.resolve(functions, at, parallel);
        let resolved_schedule = self.width.resolve(functions, at);
        if !parallel {
            resolved_schedule(functions, at, acc, rows, cols);
        } else {
            let task_size = task_size::<T>(at.len(), rows);
            acc.par_chunks_mut(task_size)
                .zip(at.par_chunks(task_size))
                .for_each(|(acc, at)| resolved_schedule(functions, at, acc, rows, cols));
        }
    }
}

impl Fused {
    /// Creates a new `Fused` superposition kernel.
    pub const fn new() -> Self {
        Self {
            strategy: Strategy::Auto,
            width: FuseWidth::PickBest,
        }
    }

    /// Sets the scheduling strategy.
    pub const fn with_strategy(self, strategy: Strategy) -> Self {
        Self {
            strategy,
            width: self.width,
        }
    }

    /// Sets the fuse width.
    pub const fn with_width(self, width: FuseWidth) -> Self {
        Self {
            width,
            strategy: self.strategy,
        }
    }
}

/// Schedules the superposition into (row, col) chunks of `M`, writing it into
/// `acc`.
fn schedule<T, E>(functions: &[E], at: &[T], acc: &mut [T], rows: usize, cols: usize)
where
    T: Copy + Zero,
    E: Evaluate<T>,
{
    for f_chunk in functions.chunks(cols) {
        for (at_c, acc_c) in at.chunks(rows).zip(acc.chunks_mut(rows)) {
            for f in f_chunk {
                for (a, eval) in acc_c
                    .iter_mut()
                    .zip(at_c.iter().copied().evaluate(f))
                {
                    *a = *a + eval;
                }
            }
        }
    }
}

/// [`schedule`] with fused evaluations of width `K`.
fn schedule_fused<T, E, const K: usize>(
    functions: &[E],
    at: &[T],
    acc: &mut [T],
    rows: usize,
    cols: usize,
) where
    T: Float,
    E: EvaluateParts<T>,
{
    // round up to nearest multiple of K. this might overflow if cols is near
    // usize::MAX but at that point something went wrong already anyway
    debug_assert!(K.is_power_of_two());
    debug_assert!(cols.checked_add(K).is_some());
    let cols = (cols + K - 1) & !(K - 1);

    for f_chunk in functions.chunks(cols) {
        for (at_c, acc_c) in at.chunks(rows).zip(acc.chunks_mut(rows)) {
            let mut fuse_groups = f_chunk.chunks_exact(K);
            for f_group in &mut fuse_groups {
                let f_group: &[E; K] = f_group.try_into().expect("chunks_exact yields K");
                for (a, &x) in acc_c.iter_mut().zip(at_c) {
                    let (num, den) =
                        fuse_fold::<T, K>(std::array::from_fn(|i| f_group[i].parts(x)));
                    *a = *a + num / den;
                }
            }
            for f in fuse_groups.remainder() {
                for (a, eval) in acc_c
                    .iter_mut()
                    .zip(at_c.iter().copied().evaluate(f))
                {
                    *a = *a + eval;
                }
            }
        }
    }
}

/// Uses the cache topology to determine the optimal submatrix shape
/// `(rows, cols)`.
const fn serial_submatrix<T, E>() -> (usize, usize) {
    submatrix::<T, E>(cache_topology::L1D)
}

/// Uses the cache topology to determine the optimal parallel submatrix shape
/// `(rows, cols)`.
const fn parallel_submatrix<T, E>() -> (usize, usize) {
    submatrix::<T, E>(cache_topology::L1D_PER_THREAD)
}

/// Computes submatrix shape `(rows, cols)` that optimally uses the L1 cache.
const fn submatrix<T, E>(l1: usize) -> (usize, usize) {
    let t = size_of::<T>();
    let e = size_of::<E>();
    assert!(t != 0);
    assert!(e != 0);
    assert!(e % t == 0);

    let elem_per_line = match cache_topology::CACHE_LINE / t {
        0 => 1,
        elem => elem,
    };
    let rows = POINT_LINES * elem_per_line;
    let point_bytes = 2 * rows * t;

    let usable = l1.saturating_mul(4) / 5;
    let col_bytes = usable.saturating_sub(point_bytes);
    let cols = match col_bytes / e {
        0 => 1,
        elems => elems,
    };

    (rows, cols)
}

/// Computes the size of the working set.
#[cfg(feature = "rayon")]
const fn working_set<T, E>(functions: &[E], at: &[T]) -> usize {
    let t_2 = 2 * size_of::<T>();
    let e = size_of::<E>();
    let functions_bytes = e.saturating_mul(functions.len());
    let points_bytes = t_2.saturating_mul(at.len());

    functions_bytes.saturating_add(points_bytes)
}

/// Computes the task size depending on the number of points and columns of the
/// resolved strategy.
#[cfg(feature = "rayon")]
fn task_size<T>(points: usize, rows: usize) -> usize {
    let l2_cap = (cache_topology::L2_PER_THREAD / (2 * size_of::<T>())).max(1);

    let threads = rayon::current_num_threads().max(1);
    let target = points
        .div_ceil(threads * TASKS_PER_THREAD)
        .max(1)
        .min(l2_cap);

    if target <= rows {
        target
    } else {
        target.div_ceil(rows) * rows
    }
}
