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

/// Superposition strategy for computing the sum of many functions at many
/// points.
///
/// # Formulation
///
/// Let `x` be the `n`-dimensional vector of points to compute the superposition
/// at, and `f₁, …, fₘ` be the functions. Further, let `M` be the `n x m` matrix
/// of function evaluations:
///
/// ```text
/// Mᵢⱼ = fⱼ(xᵢ)
/// ```
///
/// The evaluation of the superposition is then `y = M 1` where `1` is the
/// column vector filled with the multiplicative identity:
///
/// ```text
/// yᵢ = F(xᵢ) = f₁(xᵢ) + f₂(xᵢ) + ... + fₘ(xᵢ)
/// ```
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub enum Strategy {
    /// Uses a heuristic to pick the best computation order.
    ///
    /// This option should be used unless you have found a reason not to.
    #[default]
    Auto,
    /// Computes the rows of `M` one by one and performs pairwise reduction to
    /// get `y`.
    FunctionsOuter,
    /// Computes `y` in chunks of subcolumns of `M`.
    ///
    /// Multiplies `M` by vectors filled with `k` consecutive multiplicative
    /// identities and otherwise only zeros.
    ///
    /// This approach pays off when the function parameters fully fit into the
    /// L1 cache alongside the subcolumns.
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

/// A collection of functions that can be superposed over a grid of points.
pub trait BatchSuperposition<T> {
    /// Performs superposition with the given strategy.
    fn superposition_with(&self, at: &[T], strategy: Strategy) -> Vec<T>;

    /// F(x) = (f₁(x₁) + … + fₘ(x₁), ..., f₁(xₙ) + … + fₘ(xₙ)).
    fn superposition(&self, at: &[T]) -> Vec<T> {
        self.superposition_with(at, Strategy::Auto)
    }
}

impl<T, E> BatchSuperposition<T> for [E]
where
    T: Copy + Zero,
    E: Evaluate<T>,
{
    fn superposition_with(&self, at: &[T], strategy: Strategy) -> Vec<T> {
        let (rows, cols) = strategy.resolve::<T, E>(self, at, false);

        schedule_to_owned(self, at, rows, cols)
    }
}

/// A collection of functions that can be superposed over a grid of points in
/// parallel.
#[cfg(feature = "rayon")]
pub trait ParBatchSuperposition<T> {
    /// Performs superposition with the given strategy.
    fn par_superposition_with(&self, at: &[T], strategy: Strategy) -> Vec<T>;

    /// F(x) = (f₁(x₁) + … + fₘ(x₁), ..., f₁(xₙ) + … + fₘ(xₙ)).
    fn par_superposition(&self, at: &[T]) -> Vec<T> {
        self.par_superposition_with(at, Strategy::Auto)
    }
}

#[cfg(feature = "rayon")]
impl<T, E> ParBatchSuperposition<T> for [E]
where
    T: Copy + Zero + Send + Sync,
    E: Evaluate<T> + Sync,
{
    fn par_superposition_with(&self, at: &[T], strategy: Strategy) -> Vec<T> {
        let parallel = working_set(self, at) >= PAR_THRESHOLD;
        let (rows, cols) = strategy.resolve::<T, E>(self, at, parallel);

        if !parallel {
            return schedule_to_owned(self, at, rows, cols);
        }

        let mut out = vec![T::zero(); at.len()];
        let task_size = task_size::<T>(at.len(), rows);
        out.par_chunks_mut(task_size)
            .zip(at.par_chunks(task_size))
            .for_each(|(out, at)| schedule(self, at, out, rows, cols));

        out
    }
}

/// Schedules the superposition into (row, col) chunks of `M` and returns it as
/// an owned `Vec<T>`.
fn schedule_to_owned<T, E>(functions: &[E], at: &[T], rows: usize, cols: usize) -> Vec<T>
where
    T: Copy + Zero,
    E: Evaluate<T>,
{
    let mut out = vec![T::zero(); at.len()];
    schedule(functions, at, &mut out, rows, cols);

    out
}

/// Schedules the superposition into (row, col) chunks of `M`, writing it into
/// `out`.
fn schedule<T, E>(functions: &[E], at: &[T], dest: &mut [T], rows: usize, cols: usize)
where
    T: Copy + Zero,
    E: Evaluate<T>,
{
    for f_chunk in functions.chunks(cols) {
        for (at_chunk, dest_chunk) in at.chunks(rows).zip(dest.chunks_mut(rows)) {
            for f in f_chunk {
                for (d, eval) in dest_chunk
                    .iter_mut()
                    .zip(at_chunk.iter().copied().evaluate(f))
                {
                    *d = *d + eval;
                }
            }
        }
    }
}

/// A collection of functions that can be superposed over a grid of points using
/// a fusion transformation.
pub trait FusedBatchSuperposition<T> {
    /// Performs the fused superposition.
    fn fused_superposition_with(&self, at: &[T], strategy: Strategy, width: FuseWidth) -> Vec<T>;

    /// F(x) = (f₁(x₁) + … + fₘ(x₁), ..., f₁(xₙ) + … + fₘ(xₙ)).
    fn fused_superposition(&self, at: &[T]) -> Vec<T> {
        self.fused_superposition_with(at, Strategy::Auto, FuseWidth::PickBest)
    }
}

impl<T, E> FusedBatchSuperposition<T> for [E]
where
    T: Float,
    E: EvaluateParts<T>,
{
    fn fused_superposition_with(&self, at: &[T], strategy: Strategy, width: FuseWidth) -> Vec<T> {
        let (rows, cols) = strategy.resolve::<T, E>(self, at, false);

        match width.resolve(self, at) {
            FuseWidth::Eight => schedule_fused_to_owned::<T, E, 8>(self, at, rows, cols),
            FuseWidth::Four => schedule_fused_to_owned::<T, E, 4>(self, at, rows, cols),
            FuseWidth::Two => schedule_fused_to_owned::<T, E, 2>(self, at, rows, cols),
            _ => schedule_to_owned(self, at, rows, cols),
        }
    }
}

/// A collection of functions that can be superposed over a grid of points using
/// a fusion transformation in parallel.
#[cfg(feature = "rayon")]
pub trait ParFusedBatchSuperposition<T> {
    /// Performs the fused superposition in parallel.
    fn par_fused_superposition_with(
        &self,
        at: &[T],
        strategy: Strategy,
        width: FuseWidth,
    ) -> Vec<T>;

    /// F(x) = (f₁(x₁) + … + fₘ(x₁), ..., f₁(xₙ) + … + fₘ(xₙ)).
    fn par_fused_superposition(&self, at: &[T]) -> Vec<T> {
        self.par_fused_superposition_with(at, Strategy::Auto, FuseWidth::PickBest)
    }
}

#[cfg(feature = "rayon")]
impl<T, E> ParFusedBatchSuperposition<T> for [E]
where
    T: Float + Send + Sync,
    E: EvaluateParts<T> + Sync,
{
    fn par_fused_superposition_with(
        &self,
        at: &[T],
        strategy: Strategy,
        width: FuseWidth,
    ) -> Vec<T> {
        let width = width.resolve(self, at);
        let parallel = working_set(self, at) >= PAR_THRESHOLD;
        let (rows, cols) = strategy.resolve::<T, E>(self, at, parallel);

        if !parallel {
            return match width {
                FuseWidth::Eight => schedule_fused_to_owned::<T, E, 8>(self, at, rows, cols),
                FuseWidth::Four => schedule_fused_to_owned::<T, E, 4>(self, at, rows, cols),
                FuseWidth::Two => schedule_fused_to_owned::<T, E, 2>(self, at, rows, cols),
                _ => schedule_to_owned(self, at, rows, cols),
            };
        }

        let mut out = vec![T::zero(); at.len()];
        let task_size = task_size::<T>(at.len(), rows);
        let iter = out
            .par_chunks_mut(task_size)
            .zip(at.par_chunks(task_size));
        match width {
            FuseWidth::Eight => {
                iter.for_each(|(out, at)| schedule_fused::<T, E, 8>(self, at, out, rows, cols))
            }
            FuseWidth::Four => {
                iter.for_each(|(out, at)| schedule_fused::<T, E, 4>(self, at, out, rows, cols))
            }
            FuseWidth::Two => {
                iter.for_each(|(out, at)| schedule_fused::<T, E, 2>(self, at, out, rows, cols))
            }
            _ => iter.for_each(|(out, at)| schedule(self, at, out, rows, cols)),
        }

        out
    }
}

/// Number of evaluations to fuse.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
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
    /// Returns an iterator over the variants that are not `PickBest`.
    pub fn iter() -> impl DoubleEndedIterator<Item = FuseWidth> {
        [
            FuseWidth::One,
            FuseWidth::Two,
            FuseWidth::Four,
            FuseWidth::Eight,
        ]
        .into_iter()
    }

    /// Resolves `PickBest` against the data and returns self otherwise.
    ///
    /// Largest `K ∈ {8, 4, 2}` for which the fused kernel cannot overflow or
    /// underflow on this data. Returns 1 if no fusion is safe.
    ///
    /// Never returns `PickBest`.
    fn resolve<T, E>(self, functions: &[E], at: &[T]) -> FuseWidth
    where
        T: Float,
        E: EvaluateParts<T>,
    {
        if functions.is_empty() || at.is_empty() {
            return FuseWidth::One;
        }

        match self {
            FuseWidth::PickBest => {
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
                let widest_safe = FuseWidth::iter()
                    .rev()
                    .find(|&k| k.is_safe((d_min, d_max), n_max))
                    .unwrap_or(FuseWidth::One);

                match size_of::<T>() {
                    // override for f32 from benchmark
                    4 if widest_safe as u32 >= 4 => FuseWidth::Four,
                    // override for f64 from benchmark
                    8 if widest_safe as u32 >= 8 => FuseWidth::Eight,
                    _ => widest_safe,
                }
            }
            _ => self,
        }
    }

    /// Returns `true` if this fuse width does not cause over- or underflow.
    fn is_safe<T>(&self, (d_min, d_max): (T, T), n_max: T) -> bool
    where
        T: Float,
    {
        let k = if let Self::PickBest = self {
            return false;
        } else {
            *self as i32
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

/// [`schedule_to_owned`] with fused evaluations of width `K`.
fn schedule_fused_to_owned<T, E, const K: usize>(
    functions: &[E],
    at: &[T],
    rows: usize,
    cols: usize,
) -> Vec<T>
where
    T: Float,
    E: EvaluateParts<T>,
{
    let mut out = vec![T::zero(); at.len()];
    schedule_fused::<T, E, K>(functions, at, &mut out, rows, cols);

    out
}

/// [`schedule`] with fused evaluations of width `K`.
fn schedule_fused<T, E, const K: usize>(
    functions: &[E],
    at: &[T],
    out: &mut [T],
    rows: usize,
    cols: usize,
) where
    T: Float,
    E: EvaluateParts<T>,
{
    // round up to nearest multiple of K. this might overflow if rows is near
    // usize::MAX but at that point something went wrong already anyway
    debug_assert!(cols.checked_add(K).is_some());
    let cols = (cols + K - 1) & !(K - 1);

    for f_chunk in functions.chunks(cols) {
        for (at_c, out_c) in at.chunks(rows).zip(out.chunks_mut(rows)) {
            let mut groups = f_chunk.chunks_exact(K);
            for g in &mut groups {
                let g: &[E; K] = g.try_into().expect("chunks_exact yields K");
                for (o, &x) in out_c.iter_mut().zip(at_c) {
                    let (num, den) = fuse_fold::<T, K>(std::array::from_fn(|i| g[i].parts(x)));
                    *o = *o + num / den;
                }
            }
            for f in groups.remainder() {
                for (o, &x) in out_c.iter_mut().zip(at_c) {
                    *o = *o + f.evaluate(x);
                }
            }
        }
    }
}

/// Computes the size of the working set.
#[cfg(feature = "rayon")]
fn working_set<T, E>(functions: &[E], at: &[T]) -> usize {
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
