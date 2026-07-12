use crate::iter::EvaluateMap;
use crate::util::fuse_fold;
use crate::{Evaluate, EvaluateParts};
use num_traits::Zero;
use std::ops::{Div, Mul};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Problem size threshold below which parallelization does not make sense.
#[cfg(feature = "rayon")]
const PAR_THRESHOLD: usize = 0;

/// Tasks per thread scale.
///
/// Used to divide the points array into work chunks.
#[cfg(feature = "rayon")]
const TASKS_PER_THREAD: usize = 4;

/// Superposition strategy for computing the sum of many functions at many
/// points.
///
/// # Formulation
///
/// Let `x` be the `m`-dimensional vector of points to compute the superposition
/// at, and `f₁, …, fₙ` be the functions. Further, let `M` be the `n x m` matrix
/// of function evaluations:
///
/// ```text
/// Mᵢⱼ = fᵢ(xⱼ)
/// ```
///
/// The evaluation of the superposition is then `y = M 1` where `1` is the
/// column vector filled with the multiplicative identity:
///
/// ```text
/// yᵢ = F(xᵢ) = f₁(xᵢ) + f₂(xᵢ) + ... + fₙ(xᵢ)
/// ```
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub enum Strategy {
    /// Uses a heuristic to pick the best computation order.
    #[default]
    Auto,
    /// Computes the rows of `M` one by one and performs pairwise reduction to
    /// get `y`.
    FunctionsOuter,
    /// Computes `y` in chunks of subrows of `M`.
    ///
    /// Multiplies `M` by vectors filled with `k` consecutive multiplicative
    /// identities and otherwise only zeros.
    ///
    /// This approach pays off when the function parameters fully fit into the
    /// L1 cache alongside the subrows.
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
    fn resolve(self, n: usize, m: usize) -> (usize, usize) {
        match self {
            // current best from benchmarks
            Strategy::Auto => (4096.min(n), 128.min(m)),
            Strategy::FunctionsOuter => (n.max(1), m.max(1)),
            Strategy::Subvectors { p } => (n.max(1), p.min(m).max(1)),
            Strategy::Submatrices { p, f } => (f.min(n).max(1), p.min(m).max(1)),
        }
    }
}

/// A collection of functions that can be superposed over a grid of points.
pub trait BatchSuperposition<T> {
    /// Performs superposition with the given strategy.
    fn superposition_with(&self, at: &[T], strategy: Strategy) -> Vec<T>;

    /// F(x) = (f₁(x₁) + … + fₙ(x₁), ..., f₁(xₘ) + … + fₙ(xₘ)).
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
        let (rows, cols) = strategy.resolve(self.len(), at.len());

        schedule_to_owned(self, at, rows, cols)
    }
}

/// A collection of functions that can be superposed over a grid of points in
/// parallel.
#[cfg(feature = "rayon")]
pub trait ParBatchSuperposition<T> {
    /// Performs superposition with the given strategy.
    fn par_superposition_with(&self, at: &[T], strategy: Strategy) -> Vec<T>;

    /// F(x) = (f₁(x₁) + … + fₙ(x₁), ..., f₁(xₘ) + … + fₙ(xₘ)).
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
        let (rows, cols) = strategy.resolve(self.len(), at.len());

        if self.len().saturating_mul(at.len()) < PAR_THRESHOLD {
            return schedule_to_owned(self, at, rows, cols);
        }

        let mut out = vec![T::zero(); at.len()];
        let task_size = task_size(at.len(), cols);
        out.par_chunks_mut(task_size)
            .zip(at.par_chunks(task_size))
            .for_each(|(out, at)| schedule(self, at, out, rows, cols));

        out
    }
}

/// A collection of functions that can be superposed over a grid of points using
/// a fusion transformation.
pub trait FusedBatchSuperposition<T> {
    /// Performs the fused superposition with the given strategy.
    fn fused_superposition_with<const K: usize>(&self, at: &[T], strategy: Strategy) -> Vec<T>;

    /// F(x) = (f₁(x₁) + … + fₙ(x₁), ..., f₁(xₘ) + … + fₙ(xₘ)).
    fn fused_superposition<const K: usize>(&self, at: &[T]) -> Vec<T> {
        self.fused_superposition_with::<K>(at, Strategy::Auto)
    }
}

impl<T, E> FusedBatchSuperposition<T> for [E]
where
    T: Copy + Mul<T, Output = T> + Div<T, Output = T> + Zero,
    E: EvaluateParts<T>,
{
    fn fused_superposition_with<const K: usize>(&self, at: &[T], strategy: Strategy) -> Vec<T> {
        let (rows, cols) = strategy.resolve(self.len(), at.len());

        schedule_fused_to_owned::<T, E, K>(self, at, rows, cols)
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
    for f_chunk in functions.chunks(rows) {
        for (at_chunk, dest_chunk) in at.chunks(cols).zip(dest.chunks_mut(cols)) {
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

fn schedule_fused_to_owned<T, E, const K: usize>(
    functions: &[E],
    at: &[T],
    rows: usize,
    cols: usize,
) -> Vec<T>
where
    T: Copy + Mul<T, Output = T> + Div<T, Output = T> + Zero,
    E: EvaluateParts<T>,
{
    let mut out = vec![T::zero(); at.len()];
    schedule_fused::<T, E, K>(functions, at, &mut out, rows, cols);

    out
}

fn schedule_fused<T, E, const K: usize>(
    functions: &[E],
    at: &[T],
    out: &mut [T],
    rows: usize,
    cols: usize,
) where
    T: Copy + Mul<T, Output = T> + Div<T, Output = T> + Zero,
    E: EvaluateParts<T>,
{
    for f_chunk in functions.chunks(rows) {
        for (at_c, out_c) in at.chunks(cols).zip(out.chunks_mut(cols)) {
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

/// Computes the task size depending on the number of points and columns of the
/// resolved strategy.
#[cfg(feature = "rayon")]
fn task_size(points: usize, cols: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    let target = points.div_ceil(threads * TASKS_PER_THREAD).max(1);

    if target <= cols {
        target
    } else {
        target.div_ceil(cols) * cols
    }
}
