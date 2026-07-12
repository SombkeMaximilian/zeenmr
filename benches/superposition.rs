use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use num_traits::Float;
use rand::distr::Uniform;
use rand::distr::uniform::SampleUniform;
use rand::prelude::*;
use rayon::ThreadPoolBuilder;
use std::fmt::Debug;
use std::hint::black_box;
use std::time::Duration;
use zeenmr::peak_shape::{
    BatchSuperposition, Evaluate, FromArray, Gaussian, Lorentzian, ParBatchSuperposition, Strategy,
};

const SEED: u64 = 0xC0FF_EE15_600D;

trait PeakName {
    const NAME: &'static str;
}

impl<T> PeakName for Lorentzian<T> {
    const NAME: &'static str = "Lorentzian";
}

impl<T> PeakName for Gaussian<T> {
    const NAME: &'static str = "Gaussian";
}

/// Generates some random function parameters.
fn make_functions<T, E, const N: usize>(rng: &mut StdRng, dist: &Uniform<T>, n: usize) -> Vec<E>
where
    T: Float + SampleUniform,
    E: FromArray<T, N>,
{
    (0..n)
        .map(|_| E::from_array(std::array::from_fn(|_| dist.sample(rng))))
        .collect()
}

/// Makes the evaluation grid.
fn make_grid<T: Float>(m: usize) -> Vec<T> {
    let scale = T::from(100).expect("conversion from usize to T must never fail")
        / T::from(m).expect("conversion from usize to T must never fail");

    (0..m)
        .map(|i| T::from(i).expect("conversion from usize to T must never fail"))
        .map(|t| t * scale)
        .collect()
}

// include auto later
const STRATEGIES: [(&str, Strategy); 5] = [
    ("functions_outer", Strategy::FunctionsOuter),
    ("subvectors_512", Strategy::Subvectors { p: 512 }),
    ("subvectors_1024", Strategy::Subvectors { p: 1024 }),
    (
        "submatrices_512_2048",
        Strategy::Submatrices { p: 2048, f: 512 },
    ),
    (
        "submatrices_128_4096",
        Strategy::Submatrices { p: 4096, f: 128 },
    ),
];

/// (evaluators, points).
///
/// Iso-work: `n * m == 2^27` for every entry, so any throughput difference is
/// pure scheduling effect.
const ISO_WORK: [(usize, usize); 5] = [
    (1 << 10, 1 << 17),
    (1 << 12, 1 << 15),
    (1 << 14, 1 << 13),
    (1 << 15, 1 << 12),
    (1 << 16, 1 << 11),
];

/// Number of threads to run.
const THREADS: [usize; 5] = [1, 2, 4, 8, 16];

fn iso_work<T, E, const N: usize>(c: &mut Criterion, dtype: &str)
where
    T: Float + SampleUniform + Debug,
    E: Evaluate<T> + FromArray<T, N> + PeakName,
    [E]: BatchSuperposition<T>,
{
    let mut rng = StdRng::seed_from_u64(SEED);
    let dist = Uniform::new(T::one(), T::from(100).unwrap()).unwrap();
    let mut group = c.benchmark_group(format!("{}/{dtype}/iso_work", E::NAME));
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(3));
    for (n, m) in &ISO_WORK {
        group.throughput(Throughput::Elements((n * m) as u64));
        let functions = make_functions::<T, E, N>(&mut rng, &dist, *n);
        let at = make_grid::<T>(*m);
        let reference = functions.superposition_with(&at, Strategy::FunctionsOuter);
        for (name, s) in STRATEGIES {
            assert_eq!(
                functions.superposition_with(&at, s),
                reference,
                "{name} @ {n}x{m}"
            );
        }
        let id = format!("{n}x{m}");
        for (name, strategy) in STRATEGIES {
            group.bench_with_input(BenchmarkId::new(name, &id), &strategy, |b, &s| {
                b.iter(|| black_box(&functions[..]).superposition_with(black_box(&at), s))
            });
        }
    }
    group.finish();
}

fn par_iso_work<T, E, const N: usize>(c: &mut Criterion, dtype: &str)
where
    T: Float + SampleUniform + Debug + Send + Sync,
    E: Evaluate<T> + FromArray<T, N> + PeakName + Sync,
    [E]: ParBatchSuperposition<T>,
{
    let mut rng = StdRng::seed_from_u64(SEED);
    let dist = Uniform::new(T::one(), T::from(100).unwrap()).unwrap();
    let mut group = c.benchmark_group(format!("{}/{dtype}/par_iso_work", E::NAME));
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(3));
    for (n, m) in &ISO_WORK[..1] {
        group.throughput(Throughput::Elements((n * m) as u64));
        let functions = make_functions::<T, E, N>(&mut rng, &dist, *n);
        let at = make_grid::<T>(*m);
        let reference = functions.superposition_with(&at, Strategy::FunctionsOuter);
        for (name, s) in STRATEGIES {
            assert_eq!(
                functions.par_superposition_with(&at, s),
                reference,
                "par {name} @ {n}x{m}"
            );
        }
        for t in THREADS {
            let threads = ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap();
            let id = format!("{t} - {n}x{m}");
            for (name, strategy) in &STRATEGIES[1..] {
                group.bench_with_input(BenchmarkId::new(*name, &id), &strategy, |b, &s| {
                    b.iter(|| {
                        threads.install(|| {
                            black_box(&functions[..]).par_superposition_with(black_box(&at), *s)
                        })
                    })
                });
            }
        }
    }
    group.finish();
}

fn benches(c: &mut Criterion) {
    iso_work::<f32, Lorentzian<f32>, 3>(c, "f32");
    iso_work::<f64, Lorentzian<f64>, 3>(c, "f64");
    iso_work::<f32, Gaussian<f32>, 3>(c, "f32");
    iso_work::<f64, Gaussian<f64>, 3>(c, "f64");
}

fn par_benches(c: &mut Criterion) {
    par_iso_work::<f32, Lorentzian<f32>, 3>(c, "f32");
    par_iso_work::<f64, Lorentzian<f64>, 3>(c, "f64");
    par_iso_work::<f32, Gaussian<f32>, 3>(c, "f32");
    par_iso_work::<f64, Gaussian<f64>, 3>(c, "f64");
}

criterion_group!(superposition, benches, par_benches);

criterion_main!(superposition);
