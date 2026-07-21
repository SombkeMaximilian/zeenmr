use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use num_traits::Float;
use rand::distr::Uniform;
use rand::distr::uniform::SampleUniform;
use rand::prelude::*;
use rayon::ThreadPoolBuilder;
use std::hint::black_box;
use std::time::Duration;
use zeenmr::peak_shape::batch_superposition::{
    FuseWidth, Fused, ParSuperpositionKernel, Standard, Strategy, SuperpositionKernel,
};
use zeenmr::peak_shape::{FromArray, Gaussian, Lorentzian};

const SEED: u64 = 0xC0FF_EE15_600D;
const SAMPLES: usize = 10;
const TIME_SECONDS: u64 = 3;
const THREADS: [usize; 4] = [2, 4, 8, 16];
const STRATEGIES: [(&str, Strategy); 3] = [
    ("auto", Strategy::Auto),
    ("subvectors", Strategy::Subvectors { p: 1024 }),
    ("functions_outer", Strategy::FunctionsOuter),
];
const FUSE_WIDTHS: [FuseWidth; 4] = [
    FuseWidth::One,
    FuseWidth::Two,
    FuseWidth::Four,
    FuseWidth::Eight,
];
const ISO_WORK: [(usize, usize); 5] = [
    (1 << 10, 1 << 17),
    (1 << 12, 1 << 15),
    (1 << 14, 1 << 13),
    (1 << 15, 1 << 12),
    (1 << 16, 1 << 11),
];

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

fn iso_work<T, E, K, const N: usize>(c: &mut Criterion, group_name: &str, kernels: &[(String, K)])
where
    T: Float + SampleUniform,
    E: FromArray<T, N>,
    K: SuperpositionKernel<T, E>,
{
    let mut rng = StdRng::seed_from_u64(SEED);
    let dist = Uniform::new(T::one(), T::from(100).unwrap()).unwrap();
    let mut group = c.benchmark_group(group_name);
    group
        .sample_size(SAMPLES)
        .measurement_time(Duration::from_secs(TIME_SECONDS));
    for (n, m) in &ISO_WORK {
        group.throughput(Throughput::Elements((n * m) as u64));
        let functions = make_functions::<T, E, N>(&mut rng, &dist, *n);
        let at = make_grid::<T>(*m);
        let id = format!("{n}x{m}");
        for (name, kernel) in kernels {
            group.bench_with_input(BenchmarkId::new(name, &id), kernel, |b, k| {
                b.iter(|| k.superposition(black_box(&functions), black_box(&at)))
            });
        }
    }
    group.finish();
}

fn par_iso_work<T, E, K, const N: usize>(
    c: &mut Criterion,
    group_name: &str,
    kernels: &[(String, K)],
) where
    T: Float + SampleUniform + Send + Sync,
    E: FromArray<T, N> + Sync,
    K: ParSuperpositionKernel<T, E> + Send + Sync,
{
    let mut rng = StdRng::seed_from_u64(SEED);
    let dist = Uniform::new(T::one(), T::from(100).unwrap()).unwrap();
    let mut group = c.benchmark_group(group_name);
    group
        .sample_size(SAMPLES)
        .measurement_time(Duration::from_secs(TIME_SECONDS));
    for (n, m) in &ISO_WORK[..1] {
        group.throughput(Throughput::Elements((n * m) as u64));
        let functions = make_functions::<T, E, N>(&mut rng, &dist, *n);
        let at = make_grid::<T>(*m);
        for t in &THREADS {
            let threads = ThreadPoolBuilder::new()
                .num_threads(*t)
                .build()
                .unwrap();
            let id = format!("{n}x{m}_{t}threads");
            for (name, kernel) in kernels {
                group.bench_with_input(BenchmarkId::new(name, &id), kernel, |b, k| {
                    b.iter(|| {
                        threads
                            .install(|| k.par_superposition(black_box(&functions), black_box(&at)))
                    })
                });
            }
        }
    }
    group.finish();
}

fn standard(c: &mut Criterion) {
    let kernels = STRATEGIES
        .iter()
        .map(|(n, s)| (n.to_string(), Standard::new().with_strategy(*s)))
        .collect::<Vec<(String, Standard)>>();

    iso_work::<f32, Lorentzian<f32>, _, 3>(c, "Lorentzian/f32/ser_standard", &kernels);
    iso_work::<f64, Lorentzian<f64>, _, 3>(c, "Lorentzian/f64/ser_standard", &kernels);
    iso_work::<f32, Gaussian<f32>, _, 3>(c, "Gaussian/f32/ser_standard", &kernels);
    iso_work::<f64, Gaussian<f64>, _, 3>(c, "Gaussian/f64/ser_standard", &kernels);

    par_iso_work::<f32, Lorentzian<f32>, _, 3>(c, "Lorentzian/f32/par_standard", &kernels);
    par_iso_work::<f64, Lorentzian<f64>, _, 3>(c, "Lorentzian/f64/par_standard", &kernels);
    par_iso_work::<f32, Gaussian<f32>, _, 3>(c, "Gaussian/f32/par_standard", &kernels);
    par_iso_work::<f64, Gaussian<f64>, _, 3>(c, "Gaussian/f64/par_standard", &kernels);
}

fn fused(c: &mut Criterion) {
    let kernels = FUSE_WIDTHS
        .iter()
        .map(|&w| ((w as u8).to_string(), Fused::new().with_width(w)))
        .collect::<Vec<(String, Fused)>>();

    iso_work::<f32, Lorentzian<f32>, _, 3>(c, "Lorentzian/f32/ser_fused", &kernels);
    iso_work::<f64, Lorentzian<f64>, _, 3>(c, "Lorentzian/f64/ser_fused", &kernels);

    par_iso_work::<f32, Lorentzian<f32>, _, 3>(c, "Lorentzian/f32/par_fused", &kernels);
    par_iso_work::<f64, Lorentzian<f64>, _, 3>(c, "Lorentzian/f64/par_fused", &kernels);
}

criterion_group!(superposition, standard, fused,);

criterion_main!(superposition);
