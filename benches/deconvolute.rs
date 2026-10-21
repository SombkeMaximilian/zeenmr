use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use num_traits::Float;
use rayon::prelude::*;
use std::path::Path;
use std::time::Duration;
use rayon::ThreadPoolBuilder;
use zeenmr::deconvolution::fitting::ThreePoint;
use zeenmr::deconvolution::iter::ParDeconvoluteMap;
use zeenmr::deconvolution::peak_finding::CurvatureAnalysis;
use zeenmr::deconvolution::smoothing::MovingAverage;
use zeenmr::deconvolution::{Deconvolute, Deconvoluter, ParDeconvolute};
use zeenmr::parse::bruker::read_bruker_dir;
use zeenmr::parse::data::{Column, Value};
use zeenmr::peak_shape::{Lorentzian, PeakShape};
use zeenmr::spectrum::Spectrum1D;
use zeenmr::spectrum::axis::range::{FiniteBounds, FrequencyRange, ShiftRange};
use zeenmr::spectrum::axis::{FrequencyAxis, Larmor, ShiftReference, frequency_axes};
use zeenmr::spectrum::intensity_array::{Array1D, shape};

mod workspace_dir;
use workspace_dir::workspace_dir;

const SAMPLES: usize = 10;
const TIME_SECONDS: u64 = 3;
const BATCH: usize = 256;
const BATCH_TIME_SECONDS: u64 = 10;
const THREADS: [usize; 4] = [2, 4, 8, 16];

fn read_spectrum<T, P>(path: P) -> Spectrum1D<T, Box<[T]>>
where
    T: Float,
    P: AsRef<Path>,
{
    let mut bruker = read_bruker_dir(path, 10, 10).unwrap();
    let raw = match bruker.children[0].data_tables[0].remove("1r") {
        Some(Column::Integer(inner)) => inner
            .into_iter()
            .map(|x| T::from(x).unwrap())
            .collect::<Box<[T]>>(),
        _ => unreachable!(),
    };
    let freq_start = bruker
        .parameters
        .get("SW_h")
        .and_then(Value::as_f64)
        .and_then(T::from)
        .unwrap();
    let freq_range = FrequencyRange::new(freq_start, T::zero()).unwrap();
    let larmor = bruker
        .parameters
        .get("SFO1")
        .and_then(Value::as_f64)
        .and_then(T::from)
        .and_then(Larmor::new)
        .unwrap();
    let ref_freq = freq_range.start();
    let ref_shift = bruker.children[0]
        .parameters
        .get("OFFSET")
        .and_then(Value::as_f64)
        .and_then(T::from)
        .unwrap();
    let reference = ShiftReference::new(ref_shift, ref_freq).unwrap();
    let axis = FrequencyAxis::new(freq_range, larmor, reference).unwrap();
    let len = raw.len();

    Spectrum1D::new(
        frequency_axes([axis]),
        Array1D::new(raw, shape([len])).expect("non-empty 1D array"),
    )
    .expect("should be valid")
}

fn work<T, P, D>(c: &mut Criterion, group_name: &str, deconvoluters: &[(String, D)])
where
    T: Float,
    P: PeakShape<T>,
    D: Deconvolute<T, P>,
    D::Error: std::error::Error,
{
    let mut group = c.benchmark_group(group_name);
    group
        .sample_size(SAMPLES)
        .measurement_time(Duration::from_secs(TIME_SECONDS));
    group.throughput(Throughput::Elements(1));
    let data_dir = workspace_dir()
        .join("data")
        .join("bruker")
        .join("blood")
        .join("blood_01");
    let spectrum = read_spectrum::<T, _>(data_dir);
    let id = "single";
    for (name, deconvoluter) in deconvoluters {
        group.bench_function(BenchmarkId::new(name, id), |b| {
            b.iter(|| deconvoluter.deconvolute(spectrum.view()))
        });
    }
}

fn par_work<T, P, D>(c: &mut Criterion, group_name: &str, deconvoluters: &[(String, D)])
where
    T: Float + Send + Sync,
    P: PeakShape<T> + Send + Sync,
    D: ParDeconvolute<T, P> + Send + Sync,
    D::Error: std::error::Error + Send,
{
    let mut group = c.benchmark_group(group_name);
    group
        .sample_size(SAMPLES);
    let data_dir = workspace_dir()
        .join("data")
        .join("bruker")
        .join("blood")
        .join("blood_01");
    let spectrum = read_spectrum::<T, _>(data_dir);
    for (name, deconvoluter) in deconvoluters {
        for t in THREADS {
            let threads = ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap();

            group.measurement_time(Duration::from_secs(TIME_SECONDS));
            group.throughput(Throughput::Elements(1));
            let id = format!("{t}threads_single");
            group.bench_function(BenchmarkId::new(name, id), |b| {
                b.iter(|| threads.install(|| deconvoluter.par_deconvolute(spectrum.view())))
            });

            group.measurement_time(Duration::from_secs(BATCH_TIME_SECONDS));
            group.throughput(Throughput::Elements(BATCH as u64));
            let id = format!("{t}threads_batch");
            group.bench_function(BenchmarkId::new(name, id), |b| {
                b.iter(|| {
                    threads.install(|| {
                        rayon::iter::repeat_n(spectrum.view(), BATCH)
                            .deconvolute(deconvoluter)
                            .collect::<Vec<_>>()
                    })
                })
            });
        }
    }
}

fn benchmarks(c: &mut Criterion) {
    let default_f32 = [(
        "default".to_string(),
        Deconvoluter::new()
            .with_fitter(ThreePoint::<Lorentzian<f32>, _>::default())
            .with_finder(CurvatureAnalysis::default())
            .with_smoother(MovingAverage::default())
            .ignore(ShiftRange::new(4.7, 4.9).unwrap()),
    )];
    let default_f64 = [(
        "default".to_string(),
        Deconvoluter::new()
            .with_fitter(ThreePoint::<Lorentzian<f64>, _>::default())
            .with_finder(CurvatureAnalysis::default())
            .with_smoother(MovingAverage::default())
            .ignore(ShiftRange::new(4.7, 4.9).unwrap()),
    )];

    work(c, "DefaultDeconvoluter/f32/ser", &default_f32);
    work(c, "DefaultDeconvoluter/f64/ser", &default_f64);
    par_work(c, "DefaultDeconvoluter/f32/par", &default_f32);
    par_work(c, "DefaultDeconvoluter/f64/par", &default_f64);
}

criterion_group!(deconvolute, benchmarks,);

criterion_main!(deconvolute);
