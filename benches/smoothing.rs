use criterion::{Criterion, Throughput, criterion_group, criterion_main, BenchmarkId};
use num_traits::Float;
use rand::distr::Uniform;
use rand::distr::uniform::SampleUniform;
use rand::prelude::*;
use std::hint::black_box;
use std::time::Duration;
use zeenmr::deconvolution::smoothing::{MovingAverage, LeastSquares, Smooth};
use zeenmr::spectrum::intensity_array::{ArrayOwned, shape};

const SEED: u64 = 0xC0FF_EE15_600D;
const SAMPLES: usize = 50;
const TIME_SECONDS: u64 = 5;
const WORK: [usize; 2] = [1 << 16, 1 << 17];

fn make_arrays<T>(rng: &mut StdRng, dist: &Uniform<T>, len: usize) -> Vec<T>
where
    T: SampleUniform,
{
    dist.sample_iter(rng).take(len).collect()
}

fn work<T, S>(c: &mut Criterion, group_name: &str, smoothers: &[(String, S)])
where
    T: Float + SampleUniform,
    S: Smooth<T>,
{
    let mut rng = StdRng::seed_from_u64(SEED);
    let dist = Uniform::new(T::zero(), T::one()).unwrap();
    let mut group = c.benchmark_group(group_name);
    group
        .sample_size(SAMPLES)
        .measurement_time(Duration::from_secs(TIME_SECONDS));
    for n in WORK {
        group.throughput(Throughput::Elements(n as u64));
        let mut array = ArrayOwned::new(
            make_arrays(&mut rng, &dist, n).into_boxed_slice(),
            shape([n]),
        )
        .expect("hand verified");
        for (name, smoother) in smoothers {
            let id = n.to_string();
            group.bench_function(BenchmarkId::new(name, &id), |b| {
                b.iter(|| {
                    let _ = smoother.smooth_in_place(black_box(array.view_mut()));
                })
            });
        }
    }
}

fn benchmarks(c: &mut Criterion) {
    let moving_averages = [2, 3, 4]
        .iter()
        .zip(std::iter::repeat([3, 5, 7]))
        .flat_map(|(&iterations, windows)| {
            windows.map(|window| {
                (format!("{iterations}iter_{window}window"), MovingAverage::new(iterations, window))
            })
        })
        .collect::<Vec<_>>();
    let least_squares = [2, 3, 4]
        .iter()
        .zip(std::iter::repeat([3, 5, 7]))
        .zip(std::iter::repeat(2))
        .flat_map(|((&iterations, windows), order)| {
            windows.map(|window| {
                (
                    format!("{iterations}iter_{window}window_{order}order"),
                    LeastSquares::new(iterations, window, order).unwrap(),
                )
            })
        })
        .collect::<Vec<_>>();

    work::<f32, MovingAverage>(c, "MovingAverage/f32", &moving_averages);
    work::<f64, MovingAverage>(c, "MovingAverage/f64", &moving_averages);
    work::<f32, LeastSquares>(c, "LeastSquares/f32", &least_squares);
    work::<f64, LeastSquares>(c, "LeastSquares/f64", &least_squares);
}

criterion_group!(smoothing, benchmarks,);

criterion_main!(smoothing);
