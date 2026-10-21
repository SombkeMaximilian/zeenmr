use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use num_traits::Float;
use rayon::ThreadPoolBuilder;
use std::time::Duration;
use zeenmr::deconvolution::fitting::{
    Fit, FitModel, IterativeFit, ParFit, ParFitModel, ParIterativeFit, ThreePoint,
};
use zeenmr::deconvolution::peak_finding::Peak;
use zeenmr::deconvolution::util::noise::Noise;
use zeenmr::peak_shape::{Evaluate, Lorentzian, PeakShape};
use zeenmr::spectrum::SpectrumOwned;
use zeenmr::spectrum::axis::range::FrequencyRange;
use zeenmr::spectrum::axis::{FrequencyAxis, Larmor, ShiftReference, frequency_axes};
use zeenmr::spectrum::dimension::StaticDim;
use zeenmr::spectrum::intensity_array::{ArrayOwned, shape};

const SAMPLES: usize = 10;
const TIME_SECONDS: u64 = 3;
const THREADS: [usize; 4] = [2, 4, 8, 16];
const POINTS_PER_PEAK: usize = 15;
const SIGNAL_AMP: f64 = 1e6_f64;
const SIGNAL_HW: f64 = 2_f64;
const CENTER: f64 = ((POINTS_PER_PEAK - 1) as f64) / 2_f64;
const LEN: usize = 1 << 17;
const WORK: [usize; 7] = [1 << 6, 1 << 7, 1 << 8, 1 << 9, 1 << 10, 1 << 11, 1 << 12];
const ITERATIONS: usize = 1;

fn dummy_axis<T>() -> FrequencyAxis<T>
where
    T: Float,
{
    let range = FrequencyRange::new(
        T::zero(),
        T::from(LEN).expect("conversion from usize to T must never fail"),
    )
    .expect("hand verified");
    let larmor = Larmor::new(T::one()).expect("hand verified");
    let shift_reference = ShiftReference::new(T::zero(), T::zero()).expect("hand verified");

    FrequencyAxis::new(range, larmor, shift_reference).expect("hand verified")
}

fn make_spectrum<T>(
    count: usize,
) -> (
    SpectrumOwned<StaticDim<FrequencyAxis<T>, 1>, T, StaticDim<usize, 1>>,
    Vec<Peak>,
)
where
    T: Float,
{
    assert!(count > 0 && count * POINTS_PER_PEAK <= LEN);

    let signal_p = Lorentzian::from_untransformed(SIGNAL_AMP, SIGNAL_HW, CENTER);
    let signal_len = count * POINTS_PER_PEAK;
    let intensities = ArrayOwned::from_linear_fn(shape([LEN]), |i| {
        let raw = if i < signal_len {
            signal_p.evaluate((i % POINTS_PER_PEAK) as f64)
        } else {
            0_f64
        };

        T::from(raw).expect("conversion from f64 to T must never fail")
    })
    .expect("hand verified");
    let spectrum =
        SpectrumOwned::new(frequency_axes([dummy_axis()]), intensities).expect("hand verified");
    let peaks = (0..count)
        .map(|i| {
            let center = (i * POINTS_PER_PEAK) + (POINTS_PER_PEAK / 2);
            let left = center - (SIGNAL_HW as usize);
            let right = center + (SIGNAL_HW as usize);

            Peak {
                left,
                center,
                right,
            }
        })
        .collect();

    (spectrum, peaks)
}

fn _work<T, P, F>(c: &mut Criterion, group_name: &str, fitters: &[(String, F)])
where
    T: Float,
    P: PeakShape<T>,
    F: Fit<T, P>,
    F::Error: std::error::Error,
{
    let mut group = c.benchmark_group(group_name);
    group
        .sample_size(SAMPLES)
        .measurement_time(Duration::from_secs(TIME_SECONDS));
    for n in WORK {
        group.throughput(Throughput::Elements(n as u64));
        let (spectrum, peaks) = make_spectrum(n);
        for (name, fitter) in fitters {
            let id = format!("{n}peaks_full");
            group.bench_function(BenchmarkId::new(name, &id), |b| {
                b.iter(|| fitter.fit(spectrum.view(), spectrum.intensities(), &peaks, &[]))
            });
        }
    }
}

fn _par_work<T, P, F>(c: &mut Criterion, group_name: &str, fitters: &[(String, F)])
where
    T: Float + Send + Sync,
    P: PeakShape<T> + Send + Sync,
    F: ParFit<T, P> + Sync,
    F::Error: std::error::Error + Send,
{
    let mut group = c.benchmark_group(group_name);
    group
        .sample_size(SAMPLES)
        .measurement_time(Duration::from_secs(TIME_SECONDS));
    for n in WORK {
        group.throughput(Throughput::Elements(n as u64));
        let (spectrum, peaks) = make_spectrum(n);
        for t in THREADS {
            let threads = ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap();
            let id = format!("{n}peaks_{t}threads_full");
            for (name, fitter) in fitters {
                group.bench_with_input(BenchmarkId::new(name, &id), fitter, |b, f| {
                    b.iter(|| {
                        threads.install(|| {
                            f.par_fit(spectrum.view(), spectrum.intensities(), &peaks, &[])
                        })
                    })
                });
            }
        }
    }
}

fn iterative_work<T, P, F>(c: &mut Criterion, group_name: &str, fitters: &[(String, F)])
where
    T: Float,
    P: PeakShape<T>,
    F: IterativeFit<T, P>,
    F::Error: std::error::Error,
{
    let mut group = c.benchmark_group(group_name);
    group
        .sample_size(SAMPLES)
        .measurement_time(Duration::from_secs(TIME_SECONDS));
    for n in WORK {
        let (spectrum, peaks) = make_spectrum(n);
        for (name, fitter) in fitters {
            group.throughput(Throughput::Elements(ITERATIONS as u64));
            let id = format!("{n}peaks_model_setup");
            group.bench_function(BenchmarkId::new(name, &id), |b| {
                b.iter(|| fitter.model(spectrum.view(), spectrum.intensities(), &peaks, &[]))
            });
            let id = format!("{n}peaks_per_iteration");
            let mut model = fitter
                .model(spectrum.view(), spectrum.intensities(), &peaks, &[])
                .unwrap();
            group.bench_function(BenchmarkId::new(name, &id), |b| {
                b.iter(|| model.step_n(ITERATIONS))
            });
            assert_eq!(model.peak_shapes().len(), n);
        }
    }
}

fn par_iterative_work<T, P, F>(c: &mut Criterion, group_name: &str, fitters: &[(String, F)])
where
    T: Float + Send + Sync,
    P: PeakShape<T> + Send + Sync,
    F: ParIterativeFit<T, P> + Sync,
    F::Error: std::error::Error + Send,
    for<'s> F::ParModel<'s>: Send,
{
    let mut group = c.benchmark_group(group_name);
    group
        .sample_size(SAMPLES)
        .measurement_time(Duration::from_secs(TIME_SECONDS));
    for n in WORK {
        let (spectrum, peaks) = make_spectrum(n);
        for t in THREADS {
            group.throughput(Throughput::Elements(ITERATIONS as u64));
            let threads = ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap();
            let id = format!("{n}peaks_{t}threads_per_iteration");
            for (name, fitter) in fitters {
                let mut model = fitter
                    .par_model(spectrum.view(), spectrum.intensities(), &peaks, &[])
                    .unwrap();
                group.bench_function(BenchmarkId::new(name, &id), |b| {
                    b.iter(|| threads.install(|| model.par_step_n(ITERATIONS)))
                });
                assert_eq!(model.peak_shapes().len(), n);
            }
        }
    }
}

fn benchmarks(c: &mut Criterion) {
    let tp_f32 = [(
        "default".to_string(),
        ThreePoint::<Lorentzian<f32>, _>::new(10, Noise(0_f32)),
    )];
    let tp_f64 = [(
        "default".to_string(),
        ThreePoint::<Lorentzian<f64>, _>::new(10, Noise(0_f64)),
    )];

    iterative_work::<f32, _, _>(c, "ThreePoint/f32/ser", &tp_f32);
    iterative_work::<f64, _, _>(c, "ThreePoint/f64/ser", &tp_f64);
    par_iterative_work::<f32, _, _>(c, "ThreePoint/f32/par", &tp_f32);
    par_iterative_work::<f64, _, _>(c, "ThreePoint/f64/par", &tp_f64);
}

criterion_group!(fitting, benchmarks,);

criterion_main!(fitting);
