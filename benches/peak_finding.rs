use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use num_traits::Float;
use std::time::Duration;
use zeenmr::deconvolution::peak_finding::{CurvatureAnalysis, Find};
use zeenmr::peak_shape::{Evaluate, Lorentzian};
use zeenmr::spectrum::SpectrumOwned1D;
use zeenmr::spectrum::axis::range::{FrequencyRange, RelativeRange};
use zeenmr::spectrum::axis::{FrequencyAxis, Larmor, ShiftReference, frequency_axes};
use zeenmr::spectrum::intensity_array::{ArrayOwned, shape};

const SAMPLES: usize = 10;
const TIME_SECONDS: u64 = 3;
const MIN_POINTS_PER_PEAK: usize = 15;
const NOISE_AMP: f64 = 1e0_f64;
const FILTERED_AMP: f64 = 1e2_f64;
const SIGNAL_AMP: f64 = 1e6_f64;
const NOISE_HW: f64 = 6_f64;
const FILTERED_HW: f64 = 4_f64;
const SIGNAL_HW: f64 = 1.5_f64;
const CENTER: f64 = ((MIN_POINTS_PER_PEAK - 1) as f64) / 2_f64;
const LEN: usize = 1 << 17;
const NOISE_LEN: usize = LEN / 20;
const SIGNAL_LEN: usize = LEN - NOISE_LEN;
const NOISE_FRACTION: f64 = (NOISE_LEN as f64) / (LEN as f64);
const WORK: [usize; 3] = [1 << 10, 1 << 11, 1 << 12];

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

fn signal_boundaries<T>() -> RelativeRange<T>
where
    T: Float,
{
    RelativeRange::new(
        T::from(NOISE_FRACTION).expect("conversion from {float} to T must never fail"),
        T::one(),
    )
    .expect("bounds are in [0, 1]")
}

fn make_spectrum<T>(signal: usize, filtered: usize) -> SpectrumOwned1D<T, T>
where
    T: Float,
{
    let peak_count = signal + filtered - 1;
    assert!(peak_count > 0 && peak_count * MIN_POINTS_PER_PEAK <= SIGNAL_LEN);

    let points_per_peak = SIGNAL_LEN / peak_count;
    let filtered_end = NOISE_LEN + filtered * points_per_peak;
    let peaks_end = NOISE_LEN + peak_count * points_per_peak;

    let center = ((points_per_peak - 1) as f64) / 2_f64;
    let noise_p = Lorentzian::from_untransformed(NOISE_AMP, NOISE_HW, CENTER);
    let filtered_p = Lorentzian::from_untransformed(FILTERED_AMP, FILTERED_HW, center);
    let signal_p = Lorentzian::from_untransformed(SIGNAL_AMP, SIGNAL_HW, center);
    let tail = signal_p.evaluate((points_per_peak - 1) as f64);

    let intensities = ArrayOwned::from_linear_fn(shape([LEN]), |i| {
        let raw = if i < NOISE_LEN {
            noise_p.evaluate((i % MIN_POINTS_PER_PEAK) as f64)
        } else if i < filtered_end {
            filtered_p.evaluate(((i - NOISE_LEN) % points_per_peak) as f64)
        } else if i < peaks_end {
            signal_p.evaluate(((i - filtered_end) % points_per_peak) as f64)
        } else {
            tail
        };

        T::from(raw).expect("conversion from f64 to T must never fail")
    })
    .expect("hand verified");

    SpectrumOwned1D::new(frequency_axes([dummy_axis()]), intensities).expect("hand verified")
}

fn work<T, F>(c: &mut Criterion, group_name: &str, peak_finders: &[(String, F)])
where
    T: Float,
    F: Find<T>,
    F::Error: std::error::Error,
{
    let mut group = c.benchmark_group(group_name);
    group
        .sample_size(SAMPLES)
        .measurement_time(Duration::from_secs(TIME_SECONDS));
    for signal in WORK {
        for filtered in WORK {
            group.throughput(Throughput::Elements((signal + filtered) as u64));
            let spectrum = make_spectrum(signal, filtered);
            for (name, finder) in peak_finders {
                let id = format!("{signal}signals_{filtered}filtered");
                group.bench_function(BenchmarkId::new(name, &id), |b| {
                    b.iter(|| {
                        let _ = finder.find(spectrum.view(), spectrum.intensities(), &[]);
                    })
                });
            }
        }
    }
}

fn benchmarks(c: &mut Criterion) {
    let curvature_analysis_f32 = [(
        "default_filter".to_string(),
        CurvatureAnalysis::<f32, _>::new(Some(1000.0), signal_boundaries::<f32>()),
    )];
    let curvature_analysis_f64 = [(
        "default_filter".to_string(),
        CurvatureAnalysis::<f64, _>::new(Some(1000.0), signal_boundaries::<f64>()),
    )];

    work::<f32, _>(c, "CurvatureAnalysis/f32", &curvature_analysis_f32);
    work::<f64, _>(c, "CurvatureAnalysis/f64", &curvature_analysis_f64);
}

criterion_group!(peak_finding, benchmarks,);

criterion_main!(peak_finding);
