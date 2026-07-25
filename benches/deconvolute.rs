use criterion::{Criterion, criterion_group, criterion_main};
use num_traits::Float;
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;
use zeenmr::{
    deconvolution::{
        fitting::ThreePoint,
        iter::ParDeconvoluteMap,
        peak_finding::CurvatureAnalysis,
        smoothing::MovingAverage,
        {Deconvolute, Deconvoluter, ParDeconvolute},
    },
    parse::{
        bruker::read_bruker_dir,
        data::{Column, Value},
    },
    peak_shape::Lorentzian,
    spectrum::{
        Spectrum1D,
        axis::{FrequencyAxis, ShiftReference},
        builder_1d::Builder1D,
        range::{FiniteBounds, FrequencyRange},
    },
};

mod workspace_dir;
use workspace_dir::workspace_dir;

fn read_spectrum<T, P>(path: P) -> Spectrum1D<T, Arc<[T]>>
where
    T: Float,
    P: AsRef<Path>,
{
    let mut bruker = read_bruker_dir(path, 10, 10).unwrap();
    let raw = match bruker.children[0].data_tables[0].remove("1r") {
        Some(Column::Integer(inner)) => inner
            .into_iter()
            .map(|x| T::from(x).unwrap())
            .collect::<Arc<[T]>>(),
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
    let start = (0.15_f64 * raw.len() as f64) as usize;
    let end = (0.85_f64 * raw.len() as f64) as usize;

    Builder1D::real(raw)
        .unwrap()
        .axis(axis)
        .signal_range(start..end)
        .unwrap()
        .finalize()
}

fn read_spectra<T, P>(path: P) -> Vec<Spectrum1D<T, Arc<[T]>>>
where
    T: Float,
    P: AsRef<Path>,
{
    path.as_ref()
        .read_dir()
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| read_spectrum(e.path()))
        .collect()
}

fn single_and_batch(c: &mut Criterion) {
    let data_dir = workspace_dir().join("data").join("bruker");

    let blood_spectra = read_spectra::<f32, _>(data_dir.join("blood"));
    let deconvoluter = Deconvoluter::new()
        .with_fitter(ThreePoint::<Lorentzian<f32>>::default())
        .with_finder(CurvatureAnalysis::default())
        .with_smoother(MovingAverage::default());

    c.bench_function("f32_deconvolute_blood_single", |b| {
        b.iter(|| {
            deconvoluter
                .deconvolute(&blood_spectra[0])
                .unwrap()
        })
    });
    c.bench_function("f32_par_deconvolute_blood_single", |b| {
        b.iter(|| {
            deconvoluter
                .par_deconvolute(&blood_spectra[0])
                .unwrap()
        })
    });
    c.bench_function("f32_deconvolute_blood_batch", |b| {
        b.iter(|| {
            blood_spectra
                .par_iter()
                .deconvolute(&deconvoluter)
                .collect::<Vec<_>>()
        })
    });

    let blood_spectra = read_spectra::<f64, _>(data_dir.join("blood"));
    let deconvoluter = Deconvoluter::new()
        .with_fitter(ThreePoint::<Lorentzian<f64>>::default())
        .with_finder(CurvatureAnalysis::default())
        .with_smoother(MovingAverage::default());

    c.bench_function("f64_deconvolute_blood_single", |b| {
        b.iter(|| {
            deconvoluter
                .deconvolute(&blood_spectra[0])
                .unwrap()
        })
    });
    c.bench_function("f64_par_deconvolute_blood_single", |b| {
        b.iter(|| {
            deconvoluter
                .par_deconvolute(&blood_spectra[0])
                .unwrap()
        })
    });
    c.bench_function("f64_deconvolute_blood_batch", |b| {
        b.iter(|| {
            blood_spectra
                .par_iter()
                .deconvolute(&deconvoluter)
                .collect::<Vec<_>>()
        })
    });
}

criterion_group! {
    name = deconvolute;
    config = Criterion::default();
    targets = single_and_batch
}

criterion_main!(deconvolute);
