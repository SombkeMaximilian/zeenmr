use criterion::{Criterion, criterion_group, criterion_main};
use metabodecon::deconvolution::*;
use metabodecon::spectrum::*;

mod workspace_dir;
use workspace_dir::workspace_dir;

fn single_spectrum(c: &mut Criterion) {
    let data_dir = workspace_dir().join("data").join("bruker");
    let sim_path = data_dir.join("sim").join("sim_01");
    let blood_path = data_dir.join("blood").join("blood_01");
    let sim_spectrum = Bruker::read_spectrum(sim_path, 10, 10, (3.34, 3.56)).unwrap();
    let blood_spectrum = Bruker::read_spectrum(blood_path, 10, 10, (-2.2, 11.8)).unwrap();
    let deconvoluter = Deconvoluter::default();

    // Sequential
    c.bench_function("deconvolute_sim_spectrum", |b| {
        b.iter(|| deconvoluter.deconvolute_spectrum(&sim_spectrum))
    });
    c.bench_function("deconvolute_blood_spectrum", |b| {
        b.iter(|| deconvoluter.deconvolute_spectrum(&blood_spectrum))
    });

    // Parallel
    c.bench_function("parallel_deconvolute_sim_spectrum", |b| {
        b.iter(|| deconvoluter.par_deconvolute_spectrum(&sim_spectrum))
    });
    c.bench_function("parallel_deconvolute_blood_spectrum", |b| {
        b.iter(|| deconvoluter.par_deconvolute_spectrum(&blood_spectrum))
    });
}

fn multiple_spectra(c: &mut Criterion) {
    let data_dir = workspace_dir().join("data").join("bruker");
    let sim_path = data_dir.join("sim");
    let blood_path = data_dir.join("blood");
    let sim_spectra = Bruker::read_spectra(sim_path, 10, 10, (3.34, 3.56)).unwrap();
    let blood_spectra = Bruker::read_spectra(blood_path, 10, 10, (-2.2, 11.8)).unwrap();
    let deconvoluter = Deconvoluter::default();

    // Only the parallel case is interesting here
    c.bench_function("parallel_deconvolute_sim_spectra", |b| {
        b.iter(|| deconvoluter.par_deconvolute_spectra(&sim_spectra))
    });
    c.bench_function("parallel_deconvolute_blood_spectra", |b| {
        b.iter(|| deconvoluter.par_deconvolute_spectra(&blood_spectra))
    });
}

criterion_group! {
    name = deconvoluter;
    config = Criterion::default().sample_size(50);
    targets = single_spectrum, multiple_spectra
}

criterion_main!(deconvoluter);
