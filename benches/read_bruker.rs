use criterion::{Criterion, criterion_group, criterion_main};
use metabodecon::spectrum::Bruker;

mod workspace_dir;
use workspace_dir::workspace_dir;

fn read_spectrum(c: &mut Criterion) {
    let data_dir = workspace_dir().join("data").join("bruker");
    let sim_path = data_dir.join("sim").join("sim_01");
    let blood_path = data_dir.join("blood").join("blood_01");

    c.bench_function("read_bruker_sim_spectrum", |b| {
        b.iter(|| {
            let _spectrum = Bruker::read_spectrum(&sim_path, 10, 10, (3.35, 3.45)).unwrap();
        })
    });
    c.bench_function("read_bruker_blood_spectrum", |b| {
        b.iter(|| {
            let _spectrum = Bruker::read_spectrum(&blood_path, 10, 10, (0.0, 0.1)).unwrap();
        })
    });
}

fn read_spectra(c: &mut Criterion) {
    let data_dir = workspace_dir().join("data").join("bruker");
    let sim_path = data_dir.join("sim");
    let blood_path = data_dir.join("blood");

    c.bench_function("read_bruker_sim_spectra", |b| {
        b.iter(|| {
            let _spectra = Bruker::read_spectra(&sim_path, 10, 10, (3.35, 3.45)).unwrap();
        })
    });
    c.bench_function("read_bruker_blood_spectra", |b| {
        b.iter(|| {
            let _spectra = Bruker::read_spectra(&blood_path, 10, 10, (0.0, 0.1)).unwrap();
        })
    });
}

criterion_group! {
    name = bruker_reader;
    config = Criterion::default().sample_size(10);
    targets = read_spectrum, read_spectra
}

criterion_main!(bruker_reader);
