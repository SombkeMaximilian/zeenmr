use criterion::{Criterion, criterion_group, criterion_main};
use metabodecon::spectrum::JcampDx;

mod workspace_dir;
use workspace_dir::workspace_dir;

fn read_spectrum(c: &mut Criterion) {
    let data_dir = workspace_dir()
        .join("data")
        .join("jcamp-dx")
        .join("test");
    let versions = ["v5", "v6"];

    versions.iter().for_each(|version| {
        data_dir
            .join(version)
            .read_dir()
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .for_each(|path| {
                let basename = path.file_stem().unwrap().to_str().unwrap();
                let bench_name = format!("read_jdx_{}_{}", version, basename);
                c.bench_function(&bench_name, |b| {
                    b.iter(|| {
                        let _spectrum = JcampDx::read_spectrum(&path, (1.0, 1.1)).unwrap();
                    })
                });
            });
    });
}

fn read_spectra(c: &mut Criterion) {
    let blood_path = workspace_dir()
        .join("data")
        .join("jcamp-dx")
        .join("blood");

    c.bench_function("read_jdx_blood_spectra", |b| {
        b.iter(|| {
            let _spectra = JcampDx::read_spectra(&blood_path, (1.0, 1.1)).unwrap();
        })
    });
}

criterion_group! {
    name = jcampdx_reader;
    config = Criterion::default().sample_size(10);
    targets = read_spectrum, read_spectra
}

criterion_main!(jcampdx_reader);
