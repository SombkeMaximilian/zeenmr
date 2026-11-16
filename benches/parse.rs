use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::fs;
use std::time::Duration;
use zeenmr::parse::bruker::{read_bruker_dir, read_bruker_exp, read_bruker_proc};
use zeenmr::parse::jcampdx::parse_jcampdx;

mod workspace_dir;
use workspace_dir::workspace_dir;

const SAMPLES: usize = 10;
const TIME_SECONDS: u64 = 3;

const JCAMP_DX_VERSIONS: [&'static str; 2] = ["v5", "v6"];
const JCAMP_DX_FORMATS: [&'static str; 2] = ["ntuples", "xydata"];
const JCAMP_DX_ENCODINGS: [&'static str; 4] = ["affn", "pac", "sqz", "difdup"];

const BRUKER_EXP: u32 = 10;
const BRUKER_PROC: u32 = 10;

fn jcampdx_work(c: &mut Criterion) {
    let group_name = "jcamp-dx";
    let mut group = c.benchmark_group(group_name);
    group
        .sample_size(SAMPLES)
        .measurement_time(Duration::from_secs(TIME_SECONDS));
    group.throughput(Throughput::Elements(1));
    let data_dir = workspace_dir()
        .join("data")
        .join("jcamp-dx")
        .join("test");
    let combinations = JCAMP_DX_VERSIONS
        .iter()
        .flat_map(|version| {
            JCAMP_DX_FORMATS
                .iter()
                .map(move |format| (version, format))
        })
        .flat_map(|(version, format)| {
            JCAMP_DX_ENCODINGS
                .iter()
                .map(move |encoding| (version, format, encoding))
        });
    for (version, format, encoding) in combinations {
        let id = format!("{version}/{format}/{encoding}");
        let file_path = data_dir
            .join(version)
            .join(format!("{format}_{encoding}.dx"));
        let content = match fs::read_to_string(&file_path) {
            Ok(content) => content,
            Err(e) => {
                eprintln!("failed to read jcamp-dx/{id} at {:?}: {e}", &file_path);
                continue;
            }
        };
        match parse_jcampdx(&content) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("failed to parse jcamp-dx/{id} at {:?}: {e}", &file_path);
                continue;
            }
        };
        group.bench_function(id, |b| b.iter(|| parse_jcampdx(&content)));
    }
}

fn bruker_work(c: &mut Criterion) {
    let group_name = "bruker";
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
    let exp_dir = data_dir.join(BRUKER_EXP.to_string());
    let proc_dir = data_dir
        .join(BRUKER_EXP.to_string())
        .join("pdata")
        .join(BRUKER_PROC.to_string());
    match read_bruker_dir(&data_dir, BRUKER_EXP, BRUKER_PROC) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("failed to parse bruker at {:?}: {e}", &data_dir);
            return;
        }
    }
    group.bench_function("fid_only_1d", |b| b.iter(|| read_bruker_exp(&exp_dir)));
    group.bench_function("real_spectrum_only_1d", |b| {
        b.iter(|| read_bruker_proc(&proc_dir, true))
    });
    group.bench_function("complex_spectrum_only_1d", |b| {
        b.iter(|| read_bruker_proc(&proc_dir, false))
    });
    group.bench_function("full_1d", |b| {
        b.iter(|| read_bruker_dir(&data_dir, BRUKER_EXP, BRUKER_PROC))
    });
}

criterion_group!(parse, jcampdx_work, bruker_work,);

criterion_main!(parse);
