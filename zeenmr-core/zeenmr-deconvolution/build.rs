use std::path::PathBuf;
use std::{env, fs};
use zeenmr_util_build::cache_rs;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let dest = PathBuf::from(env::var("OUT_DIR").unwrap()).join("cache.rs");
    fs::write(dest, cache_rs()).unwrap();
}
