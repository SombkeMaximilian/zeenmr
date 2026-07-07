//! Bruker NMR data parsing.
pub mod error;

mod read;
pub use read::{BrukerDataset, read_bruker_dir, read_bruker_exp, read_bruker_proc};
