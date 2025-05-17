//! The Metabodecon alignment algorithm.
//!
//! # The Algorithm
//!
//! The alignment process is divided into 3 general steps:
//!
//! 1. Converting [`Deconvolutions`] into feature maps
//! 2. Filtering the feature maps to eliminate irrelevant feature assignments
//! 3. Solving the assignment problem
//!
//! [`Deconvolutions`]: crate::deconvolution::Deconvolution
//!
//! ## Feature Maps
//!
//! A feature is a deconvoluted signal in a spectrum. The possible assignments
//! of features to other features in different spectra are represented as a
//! feature map. A feature map can be thought of as a sparse matrix, where each
//! entry represents the similarity of two features in different spectra. These
//! feature maps are created based on a strategy, e.g., by selecting a reference
//! spectrum and generating all feature maps between it and the other spectra.
//!
//! ## Filtering
//!
//! Aligning NMR spectra allows some additional constraints to reduce the
//! computational complexity of the assignment problem. One such constraint is
//! that the centers of peaks representing the same signal in different spectra
//! can be assumed to be within a certain chemical shift distance of each other.
//! We can thereby filter out any assignments that do not meet this criterion.
//!
//! ## Solving the Assignment Problem
//!
//! Lastly, the assignment problem is solved to maximize the total similarity
//! of the assignments. Currently, only an ILP-based approach is available,
//! using the [HiGHS] solver.
//!
//! [HiGHS]: https://highs.dev/
//!
//! # Example: Aligning a set of spectra
//!
//! ```
//! use metabodecon::alignment::Aligner;
//! use metabodecon::deconvolution::Deconvoluter;
//! use metabodecon::spectrum::Bruker;
//!
//! # fn main() -> metabodecon::Result<()> {
//! // Read all spectra from Bruker TopSpin format directories within the root.
//! let path = "path/to/root";
//! # let path = "../data/bruker/blood";
//! let spectra = Bruker::read_spectra(
//!     path,
//!     // Experiment number
//!     10,
//!     // Processing number
//!     10,
//!     // Signal boundaries
//!     (-2.2, 11.8),
//! )?;
//!
//! // Deconvolute the spectra in parallel.
//! let deconvoluter = Deconvoluter::default();
//! let deconvolutions = deconvoluter.par_deconvolute_spectra(&spectra)?;
//!
//! // Align the deconvolutions using the default settings.
//! let aligner = Aligner::default();
//! let alignment = aligner.align_deconvolutions(&deconvolutions);
//! # Ok(())
//! # }
//! ```
//!
//! # Example: Configuring the [`Aligner`]
//!
//! [`Aligner`] can be configured with different strategies, filtering settings,
//! and solving methods independently, though currently only a few options are
//! available for the filtering settings, while alignment strategy and solving
//! method have one option each.
//!
//! ```
//! use metabodecon::alignment::{
//!     Aligner, AlignmentStrategy, FilteringSettings, SimilarityMetric, SolvingSettings,
//! };
//!
//! # fn main() -> metabodecon::Result<()> {
//! let aligner = Aligner::new(
//!     AlignmentStrategy::Pairwise,
//!     FilteringSettings::DistanceSimilarity {
//!         similarity_metric: SimilarityMetric::ShapeDistance,
//!         max_distance: 0.035,
//!         min_similarity: 0.6,
//!     },
//!     SolvingSettings::LinearProgramming,
//! )?;
//! # Ok(())
//! # }
//! ```

mod aligner;
pub use aligner::Aligner;

mod alignment;
pub use alignment::Alignment;

mod assignment;
pub use assignment::{FilteringSettings, SimilarityMetric};

mod feature;
pub use feature::AlignmentStrategy;

#[cfg(feature = "serde")]
mod serialized_alignment;
#[cfg(feature = "serde")]
pub(crate) use serialized_alignment::SerializedAlignment;

mod solving;
pub use solving::SolvingSettings;

pub mod error;
