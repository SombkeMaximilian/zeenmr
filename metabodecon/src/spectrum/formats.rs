#[cfg(any(feature = "bruker", feature = "jdx"))]
mod extract_capture;
#[cfg(any(feature = "bruker", feature = "jdx"))]
pub(crate) use extract_capture::{extract_capture, extract_row};

#[cfg(feature = "bruker")]
mod bruker;
#[cfg(feature = "bruker")]
pub use bruker::Bruker;

#[cfg(feature = "jdx")]
mod jcampdx;
#[cfg(feature = "jdx")]
pub use jcampdx::JcampDx;
