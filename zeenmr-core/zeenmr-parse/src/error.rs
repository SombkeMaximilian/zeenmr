//! Types and traits for error recording and display.

mod display;
pub use display::{Annotations, ErrorDisplay, ParseError};

mod position;
pub use position::Position;
pub(crate) use position::{CurrentPosition, LineCounter, UpdateLineCounter};
