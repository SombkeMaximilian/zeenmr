//! Types and traits for error recording and display.

mod display;
pub use display::{Annotations, ErrorDisplay, ParseError};

mod range;
pub use range::ByteRange;
