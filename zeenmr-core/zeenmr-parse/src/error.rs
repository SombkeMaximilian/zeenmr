//! Types and traits for error recording and display.

mod parse_error;
pub use parse_error::{AttachSource, Error, ParseError};

mod range;
pub use range::ByteRange;
pub(crate) use range::RangeLabel;
