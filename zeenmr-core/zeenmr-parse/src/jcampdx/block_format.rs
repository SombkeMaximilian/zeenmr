pub mod error;

mod format;
pub(crate) use format::{BlockFormat, BlockFormatBuilder, ExitStatus, LineLayout};

mod lexer;
pub(crate) use lexer::FormatToken;

mod parser;
pub(crate) use parser::FormatParser;
