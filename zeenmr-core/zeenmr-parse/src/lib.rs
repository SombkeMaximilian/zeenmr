#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! Parsers for common NMR data formats.

mod location;
pub(crate) use location::{Cursor, Location, Position, UpdateCursor};

pub mod bruker;
pub mod jcampdx;
