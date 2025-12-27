#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! Parsers for common NMR data formats.

mod location;
pub(crate) use location::{Cursor, Position};

pub mod bruker;
pub mod jcampdx;
