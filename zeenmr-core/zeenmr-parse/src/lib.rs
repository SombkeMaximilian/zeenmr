#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! Parsers for common NMR data formats.

pub(crate) const CHECK_PRECISION: f64 = 1.0e+3 * f64::EPSILON;

mod location;
pub(crate) use location::{Cursor, Location, Position, UpdateCursor};

mod stack;
pub(crate) use stack::Stack;

pub mod data;

pub mod bruker;
pub mod jcampdx;
