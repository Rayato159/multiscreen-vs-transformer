//! Backward-compatible alias for the Multiscreen model module.
//!
//! The actual implementation lives in `multiscreen.rs`. Keep this shim so old
//! callers using `crate::model::*` do not explode after the refactor.

pub use crate::multiscreen::*;
