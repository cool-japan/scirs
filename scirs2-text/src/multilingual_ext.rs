//! Cross-lingual NLP and multilingual utilities (v0.3.0)
//!
//! This module re-exports the Script-based multilingual API from related modules.
//! It exists so that callers can import either from `multilingual` or `multilingual_ext`
//! without ambiguity.

pub use crate::multilingual::ScriptFamily;
pub use crate::tokenization::unicode_normalizer::{NormForm, Script};
pub use crate::transliteration::strip_diacritics;
