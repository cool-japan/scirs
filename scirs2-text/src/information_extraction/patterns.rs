//! Regex patterns for information extraction
//!
//! This module contains predefined regex patterns for extracting
//! common entities and information from text.

#![allow(missing_docs)]

use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    // Common regex patterns for information extraction
    pub static ref EMAIL_PATTERN: Regex = Regex::new(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ).unwrap();

    pub static ref URL_PATTERN: Regex = Regex::new(
        r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)"
    ).unwrap();

    pub static ref PHONE_PATTERN: Regex = Regex::new(
        r"(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})"
    ).unwrap();

    pub static ref DATE_PATTERN: Regex = Regex::new(
        r"\b(?:(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])[/-](?:19|20)?\d{2})|(?:(?:19|20)\d{2}[/-](?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01]))|(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})|(?:\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b"
    ).unwrap();

    pub static ref TIME_PATTERN: Regex = Regex::new(
        r"\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?(?:\s*[aApP][mM])?\b"
    ).unwrap();

    pub static ref MONEY_PATTERN: Regex = Regex::new(
        r"[$€£¥]\s*\d+(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:,\d{3})*(?:\.\d{1,2})?\s*(?:dollars?|euros?|pounds?|yen)"
    ).unwrap();

    pub static ref PERCENTAGE_PATTERN: Regex = Regex::new(
        r"\b\d+(?:\.\d+)?%\b"
    ).unwrap();
}
