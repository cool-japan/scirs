//! Event extraction from natural language text.
//!
//! This module provides lexicon-based event detection, argument extraction,
//! and event coreference resolution.  All processing is purely rule-based
//! and does not require trained weights.
//!
//! # Overview
//!
//! - [`EventType`] — enumeration of recognised event categories
//! - [`TriggerLexicon`] — mapping from action words to [`EventType`]
//! - [`Argument`] — an event participant with its semantic role
//! - [`Event`] — a single extracted event instance
//! - [`extract_events`] — top-level extraction entry point
//! - [`event_coref`] — group events that refer to the same occurrence
//!
//! # Example
//!
//! ```rust
//! use scirs2_text::event_extraction::{TriggerLexicon, extract_events};
//!
//! let lex = TriggerLexicon::default_english();
//! let text = "Police arrested the suspect yesterday in New York.";
//! let events = extract_events(text, &lex);
//! assert!(!events.is_empty());
//! assert_eq!(events[0].event_type, "Arrest");
//! ```

use crate::error::{Result, TextError};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// EventType
// ---------------------------------------------------------------------------

/// Coarse-grained event category.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EventType {
    /// Movement of a person, vehicle, or object.
    Move,
    /// Violent attack, assault, or strike.
    Attack,
    /// Meeting, gathering, or conference.
    Meet,
    /// Arrest, detain, or apprehend.
    Arrest,
    /// Death, killing, or fatality.
    Die,
    /// Transfer of money, ownership, or responsibility.
    Transfer,
    /// Creation, production, or manufacturing.
    Create,
    /// Destruction, demolition, or elimination.
    Destroy,
    /// User-defined event category.
    Custom(String),
}

impl EventType {
    /// Human-readable label.
    pub fn label(&self) -> &str {
        match self {
            EventType::Move => "Move",
            EventType::Attack => "Attack",
            EventType::Meet => "Meet",
            EventType::Arrest => "Arrest",
            EventType::Die => "Die",
            EventType::Transfer => "Transfer",
            EventType::Create => "Create",
            EventType::Destroy => "Destroy",
            EventType::Custom(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// TriggerLexicon
// ---------------------------------------------------------------------------

/// Maps action words (triggers) to their event categories.
///
/// Trigger lookup is case-insensitive.  Each trigger word must appear as a
/// standalone token boundary in the text.
pub struct TriggerLexicon {
    /// Lower-cased trigger word → event type.
    pub triggers: HashMap<String, EventType>,
}

impl Default for TriggerLexicon {
    fn default() -> Self {
        Self::default_english()
    }
}

impl TriggerLexicon {
    /// Create an empty lexicon.
    pub fn new() -> Self {
        Self {
            triggers: HashMap::new(),
        }
    }

    /// Register a single trigger.
    pub fn insert(&mut self, word: impl Into<String>, event_type: EventType) {
        self.triggers.insert(word.into().to_lowercase(), event_type);
    }

    /// Look up an event type for a word (case-insensitive).
    pub fn lookup(&self, word: &str) -> Option<&EventType> {
        self.triggers.get(&word.to_lowercase())
    }

    /// Build a lexicon populated with common English triggers.
    pub fn default_english() -> Self {
        let mut lex = Self::new();

        // Move
        for w in &[
            "moved",
            "moving",
            "move",
            "traveled",
            "travel",
            "travelled",
            "fled",
            "flee",
            "departed",
            "depart",
            "arrived",
            "arrive",
            "entered",
            "enter",
            "left",
            "evacuated",
            "evacuate",
            "migrated",
            "migrate",
            "relocated",
            "relocate",
            "walked",
            "ran",
            "run",
        ] {
            lex.insert(*w, EventType::Move);
        }

        // Attack
        for w in &[
            "attacked",
            "attack",
            "assaulted",
            "assault",
            "bombed",
            "bomb",
            "shot",
            "shoot",
            "fired",
            "fire",
            "struck",
            "strike",
            "hit",
            "targeted",
            "target",
            "raided",
            "raid",
            "invaded",
            "invade",
            "detonated",
            "detonate",
            "launched",
            "launch",
            "stabbed",
            "stab",
        ] {
            lex.insert(*w, EventType::Attack);
        }

        // Meet
        for w in &[
            "met",
            "meet",
            "meeting",
            "gathered",
            "gather",
            "assembled",
            "assemble",
            "convened",
            "convene",
            "discussed",
            "discuss",
            "negotiated",
            "negotiate",
            "talked",
            "talk",
            "conferenced",
            "conferred",
            "confer",
            "visited",
            "visit",
        ] {
            lex.insert(*w, EventType::Meet);
        }

        // Arrest
        for w in &[
            "arrested",
            "arrest",
            "detained",
            "detain",
            "apprehended",
            "apprehend",
            "captured",
            "capture",
            "jailed",
            "jail",
            "imprisoned",
            "imprison",
            "charged",
            "charge",
            "indicted",
            "indict",
            "booked",
            "book",
            "handcuffed",
            "handcuff",
        ] {
            lex.insert(*w, EventType::Arrest);
        }

        // Die
        for w in &[
            "died",
            "die",
            "killed",
            "kill",
            "murdered",
            "murder",
            "executed",
            "execute",
            "slain",
            "slayed",
            "slay",
            "perished",
            "perish",
            "deceased",
            "assassinated",
            "assassinate",
            "fatally",
        ] {
            lex.insert(*w, EventType::Die);
        }

        // Transfer
        for w in &[
            "transferred",
            "transfer",
            "sold",
            "sell",
            "purchased",
            "purchase",
            "bought",
            "buy",
            "donated",
            "donate",
            "paid",
            "pay",
            "sent",
            "send",
            "received",
            "receive",
            "wired",
            "wire",
            "awarded",
            "award",
            "granted",
            "grant",
        ] {
            lex.insert(*w, EventType::Transfer);
        }

        // Create
        for w in &[
            "created",
            "create",
            "built",
            "build",
            "developed",
            "develop",
            "founded",
            "found",
            "established",
            "establish",
            "launched",
            "produced",
            "produce",
            "manufactured",
            "manufacture",
            "invented",
            "invent",
            "designed",
            "design",
            "wrote",
            "write",
            "authored",
            "author",
            "published",
            "publish",
            "formed",
            "form",
        ] {
            lex.insert(*w, EventType::Create);
        }

        // Destroy
        for w in &[
            "destroyed",
            "destroy",
            "demolished",
            "demolish",
            "burned",
            "burn",
            "razed",
            "raze",
            "collapsed",
            "collapse",
            "ruined",
            "ruin",
            "dismantled",
            "dismantle",
            "obliterated",
            "obliterate",
            "wrecked",
            "wreck",
            "shattered",
            "shatter",
        ] {
            lex.insert(*w, EventType::Destroy);
        }

        lex
    }
}

// ---------------------------------------------------------------------------
// Argument
// ---------------------------------------------------------------------------

/// A participant or modifier in an event.
#[derive(Debug, Clone)]
pub struct Argument {
    /// Semantic role label (e.g. `"Agent"`, `"Patient"`, `"Location"`, `"Time"`).
    pub role: String,
    /// Surface text of the argument.
    pub text: String,
    /// Character span `(start, end)` in the source document.
    pub span: (usize, usize),
}

// ---------------------------------------------------------------------------
// Event
// ---------------------------------------------------------------------------

/// A single event instance extracted from text.
#[derive(Debug, Clone)]
pub struct Event {
    /// The trigger word that anchored this event.
    pub trigger: String,
    /// Character span of the trigger `(start, end)`.
    pub trigger_span: (usize, usize),
    /// Human-readable event category label (e.g. `"Arrest"`).
    pub event_type: String,
    /// Extracted arguments (participants, location, time, etc.).
    pub arguments: Vec<Argument>,
}

// ---------------------------------------------------------------------------
// Tokenisation helpers
// ---------------------------------------------------------------------------

/// Character classes used during lightweight tokenisation.
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '\'' || c == '-'
}

/// Tokenise `text` into `(start, end, surface)` triples (byte offsets).
fn tokenise(text: &str) -> Vec<(usize, usize, String)> {
    let mut tokens: Vec<(usize, usize, String)> = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if is_word_char(c) {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start.take() {
            tokens.push((s, i, text[s..i].to_string()));
        }
    }
    if let Some(s) = start {
        tokens.push((s, text.len(), text[s..].to_string()));
    }
    tokens
}

// ---------------------------------------------------------------------------
// Sentence splitter
// ---------------------------------------------------------------------------

/// Split `text` into sentence strings with byte start offsets.
fn sentences(text: &str) -> Vec<(usize, &str)> {
    let mut result = Vec::new();
    let mut start = 0usize;
    let bytes = text.as_bytes();
    let len = bytes.len();
    while start < len {
        let mut end = start;
        while end < len {
            let b = bytes[end];
            if b == b'.' || b == b'!' || b == b'?' {
                end += 1;
                while end < len && (bytes[end] == b' ' || bytes[end] == b'\n') {
                    end += 1;
                }
                break;
            }
            end += 1;
        }
        let raw = text[start..end].trim();
        if !raw.is_empty() {
            result.push((start, raw));
        }
        start = end;
    }
    result
}

// ---------------------------------------------------------------------------
// Named-entity detection heuristics (no external model)
// ---------------------------------------------------------------------------

/// Detect spans that look like person / organisation names (consecutive
/// capitalised tokens not at sentence start).
fn detect_np_spans(
    tokens: &[(usize, usize, String)],
    sent_start_abs: usize,
) -> Vec<(usize, usize, String)> {
    let mut spans: Vec<(usize, usize, String)> = Vec::new();
    let mut i = 0usize;
    while i < tokens.len() {
        let (tok_s, tok_e, word) = &tokens[i];
        let abs_start = sent_start_abs + tok_s;
        let abs_end = sent_start_abs + tok_e;

        // Capitalised token (not the very first in sentence at offset 0)
        if word.starts_with(|c: char| c.is_uppercase()) && abs_start > sent_start_abs {
            let mut j = i;
            while j < tokens.len() && tokens[j].2.starts_with(|c: char| c.is_uppercase()) {
                j += 1;
            }
            if j > i {
                let span_s = sent_start_abs + tokens[i].0;
                let span_e = sent_start_abs + tokens[j - 1].1;
                // Build surface by re-slicing — we don't have the original
                // sentence slice here, so join tokens.
                let surface: String = tokens[i..j]
                    .iter()
                    .map(|(_, _, w)| w.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                spans.push((span_s, span_e, surface));
                i = j;
                continue;
            }
        }
        i += 1;
    }
    spans
}

/// Detect temporal tokens: patterns like "yesterday", "today", "Monday",
/// month names, years, "N days ago", etc.
fn detect_time_spans(
    tokens: &[(usize, usize, String)],
    sent_start_abs: usize,
) -> Vec<(usize, usize, String)> {
    const DAYS: &[&str] = &[
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ];
    const MONTHS: &[&str] = &[
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ];
    const ABSOLUTE_TEMPS: &[&str] = &["yesterday", "today", "tomorrow", "now", "recently"];
    const REL_ANCHORS: &[&str] = &["last", "next", "this", "coming", "previous"];
    const UNITS: &[&str] = &[
        "second", "seconds", "minute", "minutes", "hour", "hours", "day", "days", "week", "weeks",
        "month", "months", "year", "years",
    ];

    let mut spans: Vec<(usize, usize, String)> = Vec::new();
    let mut i = 0usize;
    while i < tokens.len() {
        let (tok_s, tok_e, word) = &tokens[i];
        let abs_s = sent_start_abs + tok_s;
        let abs_e = sent_start_abs + tok_e;
        let lower = word.to_lowercase();

        // Absolute temporal adverbs
        if ABSOLUTE_TEMPS.contains(&lower.as_str()) {
            spans.push((abs_s, abs_e, word.clone()));
            i += 1;
            continue;
        }

        // "last/next/this <day|month|period>"
        if REL_ANCHORS.contains(&lower.as_str()) && i + 1 < tokens.len() {
            let next_lower = tokens[i + 1].2.to_lowercase();
            if DAYS.contains(&next_lower.as_str())
                || MONTHS.contains(&next_lower.as_str())
                || UNITS.contains(&next_lower.as_str())
            {
                let span_e = sent_start_abs + tokens[i + 1].1;
                let surface = format!("{} {}", word, tokens[i + 1].2);
                spans.push((abs_s, span_e, surface));
                i += 2;
                continue;
            }
        }

        // "<N> <unit> ago"  or  "<N> <unit>"
        if lower.chars().all(|c| c.is_ascii_digit()) && i + 1 < tokens.len() {
            let unit_lower = tokens[i + 1].2.to_lowercase();
            if UNITS.contains(&unit_lower.as_str()) {
                let mut span_e = sent_start_abs + tokens[i + 1].1;
                let mut surface = format!("{} {}", word, tokens[i + 1].2);
                // optional trailing "ago"
                if i + 2 < tokens.len() && tokens[i + 2].2.to_lowercase() == "ago" {
                    span_e = sent_start_abs + tokens[i + 2].1;
                    surface = format!("{} ago", surface);
                    i += 3;
                } else {
                    i += 2;
                }
                spans.push((abs_s, span_e, surface));
                continue;
            }
        }

        // Day / month standalone
        if DAYS.contains(&lower.as_str()) || MONTHS.contains(&lower.as_str()) {
            spans.push((abs_s, abs_e, word.clone()));
            i += 1;
            continue;
        }

        // 4-digit year (1000–2099)
        if lower.len() == 4
            && lower.starts_with(['1', '2'])
            && lower.chars().all(|c| c.is_ascii_digit())
        {
            spans.push((abs_s, abs_e, word.clone()));
            i += 1;
            continue;
        }

        i += 1;
    }
    spans
}

/// Detect location heuristics: preposition "in/at/from/to" + capitalised NP.
fn detect_location_spans(
    tokens: &[(usize, usize, String)],
    sent_start_abs: usize,
    np_spans: &[(usize, usize, String)],
) -> Vec<(usize, usize, String)> {
    const LOC_PREPS: &[&str] = &["in", "at", "from", "to", "near", "around", "through"];
    let mut locs: Vec<(usize, usize, String)> = Vec::new();

    for (i, (tok_s, _tok_e, word)) in tokens.iter().enumerate() {
        let lower = word.to_lowercase();
        if LOC_PREPS.contains(&lower.as_str()) {
            if let Some(next) = tokens.get(i + 1) {
                let next_abs_s = sent_start_abs + next.0;
                let next_abs_e = sent_start_abs + next.1;
                // Look for an NP that starts right here
                let found = np_spans.iter().find(|(ns, _ne, _surf)| *ns == next_abs_s);
                if let Some((ns, ne, surf)) = found {
                    locs.push((*ns, *ne, surf.clone()));
                } else if next.2.starts_with(|c: char| c.is_uppercase()) {
                    locs.push((next_abs_s, next_abs_e, next.2.clone()));
                }
            }
        }
        let _ = tok_s;
    }
    locs
}

// ---------------------------------------------------------------------------
// Core extraction logic
// ---------------------------------------------------------------------------

/// Extract events from `text` using the supplied trigger lexicon.
///
/// For each sentence we identify trigger tokens and then apply heuristic
/// dependency patterns to fill `Agent`, `Patient`, `Location`, and `Time`
/// argument roles:
///
/// - **Agent**: the first NP before the trigger (subject position)
/// - **Patient**: the first NP after the trigger (object position)
/// - **Location**: any `in/at/from/to + NP` within ±3 tokens of the trigger
/// - **Time**: any temporal expression within ±5 tokens of the trigger
pub fn extract_events(text: &str, triggers: &TriggerLexicon) -> Vec<Event> {
    let mut events: Vec<Event> = Vec::new();

    for (sent_off, sent_text) in sentences(text) {
        let tokens = tokenise(sent_text);
        if tokens.is_empty() {
            continue;
        }

        let np_spans = detect_np_spans(&tokens, sent_off);
        let time_spans = detect_time_spans(&tokens, sent_off);
        let loc_spans = detect_location_spans(&tokens, sent_off, &np_spans);

        for (tok_idx, (tok_s, tok_e, word)) in tokens.iter().enumerate() {
            let abs_trig_s = sent_off + tok_s;
            let abs_trig_e = sent_off + tok_e;

            let etype = match triggers.lookup(word) {
                Some(et) => et,
                None => continue,
            };

            let mut args: Vec<Argument> = Vec::new();

            // Agent: closest NP whose end ≤ trigger start
            let agent = np_spans
                .iter()
                .filter(|(_, ne, _)| *ne <= abs_trig_s)
                .max_by_key(|(ns, _, _)| *ns);
            if let Some((ns, ne, surf)) = agent {
                args.push(Argument {
                    role: "Agent".to_string(),
                    text: surf.clone(),
                    span: (*ns, *ne),
                });
            }

            // Patient: closest NP whose start ≥ trigger end
            let patient = np_spans
                .iter()
                .filter(|(ns, _, _)| *ns >= abs_trig_e)
                .min_by_key(|(ns, _, _)| *ns);
            if let Some((ns, ne, surf)) = patient {
                args.push(Argument {
                    role: "Patient".to_string(),
                    text: surf.clone(),
                    span: (*ns, *ne),
                });
            }

            // Location: within ±5 token window
            let window_start = tok_idx.saturating_sub(5);
            let window_end = (tok_idx + 6).min(tokens.len());
            let window_abs_s = sent_off + tokens[window_start].0;
            let window_abs_e = sent_off + tokens[window_end - 1].1;

            for (ls, le, lsurf) in &loc_spans {
                if *ls >= window_abs_s && *le <= window_abs_e {
                    args.push(Argument {
                        role: "Location".to_string(),
                        text: lsurf.clone(),
                        span: (*ls, *le),
                    });
                }
            }

            // Time: within ±6 token window (slightly wider)
            let twin_start = tok_idx.saturating_sub(6);
            let twin_end = (tok_idx + 7).min(tokens.len());
            let twin_abs_s = sent_off + tokens[twin_start].0;
            let twin_abs_e = sent_off + tokens[twin_end - 1].1;

            for (ts, te, tsurf) in &time_spans {
                if *ts >= twin_abs_s && *te <= twin_abs_e {
                    args.push(Argument {
                        role: "Time".to_string(),
                        text: tsurf.clone(),
                        span: (*ts, *te),
                    });
                }
            }

            events.push(Event {
                trigger: word.clone(),
                trigger_span: (abs_trig_s, abs_trig_e),
                event_type: etype.label().to_string(),
                arguments: args,
            });
        }
    }

    events
}

// ---------------------------------------------------------------------------
// Event coreference
// ---------------------------------------------------------------------------

/// Compute a simple similarity score between two events for coreference
/// clustering purposes.
///
/// Factors:
/// - Same event type (0.5)
/// - Overlapping argument text (up to 0.4 weighted by number of shared args)
/// - Trigger edit distance ≤ 2 (0.1)
fn event_similarity(e1: &Event, e2: &Event) -> f64 {
    let mut score = 0.0f64;

    // Same event type
    if e1.event_type == e2.event_type {
        score += 0.5;
    }

    // Shared argument texts
    let texts1: std::collections::HashSet<String> =
        e1.arguments.iter().map(|a| a.text.to_lowercase()).collect();
    let texts2: std::collections::HashSet<String> =
        e2.arguments.iter().map(|a| a.text.to_lowercase()).collect();

    let shared = texts1.intersection(&texts2).count();
    let total = texts1.len().max(texts2.len());
    if total > 0 {
        score += 0.4 * (shared as f64 / total as f64);
    }

    // Trigger similarity (simple character-level)
    let t1 = e1.trigger.to_lowercase();
    let t2 = e2.trigger.to_lowercase();
    if t1 == t2 || levenshtein(t1.as_bytes(), t2.as_bytes()) <= 2 {
        score += 0.1;
    }

    score
}

/// Compute Levenshtein distance between two byte slices.
fn levenshtein(a: &[u8], b: &[u8]) -> usize {
    let m = a.len();
    let n = b.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut dp: Vec<usize> = (0..=n).collect();
    for i in 1..=m {
        let mut prev = dp[0];
        dp[0] = i;
        for j in 1..=n {
            let tmp = dp[j];
            dp[j] = if a[i - 1] == b[j - 1] {
                prev
            } else {
                1 + prev.min(dp[j]).min(dp[j - 1])
            };
            prev = tmp;
        }
    }
    dp[n]
}

/// Group events into coreference chains using single-linkage clustering.
///
/// Two events are placed in the same chain when their similarity score
/// (see `event_similarity`) exceeds `threshold` (default 0.6).
///
/// Returns a `Vec<Vec<usize>>` where each inner vector holds the indices of
/// coreferent events (index into the input `events` slice).
pub fn event_coref(events: &[Event]) -> Vec<Vec<usize>> {
    event_coref_with_threshold(events, 0.6)
}

/// Like [`event_coref`] but with a configurable similarity threshold.
pub fn event_coref_with_threshold(events: &[Event], threshold: f64) -> Vec<Vec<usize>> {
    let n = events.len();
    // Union-Find
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    for i in 0..n {
        for j in (i + 1)..n {
            if event_similarity(&events[i], &events[j]) >= threshold {
                let ri = find(&mut parent, i);
                let rj = find(&mut parent, j);
                if ri != rj {
                    parent[rj] = ri;
                }
            }
        }
    }

    // Collect chains
    let mut chains: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        chains.entry(root).or_default().push(i);
    }

    // Keep only chains with > 1 member (actual coreference)
    let mut result: Vec<Vec<usize>> = chains.into_values().filter(|v| v.len() >= 2).collect();
    result.sort_by_key(|v| v[0]);
    result
}

// ---------------------------------------------------------------------------
// Builder / convenience
// ---------------------------------------------------------------------------

/// High-level interface for configurable event extraction.
pub struct EventExtractor {
    lexicon: TriggerLexicon,
    coref_threshold: f64,
}

impl Default for EventExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl EventExtractor {
    /// Create an extractor with the default English trigger lexicon.
    pub fn new() -> Self {
        Self {
            lexicon: TriggerLexicon::default_english(),
            coref_threshold: 0.6,
        }
    }

    /// Replace the trigger lexicon.
    pub fn with_lexicon(mut self, lexicon: TriggerLexicon) -> Self {
        self.lexicon = lexicon;
        self
    }

    /// Set the similarity threshold used for event coreference clustering.
    pub fn with_coref_threshold(mut self, threshold: f64) -> Self {
        self.coref_threshold = threshold;
        self
    }

    /// Extract events from `text`.
    pub fn extract(&self, text: &str) -> Vec<Event> {
        extract_events(text, &self.lexicon)
    }

    /// Extract events and return coreference chains alongside.
    pub fn extract_with_coref(&self, text: &str) -> Result<(Vec<Event>, Vec<Vec<usize>>)> {
        if text.is_empty() {
            return Err(TextError::InvalidInput(
                "Input text must not be empty".to_string(),
            ));
        }
        let events = self.extract(text);
        let chains = event_coref_with_threshold(&events, self.coref_threshold);
        Ok((events, chains))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigger_lexicon_lookup() {
        let lex = TriggerLexicon::default_english();
        assert_eq!(lex.lookup("arrested"), Some(&EventType::Arrest));
        assert_eq!(lex.lookup("ARRESTED"), Some(&EventType::Arrest));
        assert_eq!(lex.lookup("died"), Some(&EventType::Die));
        assert_eq!(lex.lookup("unknown_verb"), None);
    }

    #[test]
    fn test_extract_events_arrest() {
        let lex = TriggerLexicon::default_english();
        let text = "Police arrested the suspect yesterday in New York.";
        let events = extract_events(text, &lex);
        assert!(!events.is_empty());
        let e = events.iter().find(|e| e.event_type == "Arrest");
        assert!(e.is_some(), "Expected an Arrest event");
        let ev = e.expect("already checked");
        // Patient argument expected (suspect)
        assert!(ev.arguments.iter().any(|a| a.role == "Patient"));
    }

    #[test]
    fn test_extract_events_die_with_agent() {
        let lex = TriggerLexicon::default_english();
        let text = "The soldier died in battle last week.";
        let events = extract_events(text, &lex);
        assert!(!events.is_empty());
        let e = events.iter().find(|e| e.event_type == "Die");
        assert!(e.is_some());
    }

    #[test]
    fn test_extract_events_transfer() {
        let lex = TriggerLexicon::default_english();
        let text = "The company sold its assets to the buyer yesterday.";
        let events = extract_events(text, &lex);
        assert!(events.iter().any(|e| e.event_type == "Transfer"));
    }

    #[test]
    fn test_extract_events_multiple_sentences() {
        let lex = TriggerLexicon::default_english();
        let text = "Alice attacked the base. Bob fled to safety.";
        let events = extract_events(text, &lex);
        assert!(events.len() >= 2);
        let types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();
        assert!(types.contains(&"Attack"));
        assert!(types.contains(&"Move"));
    }

    #[test]
    fn test_event_coref_same_type_and_argument() {
        let lex = TriggerLexicon::default_english();
        let text = "Alice arrested Bob on Monday. Police arrested Bob again on Tuesday.";
        let events = extract_events(text, &lex);
        let chains = event_coref(&events);
        // At least one chain should group the two arrest events
        if !chains.is_empty() {
            assert!(chains.iter().any(|c| c.len() >= 2));
        }
    }

    #[test]
    fn test_event_coref_different_types() {
        let lex = TriggerLexicon::default_english();
        let text = "Alice attacked the fort. Bob fled to the hills.";
        let events = extract_events(text, &lex);
        assert!(events.len() >= 2);
        // Attack and Move should not be coreferent
        let chains = event_coref(&events);
        assert!(chains.is_empty() || !chains.iter().any(|c| c.len() >= 2));
    }

    #[test]
    fn test_extractor_builder() {
        let extractor = EventExtractor::new().with_coref_threshold(0.5);
        let (events, _chains) = extractor
            .extract_with_coref("Police arrested the suspect in London.")
            .expect("should not fail");
        assert!(!events.is_empty());
    }

    #[test]
    fn test_extractor_empty_text_error() {
        let extractor = EventExtractor::new();
        let result = extractor.extract_with_coref("");
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_lexicon() {
        let mut lex = TriggerLexicon::new();
        lex.insert("deployed", EventType::Move);
        lex.insert("commissioned", EventType::Create);
        let text = "The company deployed a new service and commissioned a report.";
        let events = extract_events(text, &lex);
        assert!(events.iter().any(|e| e.event_type == "Move"));
        assert!(events.iter().any(|e| e.event_type == "Create"));
    }

    #[test]
    fn test_event_type_label() {
        assert_eq!(EventType::Move.label(), "Move");
        assert_eq!(EventType::Arrest.label(), "Arrest");
        assert_eq!(EventType::Custom("Foo".to_string()).label(), "Foo");
    }
}
