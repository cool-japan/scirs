//! Dialogue system components
//!
//! Provides rule-based building blocks for constructing conversational agents:
//!
//! - [`DialogState`] – conversation context, entity map, and slot map.
//! - [`IntentClassifier`] – pattern-matching intent recognition.
//! - [`EntityExtractor`] – rule-based extraction of dates, numbers, names, and locations.
//! - [`SlotFiller`] – template-based slot value extraction.
//! - [`DialogPolicy`] – simple state-machine dialog management.
//! - [`DialogAct`] – enum of high-level dialog acts.
//! - [`response_template`] – generate natural-language responses from acts and slots.
//!
//! All components are 100% Pure Rust with no external NLP models.

use crate::error::{Result, TextError};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// DialogAct
// ---------------------------------------------------------------------------

/// High-level dialog act categories used by the dialog policy and response generator.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DialogAct {
    /// Opening greeting.
    Greet,
    /// Request for information or an action from the user.
    Request,
    /// Inform the user of a fact or status.
    Inform,
    /// Ask the user to confirm a value or action.
    Confirm,
    /// Reject a proposed value or action.
    Reject,
    /// Closing farewell.
    Goodbye,
    /// System does not understand the utterance.
    Unknown,
}

impl std::fmt::Display for DialogAct {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::Greet => "GREET",
            Self::Request => "REQUEST",
            Self::Inform => "INFORM",
            Self::Confirm => "CONFIRM",
            Self::Reject => "REJECT",
            Self::Goodbye => "GOODBYE",
            Self::Unknown => "UNKNOWN",
        };
        write!(f, "{label}")
    }
}

// ---------------------------------------------------------------------------
// DialogState
// ---------------------------------------------------------------------------

/// Complete state of an ongoing dialogue.
///
/// Carries the raw utterance history, a map of extracted entities, and a map
/// of domain-specific slot values filled so far.
///
/// # Example
///
/// ```rust
/// use scirs2_text::dialog::{DialogState, DialogAct};
///
/// let mut state = DialogState::new();
/// state.add_utterance("Hello, I want to book a flight to Paris.");
/// state.set_slot("destination", "Paris");
/// assert_eq!(state.get_slot("destination"), Some("Paris"));
/// ```
#[derive(Debug, Clone, Default)]
pub struct DialogState {
    /// Raw utterance history (user turns only).
    pub context: Vec<String>,
    /// Named entities extracted so far.  Maps entity type label → value.
    pub entities: HashMap<String, String>,
    /// Domain-specific slot values.
    pub slots: HashMap<String, String>,
    /// Current dialog act (last recognised).
    pub current_act: Option<DialogAct>,
    /// Number of turns completed.
    pub turn_count: usize,
}

impl DialogState {
    /// Create an empty `DialogState`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a user utterance to the context history.
    pub fn add_utterance(&mut self, utterance: &str) {
        self.context.push(utterance.to_string());
        self.turn_count += 1;
    }

    /// Set a slot value.
    pub fn set_slot(&mut self, slot: &str, value: &str) {
        self.slots.insert(slot.to_string(), value.to_string());
    }

    /// Get a slot value by name.
    pub fn get_slot(&self, slot: &str) -> Option<&str> {
        self.slots.get(slot).map(|s| s.as_str())
    }

    /// Set an entity value.
    pub fn set_entity(&mut self, entity_type: &str, value: &str) {
        self.entities
            .insert(entity_type.to_string(), value.to_string());
    }

    /// Get an entity value by type label.
    pub fn get_entity(&self, entity_type: &str) -> Option<&str> {
        self.entities.get(entity_type).map(|s| s.as_str())
    }

    /// Reset all state (slots, entities, context).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Return the last utterance, or `None` if the context is empty.
    pub fn last_utterance(&self) -> Option<&str> {
        self.context.last().map(|s| s.as_str())
    }

    /// Return `true` if the required `slots` are all filled.
    pub fn slots_filled(&self, required: &[&str]) -> bool {
        required.iter().all(|s| self.slots.contains_key(*s))
    }
}

// ---------------------------------------------------------------------------
// IntentClassifier
// ---------------------------------------------------------------------------

/// Pattern-matching intent classifier.
///
/// Each intent is represented by a name and a list of keyword patterns.  An
/// utterance is matched by counting how many patterns contain at least one
/// word from the utterance; the intent with the most matches wins.
///
/// # Example
///
/// ```rust
/// use scirs2_text::dialog::{IntentClassifier, classify_intent};
///
/// let mut clf = IntentClassifier::new();
/// clf.add_intent("book_flight", vec!["book", "flight", "fly", "ticket"]);
/// clf.add_intent("cancel", vec!["cancel", "undo", "remove"]);
///
/// let (intent, confidence) = classify_intent("I want to book a flight", &clf);
/// assert_eq!(intent, "book_flight");
/// assert!(confidence > 0.0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct IntentClassifier {
    /// Registered intent names in registration order.
    pub intents: Vec<String>,
    /// Patterns for each intent (parallel to `intents`).
    ///
    /// Each element is a list of keyword strings that evidence the intent.
    pub patterns: Vec<Vec<String>>,
}

impl IntentClassifier {
    /// Create an empty classifier.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new intent with its keyword patterns.
    ///
    /// All pattern strings are lower-cased and stored as-is; matching is
    /// performed case-insensitively.
    pub fn add_intent(&mut self, name: &str, patterns: Vec<&str>) {
        self.intents.push(name.to_string());
        self.patterns
            .push(patterns.into_iter().map(|p| p.to_lowercase()).collect());
    }

    /// Return the number of registered intents.
    pub fn len(&self) -> usize {
        self.intents.len()
    }

    /// Return `true` when no intents are registered.
    pub fn is_empty(&self) -> bool {
        self.intents.is_empty()
    }
}

/// Classify an utterance using `classifier`.
///
/// Returns `(intent_name, confidence)`.  Confidence is the normalised fraction
/// of the winning intent's patterns that matched at least one token in the
/// utterance.  When no intent is registered or no patterns match, returns
/// `("unknown", 0.0)`.
pub fn classify_intent(utterance: &str, classifier: &IntentClassifier) -> (String, f64) {
    if classifier.intents.is_empty() {
        return ("unknown".to_string(), 0.0);
    }

    let utt_lower = utterance.to_lowercase();
    let utt_tokens: Vec<&str> = utt_lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();

    let mut best_intent = "unknown".to_string();
    let mut best_score = 0.0_f64;
    let mut best_matches = 0usize;

    for (intent_idx, patterns) in classifier.patterns.iter().enumerate() {
        if patterns.is_empty() {
            continue;
        }
        let total = patterns.len();
        let matches = patterns
            .iter()
            .filter(|pat| {
                // A pattern matches if any token in the utterance starts with or equals the pattern.
                utt_tokens.iter().any(|tok| {
                    *tok == pat.as_str()
                        || tok.starts_with(pat.as_str())
                        || utt_lower.contains(pat.as_str())
                })
            })
            .count();

        let score = matches as f64 / total as f64;
        if matches > best_matches || (matches == best_matches && score > best_score) {
            best_matches = matches;
            best_score = score;
            best_intent = classifier.intents[intent_idx].clone();
        }
    }

    if best_matches == 0 {
        ("unknown".to_string(), 0.0)
    } else {
        (best_intent, best_score)
    }
}

// ---------------------------------------------------------------------------
// EntityExtractor
// ---------------------------------------------------------------------------

/// Recognised entity type labels returned by [`EntityExtractor`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntityKind {
    /// A date expression (e.g. "January 15" or "15/01/2024").
    Date,
    /// A cardinal number or decimal.
    Number,
    /// A probable proper name (heuristic: consecutive capitalised tokens).
    Name,
    /// A location keyword match (heuristic: following "in", "to", "from" etc.).
    Location,
    /// Custom user-defined entity kind.
    Custom(String),
}

impl std::fmt::Display for EntityKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Date => write!(f, "DATE"),
            Self::Number => write!(f, "NUMBER"),
            Self::Name => write!(f, "NAME"),
            Self::Location => write!(f, "LOCATION"),
            Self::Custom(s) => write!(f, "CUSTOM({})", s),
        }
    }
}

/// A single entity extracted from an utterance.
#[derive(Debug, Clone)]
pub struct ExtractedEntity {
    /// The matched text.
    pub text: String,
    /// Entity kind.
    pub kind: EntityKind,
    /// Character start offset (byte index).
    pub start: usize,
    /// Character end offset (byte index, exclusive).
    pub end: usize,
}

/// Rule-based entity extractor for dialogue systems.
///
/// Extracts dates, numbers, potential names (capitalised phrases), and
/// location hints without relying on trained models.
#[derive(Debug, Default)]
pub struct EntityExtractor {
    /// Additional custom gazetteer entries (lowercase term → EntityKind).
    gazetteer: Vec<(String, EntityKind)>,
}

impl EntityExtractor {
    /// Create a new `EntityExtractor` with default rules.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a custom gazetteer entry.  Matching is case-insensitive.
    pub fn add_gazetteer_entry(&mut self, term: &str, kind: EntityKind) {
        self.gazetteer.push((term.to_lowercase(), kind));
    }

    /// Extract entities from `utterance`.
    ///
    /// The extraction order is: gazetteer, dates, numbers, names (consecutive
    /// capitalised tokens), location hints.  Overlapping spans are not
    /// deduplicated; callers should post-process if needed.
    pub fn extract(&self, utterance: &str) -> Vec<ExtractedEntity> {
        let mut entities: Vec<ExtractedEntity> = Vec::new();

        self.extract_gazetteer(utterance, &mut entities);
        self.extract_dates(utterance, &mut entities);
        self.extract_numbers(utterance, &mut entities);
        self.extract_names(utterance, &mut entities);
        self.extract_locations(utterance, &mut entities);

        // Sort by start position.
        entities.sort_by_key(|e| e.start);
        entities
    }

    /// Match gazetteer entries (exact, case-insensitive substring search).
    fn extract_gazetteer(&self, text: &str, out: &mut Vec<ExtractedEntity>) {
        let text_lower = text.to_lowercase();
        for (term, kind) in &self.gazetteer {
            let mut search_start = 0usize;
            while let Some(offset) = text_lower[search_start..].find(term.as_str()) {
                let abs_start = search_start + offset;
                let abs_end = abs_start + term.len();
                out.push(ExtractedEntity {
                    text: text[abs_start..abs_end].to_string(),
                    kind: kind.clone(),
                    start: abs_start,
                    end: abs_end,
                });
                search_start = abs_end;
            }
        }
    }

    /// Extract date expressions using simple patterns.
    ///
    /// Recognises:
    /// - `DD/MM/YYYY` or `MM/DD/YYYY` (slash-separated numbers).
    /// - Month name followed by an optional day number.
    fn extract_dates(&self, text: &str, out: &mut Vec<ExtractedEntity>) {
        // Slash-delimited dates: digits / digits (/ digits)?
        let mut i = 0;
        let bytes = text.as_bytes();
        let len = bytes.len();

        while i < len {
            // Skip non-digit characters.
            if !bytes[i].is_ascii_digit() {
                i += 1;
                continue;
            }
            // Consume digit run.
            let start = i;
            while i < len && bytes[i].is_ascii_digit() {
                i += 1;
            }
            // Look for /digits(/digits)?
            if i < len && bytes[i] == b'/' {
                let slash1 = i;
                i += 1;
                let seg2_start = i;
                while i < len && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                if i > seg2_start {
                    let end = if i < len && bytes[i] == b'/' {
                        i += 1; // consume second slash
                        let seg3_start = i;
                        while i < len && bytes[i].is_ascii_digit() {
                            i += 1;
                        }
                        if i > seg3_start {
                            i
                        } else {
                            // Backtrack the slash.
                            i = slash1 + 1 + (i - slash1 - 1);
                            slash1
                        }
                    } else {
                        i
                    };
                    let matched = &text[start..end];
                    if matched.contains('/') {
                        out.push(ExtractedEntity {
                            text: matched.to_string(),
                            kind: EntityKind::Date,
                            start,
                            end,
                        });
                    }
                }
                continue;
            }
        }

        // Month-name patterns.
        let months = [
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
        let text_lower = text.to_lowercase();
        for month in &months {
            let mut search_pos = 0usize;
            while let Some(offset) = text_lower[search_pos..].find(month) {
                let abs_start = search_pos + offset;
                let abs_end = abs_start + month.len();

                // Make sure it's a word boundary (not mid-word).
                let before_ok =
                    abs_start == 0 || !text.as_bytes()[abs_start - 1].is_ascii_alphanumeric();
                let after_ok =
                    abs_end >= text.len() || !text.as_bytes()[abs_end].is_ascii_alphanumeric();

                if before_ok && after_ok {
                    // Optionally consume a following number (day).
                    let mut end = abs_end;
                    let rest = &text[abs_end..];
                    let after_space: &str = rest.trim_start_matches(' ');
                    let day_len: usize = after_space
                        .chars()
                        .take_while(|c| c.is_ascii_digit())
                        .map(|c| c.len_utf8())
                        .sum();
                    if day_len > 0 {
                        let spaces = rest.len() - after_space.len();
                        end += spaces + day_len;
                    }

                    // Also try to consume a following 4-digit year.
                    let rest2 = &text[end..];
                    let after_space2: &str = rest2.trim_start_matches(' ');
                    let year_candidate: String = after_space2
                        .chars()
                        .take_while(|c| c.is_ascii_digit())
                        .collect();
                    if year_candidate.len() == 4 {
                        let spaces2 = rest2.len() - after_space2.len();
                        end += spaces2 + 4;
                    }

                    out.push(ExtractedEntity {
                        text: text[abs_start..end].to_string(),
                        kind: EntityKind::Date,
                        start: abs_start,
                        end,
                    });
                }

                search_pos = abs_end;
            }
        }
    }

    /// Extract cardinal numbers (integers and decimals).
    fn extract_numbers(&self, text: &str, out: &mut Vec<ExtractedEntity>) {
        let mut i = 0;
        let bytes = text.as_bytes();
        let len = bytes.len();

        while i < len {
            if !bytes[i].is_ascii_digit() {
                i += 1;
                continue;
            }
            let start = i;
            while i < len && bytes[i].is_ascii_digit() {
                i += 1;
            }
            // Optionally consume a decimal part.
            if i < len && (bytes[i] == b'.' || bytes[i] == b',') {
                let sep = i;
                i += 1;
                let frac_start = i;
                while i < len && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                if i == frac_start {
                    // Nothing after separator — backtrack.
                    i = sep;
                }
            }
            // Ensure not part of a slash-date already extracted (heuristic: skip
            // entries that overlap with a date entity).
            let end = i;
            let candidate = &text[start..end];
            let already_date = out
                .iter()
                .any(|e| e.kind == EntityKind::Date && e.start <= start && e.end >= end);
            if !already_date {
                out.push(ExtractedEntity {
                    text: candidate.to_string(),
                    kind: EntityKind::Number,
                    start,
                    end,
                });
            }
        }
    }

    /// Extract probable proper names: runs of two or more consecutive
    /// capitalised tokens (words beginning with an uppercase letter) that are
    /// not preceded by a sentence-initial position.
    fn extract_names(&self, text: &str, out: &mut Vec<ExtractedEntity>) {
        // Split on whitespace while tracking byte offsets.
        let mut word_spans: Vec<(usize, usize, &str)> = Vec::new();
        let mut pos = 0usize;
        for word in text.split_ascii_whitespace() {
            // Find the word in text starting from pos.
            if let Some(offset) = text[pos..].find(word) {
                let start = pos + offset;
                let end = start + word.len();
                word_spans.push((start, end, word));
                pos = end;
            }
        }

        // Find runs of capitalised tokens (heuristic proper-name detection).
        let mut i = 0usize;
        while i < word_spans.len() {
            let (start, _, word) = word_spans[i];
            // Strip leading punctuation.
            let first_alpha = word.chars().find(|c| c.is_alphabetic());
            let is_cap = first_alpha.map(|c| c.is_uppercase()).unwrap_or(false);

            if !is_cap {
                i += 1;
                continue;
            }

            // Consume the run.
            let run_start = start;
            let mut j = i;
            while j < word_spans.len() {
                let (_, _, w) = word_spans[j];
                let fc = w.chars().find(|c| c.is_alphabetic());
                if fc.map(|c| c.is_uppercase()).unwrap_or(false) {
                    j += 1;
                } else {
                    break;
                }
            }

            // Only emit runs of 2+ tokens.
            if j - i >= 2 {
                let (_, run_end, _) = word_spans[j - 1];
                let name_text = &text[run_start..run_end];
                out.push(ExtractedEntity {
                    text: name_text.to_string(),
                    kind: EntityKind::Name,
                    start: run_start,
                    end: run_end,
                });
                i = j;
            } else {
                i += 1;
            }
        }
    }

    /// Extract location hints using positional keywords ("in", "to", "from",
    /// "at", "near", "between") followed by a capitalised token.
    fn extract_locations(&self, text: &str, out: &mut Vec<ExtractedEntity>) {
        let location_triggers = ["in ", "to ", "from ", "at ", "near ", "between "];
        for trigger in &location_triggers {
            let text_lower = text.to_lowercase();
            let mut search_pos = 0usize;
            while let Some(offset) = text_lower[search_pos..].find(trigger) {
                let abs_trigger_start = search_pos + offset;
                let candidate_start = abs_trigger_start + trigger.len();
                if candidate_start >= text.len() {
                    break;
                }

                // Consume the capitalised word (or phrase of capitalised words).
                let rest = &text[candidate_start..];
                let mut loc_end = candidate_start;
                for word in rest.split_ascii_whitespace() {
                    let first_char = word
                        .trim_matches(|c: char| !c.is_alphabetic())
                        .chars()
                        .next();
                    if first_char.map(|c| c.is_uppercase()).unwrap_or(false) {
                        loc_end += word.len() + 1; // +1 for the space
                    } else {
                        break;
                    }
                }
                // Trim trailing separator.
                let loc_end = loc_end.min(text.len());
                if loc_end > candidate_start {
                    let loc_text = text[candidate_start..loc_end].trim().to_string();
                    if !loc_text.is_empty() {
                        let actual_end = candidate_start + loc_text.len();
                        out.push(ExtractedEntity {
                            text: loc_text,
                            kind: EntityKind::Location,
                            start: candidate_start,
                            end: actual_end,
                        });
                    }
                }

                search_pos = candidate_start;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SlotFiller
// ---------------------------------------------------------------------------

/// Template-based slot filler.
///
/// A slot template is a string like `"fly from {origin} to {destination}"`.
/// The filler extracts the values of `{origin}` and `{destination}` by
/// matching the literal parts of the template against the utterance.
///
/// # Example
///
/// ```rust
/// use scirs2_text::dialog::SlotFiller;
///
/// let sf = SlotFiller::new();
/// let slots = sf.fill("book a flight from London to Paris",
///                      "flight from {origin} to {destination}").unwrap();
/// assert_eq!(slots.get("origin").map(|s| s.as_str()), Some("London"));
/// assert_eq!(slots.get("destination").map(|s| s.as_str()), Some("Paris"));
/// ```
#[derive(Debug, Default, Clone)]
pub struct SlotFiller;

impl SlotFiller {
    /// Create a new `SlotFiller`.
    pub fn new() -> Self {
        Self
    }

    /// Fill slots defined by `template` from `utterance`.
    ///
    /// Template syntax: literal text with `{slot_name}` placeholders.
    ///
    /// Returns a map of slot names to their extracted values, or an error if
    /// the template cannot be parsed.
    pub fn fill(&self, utterance: &str, template: &str) -> Result<HashMap<String, String>> {
        // Parse the template into alternating literals and slot names.
        let parts = parse_template(template)?;
        let mut slots: HashMap<String, String> = HashMap::new();

        // Try to match the utterance against the template parts.
        let utt_lower = utterance.to_lowercase();
        let mut search_pos = 0usize;

        let n = parts.len();
        let mut pi = 0usize;

        while pi < n {
            match &parts[pi] {
                TemplatePart::Literal(lit) => {
                    let lit_lower = lit.to_lowercase();
                    if lit_lower.is_empty() {
                        pi += 1;
                        continue;
                    }
                    if let Some(offset) = utt_lower[search_pos..].find(lit_lower.as_str()) {
                        search_pos += offset + lit.len();
                        pi += 1;
                    } else {
                        // Literal not found; stop matching.
                        break;
                    }
                }
                TemplatePart::Slot(slot_name) => {
                    // The slot value runs up to the next literal (or end of string).
                    let next_literal: Option<&str> = parts[pi + 1..].iter().find_map(|p| {
                        if let TemplatePart::Literal(s) = p {
                            if !s.is_empty() {
                                Some(s.as_str())
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    });

                    let value_end = if let Some(next_lit) = next_literal {
                        let next_lit_lower = next_lit.to_lowercase();
                        utt_lower[search_pos..]
                            .find(next_lit_lower.as_str())
                            .map(|o| search_pos + o)
                            .unwrap_or(utt_lower.len())
                    } else {
                        utt_lower.len()
                    };

                    let raw_value = utterance[search_pos..value_end].trim().to_string();
                    if !raw_value.is_empty() {
                        slots.insert(slot_name.clone(), raw_value);
                    }
                    search_pos = value_end;
                    pi += 1;
                }
            }
        }

        Ok(slots)
    }
}

/// Internal template part.
#[derive(Debug)]
enum TemplatePart {
    Literal(String),
    Slot(String),
}

/// Parse a template string into a vector of [`TemplatePart`] items.
fn parse_template(template: &str) -> Result<Vec<TemplatePart>> {
    let mut parts: Vec<TemplatePart> = Vec::new();
    let mut chars = template.char_indices().peekable();
    let mut buf = String::new();

    while let Some((_, ch)) = chars.next() {
        if ch == '{' {
            // Flush literal buffer.
            if !buf.is_empty() {
                parts.push(TemplatePart::Literal(std::mem::take(&mut buf)));
            }
            // Read slot name until '}'.
            let mut slot_name = String::new();
            let mut closed = false;
            for (_, sc) in chars.by_ref() {
                if sc == '}' {
                    closed = true;
                    break;
                }
                slot_name.push(sc);
            }
            if !closed {
                return Err(TextError::InvalidInput(
                    "Unclosed '{' in slot template".to_string(),
                ));
            }
            if slot_name.is_empty() {
                return Err(TextError::InvalidInput(
                    "Empty slot name '{}' in template".to_string(),
                ));
            }
            parts.push(TemplatePart::Slot(slot_name));
        } else {
            buf.push(ch);
        }
    }

    if !buf.is_empty() {
        parts.push(TemplatePart::Literal(buf));
    }

    Ok(parts)
}

// ---------------------------------------------------------------------------
// DialogPolicy
// ---------------------------------------------------------------------------

/// States of the built-in dialog state machine.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PolicyState {
    /// Initial state (no turns yet).
    Initial,
    /// Greeting has been exchanged.
    Greeted,
    /// System is collecting slot values.
    SlotCollection,
    /// All required slots are filled; awaiting confirmation.
    PendingConfirmation,
    /// Transaction confirmed and executed.
    Confirmed,
    /// Dialog has ended.
    Ended,
}

impl std::fmt::Display for PolicyState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Initial => "INITIAL",
            Self::Greeted => "GREETED",
            Self::SlotCollection => "SLOT_COLLECTION",
            Self::PendingConfirmation => "PENDING_CONFIRMATION",
            Self::Confirmed => "CONFIRMED",
            Self::Ended => "ENDED",
        };
        write!(f, "{s}")
    }
}

/// A recommended system action emitted by the policy.
#[derive(Debug, Clone)]
pub struct PolicyAction {
    /// The recommended dialog act.
    pub act: DialogAct,
    /// Which slot to request next (if act == Request).
    pub request_slot: Option<String>,
    /// Slots to confirm (if act == Confirm).
    pub confirm_slots: Vec<String>,
}

/// Simple state-machine dialog policy.
///
/// Drives a slot-filling task dialog through greeting → slot collection →
/// confirmation → end.
///
/// # Example
///
/// ```rust
/// use scirs2_text::dialog::{DialogPolicy, DialogState, DialogAct};
///
/// let mut policy = DialogPolicy::new(vec!["origin".to_string(), "destination".to_string()]);
/// let mut state = DialogState::new();
///
/// let action = policy.next_action(&state);
/// assert_eq!(action.act, DialogAct::Greet);
/// ```
pub struct DialogPolicy {
    /// Slots that must be filled before confirming.
    required_slots: Vec<String>,
    /// Current state-machine state.
    policy_state: PolicyState,
}

impl DialogPolicy {
    /// Create a new `DialogPolicy` requiring the specified slots.
    pub fn new(required_slots: Vec<String>) -> Self {
        Self {
            required_slots,
            policy_state: PolicyState::Initial,
        }
    }

    /// Current state-machine state.
    pub fn state(&self) -> &PolicyState {
        &self.policy_state
    }

    /// Compute the next recommended action given the current `dialog_state`.
    ///
    /// This also advances the internal state machine.
    pub fn next_action(&mut self, dialog_state: &DialogState) -> PolicyAction {
        match self.policy_state {
            PolicyState::Initial => {
                self.policy_state = PolicyState::Greeted;
                PolicyAction {
                    act: DialogAct::Greet,
                    request_slot: None,
                    confirm_slots: Vec::new(),
                }
            }
            PolicyState::Greeted | PolicyState::SlotCollection => {
                // Find the first unfilled required slot.
                let missing = self
                    .required_slots
                    .iter()
                    .find(|s| !dialog_state.slots.contains_key(*s))
                    .cloned();

                if let Some(slot) = missing {
                    self.policy_state = PolicyState::SlotCollection;
                    PolicyAction {
                        act: DialogAct::Request,
                        request_slot: Some(slot),
                        confirm_slots: Vec::new(),
                    }
                } else {
                    // All slots filled.
                    self.policy_state = PolicyState::PendingConfirmation;
                    PolicyAction {
                        act: DialogAct::Confirm,
                        request_slot: None,
                        confirm_slots: self.required_slots.clone(),
                    }
                }
            }
            PolicyState::PendingConfirmation => {
                // Check the last utterance for yes/no.
                let confirmed = dialog_state
                    .last_utterance()
                    .map(|u| {
                        let ul = u.to_lowercase();
                        ul.contains("yes")
                            || ul.contains("correct")
                            || ul.contains("right")
                            || ul.contains("confirm")
                    })
                    .unwrap_or(false);

                if confirmed {
                    self.policy_state = PolicyState::Confirmed;
                    PolicyAction {
                        act: DialogAct::Inform,
                        request_slot: None,
                        confirm_slots: Vec::new(),
                    }
                } else {
                    // Assume rejection / restart slot collection.
                    self.policy_state = PolicyState::SlotCollection;
                    PolicyAction {
                        act: DialogAct::Reject,
                        request_slot: None,
                        confirm_slots: Vec::new(),
                    }
                }
            }
            PolicyState::Confirmed => {
                self.policy_state = PolicyState::Ended;
                PolicyAction {
                    act: DialogAct::Goodbye,
                    request_slot: None,
                    confirm_slots: Vec::new(),
                }
            }
            PolicyState::Ended => PolicyAction {
                act: DialogAct::Goodbye,
                request_slot: None,
                confirm_slots: Vec::new(),
            },
        }
    }

    /// Reset the policy to its initial state.
    pub fn reset(&mut self) {
        self.policy_state = PolicyState::Initial;
    }
}

// ---------------------------------------------------------------------------
// response_template
// ---------------------------------------------------------------------------

/// Generate a natural-language response for the given `act` and `slots`.
///
/// Slot values are substituted into the response where the placeholder
/// `{slot_name}` appears.  Unknown slot references are left as-is.
///
/// # Example
///
/// ```rust
/// use scirs2_text::dialog::{response_template, DialogAct};
/// use std::collections::HashMap;
///
/// let mut slots = HashMap::new();
/// slots.insert("destination".to_string(), "Paris".to_string());
///
/// let response = response_template(DialogAct::Inform, &slots);
/// assert!(!response.is_empty());
/// ```
pub fn response_template(act: DialogAct, slots: &HashMap<String, String>) -> String {
    let template = match act {
        DialogAct::Greet => "Hello! How can I help you today?".to_string(),
        DialogAct::Request => {
            // Pick the first slot that has a value in the slot map as a hint,
            // otherwise fall back to a generic request.
            let slot_hint = slots
                .keys()
                .next()
                .map(|s| s.as_str())
                .unwrap_or("information");
            format!("Could you please provide the {slot_hint}?")
        }
        DialogAct::Inform => {
            if slots.is_empty() {
                "I have processed your request successfully.".to_string()
            } else {
                let details: Vec<String> = slots.iter().map(|(k, v)| format!("{k}: {v}")).collect();
                format!("Here is the information: {}.", details.join(", "))
            }
        }
        DialogAct::Confirm => {
            if slots.is_empty() {
                "Can you please confirm your request?".to_string()
            } else {
                let details: Vec<String> =
                    slots.iter().map(|(k, v)| format!("{k} = {v}")).collect();
                format!(
                    "Just to confirm, you would like to proceed with {}. Is that correct?",
                    details.join(", ")
                )
            }
        }
        DialogAct::Reject => {
            "I'm sorry, that does not match what we have. Let's try again.".to_string()
        }
        DialogAct::Goodbye => "Thank you for using our service. Goodbye!".to_string(),
        DialogAct::Unknown => {
            "I'm sorry, I didn't understand that. Could you rephrase?".to_string()
        }
    };

    // Substitute slot values into the template.
    let mut result = template;
    for (key, value) in slots {
        let placeholder = format!("{{{key}}}");
        result = result.replace(&placeholder, value);
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- DialogState --

    #[test]
    fn test_dialog_state_slots() {
        let mut state = DialogState::new();
        state.set_slot("destination", "Paris");
        assert_eq!(state.get_slot("destination"), Some("Paris"));
        assert_eq!(state.get_slot("origin"), None);
    }

    #[test]
    fn test_dialog_state_entities() {
        let mut state = DialogState::new();
        state.set_entity("DATE", "January 15");
        assert_eq!(state.get_entity("DATE"), Some("January 15"));
    }

    #[test]
    fn test_dialog_state_utterances() {
        let mut state = DialogState::new();
        assert!(state.last_utterance().is_none());
        state.add_utterance("Hello");
        assert_eq!(state.last_utterance(), Some("Hello"));
        state.add_utterance("Goodbye");
        assert_eq!(state.last_utterance(), Some("Goodbye"));
        assert_eq!(state.turn_count, 2);
    }

    #[test]
    fn test_dialog_state_slots_filled() {
        let mut state = DialogState::new();
        state.set_slot("a", "1");
        state.set_slot("b", "2");
        assert!(state.slots_filled(&["a", "b"]));
        assert!(!state.slots_filled(&["a", "b", "c"]));
    }

    #[test]
    fn test_dialog_state_reset() {
        let mut state = DialogState::new();
        state.set_slot("x", "y");
        state.add_utterance("hello");
        state.reset();
        assert!(state.slots.is_empty());
        assert!(state.context.is_empty());
        assert_eq!(state.turn_count, 0);
    }

    // -- IntentClassifier --

    #[test]
    fn test_classify_intent_basic() {
        let mut clf = IntentClassifier::new();
        clf.add_intent("book_flight", vec!["book", "flight", "fly", "ticket"]);
        clf.add_intent("cancel", vec!["cancel", "undo", "delete"]);

        let (intent, conf) = classify_intent("I want to book a flight", &clf);
        assert_eq!(intent, "book_flight");
        assert!(conf > 0.0);
    }

    #[test]
    fn test_classify_intent_unknown() {
        let clf = IntentClassifier::new();
        let (intent, conf) = classify_intent("hello", &clf);
        assert_eq!(intent, "unknown");
        assert_eq!(conf, 0.0);
    }

    #[test]
    fn test_classify_intent_no_match() {
        let mut clf = IntentClassifier::new();
        clf.add_intent("book_flight", vec!["book", "flight"]);
        let (intent, conf) = classify_intent("tell me the weather", &clf);
        assert_eq!(intent, "unknown");
        assert_eq!(conf, 0.0);
    }

    #[test]
    fn test_classify_intent_case_insensitive() {
        let mut clf = IntentClassifier::new();
        clf.add_intent("greet", vec!["hello", "hi", "hey"]);
        let (intent, _conf) = classify_intent("HELLO there", &clf);
        assert_eq!(intent, "greet");
    }

    // -- EntityExtractor --

    #[test]
    fn test_extract_numbers() {
        let ext = EntityExtractor::new();
        let entities = ext.extract("I need 3 tickets and 12.5 kg baggage");
        let numbers: Vec<&str> = entities
            .iter()
            .filter(|e| e.kind == EntityKind::Number)
            .map(|e| e.text.as_str())
            .collect();
        assert!(numbers.contains(&"3"), "Missing '3': {:?}", numbers);
        assert!(numbers.contains(&"12.5"), "Missing '12.5': {:?}", numbers);
    }

    #[test]
    fn test_extract_date_month_name() {
        let ext = EntityExtractor::new();
        let entities = ext.extract("The flight is on January 15");
        let dates: Vec<&str> = entities
            .iter()
            .filter(|e| e.kind == EntityKind::Date)
            .map(|e| e.text.as_str())
            .collect();
        assert!(!dates.is_empty(), "Expected at least one date entity");
        assert!(
            dates.iter().any(|d| d.contains("January")),
            "Expected 'January' in dates: {:?}",
            dates
        );
    }

    #[test]
    fn test_extract_gazetteer() {
        let mut ext = EntityExtractor::new();
        ext.add_gazetteer_entry("london", EntityKind::Location);
        let entities = ext.extract("I want to travel to London");
        let locs: Vec<&str> = entities
            .iter()
            .filter(|e| e.kind == EntityKind::Location)
            .map(|e| e.text.as_str())
            .collect();
        assert!(!locs.is_empty(), "Expected location entity");
    }

    // -- SlotFiller --

    #[test]
    fn test_slot_filler_basic() {
        let sf = SlotFiller::new();
        let slots = sf
            .fill(
                "book a flight from London to Paris",
                "flight from {origin} to {destination}",
            )
            .expect("fill should succeed");
        assert_eq!(slots.get("origin").map(|s| s.as_str()), Some("London"));
        assert_eq!(slots.get("destination").map(|s| s.as_str()), Some("Paris"));
    }

    #[test]
    fn test_slot_filler_single_slot() {
        let sf = SlotFiller::new();
        let slots = sf
            .fill("my name is Alice", "my name is {name}")
            .expect("fill should succeed");
        assert_eq!(slots.get("name").map(|s| s.as_str()), Some("Alice"));
    }

    #[test]
    fn test_slot_filler_unclosed_brace_error() {
        let sf = SlotFiller::new();
        let result = sf.fill("hello world", "hello {world");
        assert!(result.is_err(), "Expected error for unclosed brace");
    }

    #[test]
    fn test_slot_filler_no_match() {
        let sf = SlotFiller::new();
        let slots = sf
            .fill(
                "completely different text",
                "flight from {origin} to {destination}",
            )
            .expect("should not error");
        // Slots should be empty since the literal prefix didn't match.
        assert!(
            !slots.contains_key("origin") && !slots.contains_key("destination"),
            "Expected no slots when template does not match"
        );
    }

    // -- DialogPolicy --

    #[test]
    fn test_policy_initial_greet() {
        let mut policy = DialogPolicy::new(vec!["origin".to_string(), "destination".to_string()]);
        let state = DialogState::new();
        let action = policy.next_action(&state);
        assert_eq!(action.act, DialogAct::Greet);
    }

    #[test]
    fn test_policy_requests_missing_slot() {
        let mut policy = DialogPolicy::new(vec!["origin".to_string(), "destination".to_string()]);
        let mut state = DialogState::new();
        policy.next_action(&state); // Greet
        let action = policy.next_action(&state);
        assert_eq!(action.act, DialogAct::Request);
        assert!(action.request_slot.is_some());
    }

    #[test]
    fn test_policy_confirms_when_slots_filled() {
        let mut policy = DialogPolicy::new(vec!["origin".to_string(), "destination".to_string()]);
        let mut state = DialogState::new();
        policy.next_action(&state); // Greet
        state.set_slot("origin", "London");
        state.set_slot("destination", "Paris");
        let action = policy.next_action(&state);
        assert_eq!(action.act, DialogAct::Confirm);
    }

    #[test]
    fn test_policy_informs_after_confirmation() {
        let mut policy = DialogPolicy::new(vec!["origin".to_string()]);
        let mut state = DialogState::new();
        policy.next_action(&state); // Greet
        state.set_slot("origin", "London");
        policy.next_action(&state); // Confirm
        state.add_utterance("yes");
        let action = policy.next_action(&state);
        assert_eq!(action.act, DialogAct::Inform);
    }

    #[test]
    fn test_policy_goodbye_at_end() {
        let mut policy = DialogPolicy::new(vec!["origin".to_string()]);
        let mut state = DialogState::new();
        policy.next_action(&state); // Greet (→ Greeted)
        state.set_slot("origin", "London");
        policy.next_action(&state); // Confirm (→ PendingConfirmation)
        state.add_utterance("yes");
        policy.next_action(&state); // Inform (→ Confirmed)
        let action = policy.next_action(&state); // Goodbye (→ Ended)
        assert_eq!(action.act, DialogAct::Goodbye);
    }

    #[test]
    fn test_policy_reset() {
        let mut policy = DialogPolicy::new(vec!["slot_a".to_string()]);
        let state = DialogState::new();
        policy.next_action(&state);
        assert_ne!(*policy.state(), PolicyState::Initial);
        policy.reset();
        assert_eq!(*policy.state(), PolicyState::Initial);
    }

    // -- response_template --

    #[test]
    fn test_response_greet() {
        let slots: HashMap<String, String> = HashMap::new();
        let response = response_template(DialogAct::Greet, &slots);
        assert!(!response.is_empty());
        let lower = response.to_lowercase();
        assert!(
            lower.contains("hello") || lower.contains("hi") || lower.contains("help"),
            "Greet response should be a greeting: '{response}'"
        );
    }

    #[test]
    fn test_response_inform_with_slots() {
        let mut slots: HashMap<String, String> = HashMap::new();
        slots.insert("destination".to_string(), "Paris".to_string());
        let response = response_template(DialogAct::Inform, &slots);
        assert!(
            response.contains("Paris"),
            "Response should contain 'Paris': '{response}'"
        );
    }

    #[test]
    fn test_response_goodbye() {
        let slots: HashMap<String, String> = HashMap::new();
        let response = response_template(DialogAct::Goodbye, &slots);
        let lower = response.to_lowercase();
        assert!(
            lower.contains("goodbye") || lower.contains("bye") || lower.contains("thank"),
            "Goodbye response unexpected: '{response}'"
        );
    }

    #[test]
    fn test_response_confirm_with_slots() {
        let mut slots: HashMap<String, String> = HashMap::new();
        slots.insert("origin".to_string(), "London".to_string());
        slots.insert("destination".to_string(), "Tokyo".to_string());
        let response = response_template(DialogAct::Confirm, &slots);
        assert!(!response.is_empty());
    }

    #[test]
    fn test_response_unknown() {
        let slots: HashMap<String, String> = HashMap::new();
        let response = response_template(DialogAct::Unknown, &slots);
        assert!(!response.is_empty());
    }

    // -- DialogAct display --

    #[test]
    fn test_dialog_act_display() {
        assert_eq!(DialogAct::Greet.to_string(), "GREET");
        assert_eq!(DialogAct::Goodbye.to_string(), "GOODBYE");
        assert_eq!(DialogAct::Unknown.to_string(), "UNKNOWN");
    }
}
