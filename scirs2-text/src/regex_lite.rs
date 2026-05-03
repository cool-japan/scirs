//! # Regex-Lite Engine
//!
//! A pure Rust, NFA-based regular expression engine using Thompson construction.
//! Supports character classes, quantifiers, alternation, grouping, anchors,
//! and character class shortcuts.
//!
//! This module provides a lightweight regex engine that does not depend on any
//! external regex crate. It compiles patterns into an NFA (nondeterministic finite
//! automaton) and simulates it for matching.
//!
//! ## Supported Syntax
//!
//! - **Literal characters**: `a`, `b`, `1`
//! - **Escaped characters**: `\\n`, `\\t`, `\\r`, `\\.`, `\\*`, `\\+`, `\\?`, `\\(`, `\\)`
//! - **Character classes**: `[abc]`, `[a-z]`, `[^0-9]`
//! - **Shortcuts**: `\\d` (digit), `\\w` (word char), `\\s` (whitespace), `\\D`, `\\W`, `\\S`
//! - **Dot**: `.` matches any character except newline
//! - **Quantifiers**: `*` (zero or more), `+` (one or more), `?` (zero or one)
//! - **Alternation**: `a|b`
//! - **Grouping**: `(abc)`
//! - **Anchors**: `^` (start), `$` (end)
//!
//! ## Example
//!
//! ```rust
//! use scirs2_text::regex_lite::RegexLite;
//!
//! let re = RegexLite::new(r"\d+").unwrap();
//! assert!(re.is_match("hello 42 world"));
//!
//! let m = re.find("price: 99 dollars").unwrap();
//! assert_eq!(m.as_str(), "99");
//!
//! let all: Vec<_> = re.find_all("1 and 22 and 333").iter().map(|m| m.as_str()).collect();
//! assert_eq!(all, vec!["1", "22", "333"]);
//! ```

use crate::error::{Result, TextError};

// ---------------------------------------------------------------------------
// AST (parsed regex tree)
// ---------------------------------------------------------------------------

/// A node in the parsed regex abstract syntax tree.
#[derive(Debug, Clone)]
enum Ast {
    /// Match a single literal character.
    Literal(char),
    /// `.` — match any character except newline.
    Dot,
    /// Character class `[...]` with ranges and negation flag.
    CharClass {
        ranges: Vec<(char, char)>,
        negated: bool,
    },
    /// Alternation `a|b`.
    Alt(Box<Ast>, Box<Ast>),
    /// Concatenation of two sub-expressions.
    Concat(Box<Ast>, Box<Ast>),
    /// Zero or more (`*`).
    Star(Box<Ast>),
    /// One or more (`+`).
    Plus(Box<Ast>),
    /// Zero or one (`?`).
    Quest(Box<Ast>),
    /// Capturing group `(...)`.
    Group(Box<Ast>),
    /// `^` anchor.
    AnchorStart,
    /// `$` anchor.
    AnchorEnd,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parser that converts a pattern string into an AST.
struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn new(pattern: &str) -> Self {
        Self {
            chars: pattern.chars().collect(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn parse(&mut self) -> Result<Ast> {
        let ast = self.parse_alternation()?;
        if self.pos < self.chars.len() {
            return Err(TextError::InvalidInput(format!(
                "Unexpected character '{}' at position {} in regex",
                self.chars[self.pos], self.pos
            )));
        }
        Ok(ast)
    }

    fn parse_alternation(&mut self) -> Result<Ast> {
        let mut left = self.parse_concat()?;
        while self.peek() == Some('|') {
            self.advance(); // consume '|'
            let right = self.parse_concat()?;
            left = Ast::Alt(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_concat(&mut self) -> Result<Ast> {
        let mut items: Vec<Ast> = Vec::new();
        while let Some(c) = self.peek() {
            if c == ')' || c == '|' {
                break;
            }
            let atom = self.parse_quantified()?;
            items.push(atom);
        }
        if items.is_empty() {
            // Empty regex matches empty string — represent as Literal match of nothing
            // We use a concat of zero items => treat as matching empty
            return Ok(Ast::Literal('\0'));
        }
        let mut result = items.remove(0);
        for item in items {
            result = Ast::Concat(Box::new(result), Box::new(item));
        }
        Ok(result)
    }

    fn parse_quantified(&mut self) -> Result<Ast> {
        let atom = self.parse_atom()?;
        match self.peek() {
            Some('*') => {
                self.advance();
                Ok(Ast::Star(Box::new(atom)))
            }
            Some('+') => {
                self.advance();
                Ok(Ast::Plus(Box::new(atom)))
            }
            Some('?') => {
                self.advance();
                Ok(Ast::Quest(Box::new(atom)))
            }
            _ => Ok(atom),
        }
    }

    fn parse_atom(&mut self) -> Result<Ast> {
        match self.peek() {
            None => Err(TextError::InvalidInput(
                "Unexpected end of regex pattern".to_string(),
            )),
            Some('^') => {
                self.advance();
                Ok(Ast::AnchorStart)
            }
            Some('$') => {
                self.advance();
                Ok(Ast::AnchorEnd)
            }
            Some('.') => {
                self.advance();
                Ok(Ast::Dot)
            }
            Some('(') => {
                self.advance(); // consume '('
                let inner = self.parse_alternation()?;
                if self.advance() != Some(')') {
                    return Err(TextError::InvalidInput(
                        "Unmatched '(' in regex pattern".to_string(),
                    ));
                }
                Ok(Ast::Group(Box::new(inner)))
            }
            Some('[') => self.parse_char_class(),
            Some('\\') => self.parse_escape(),
            Some(c) if c == '*' || c == '+' || c == '?' => Err(TextError::InvalidInput(format!(
                "Unexpected quantifier '{}' without preceding element",
                c
            ))),
            Some(c) => {
                self.advance();
                Ok(Ast::Literal(c))
            }
        }
    }

    fn parse_escape(&mut self) -> Result<Ast> {
        self.advance(); // consume '\\'
        match self.advance() {
            None => Err(TextError::InvalidInput(
                "Trailing backslash in regex".to_string(),
            )),
            Some('d') => Ok(Ast::CharClass {
                ranges: vec![('0', '9')],
                negated: false,
            }),
            Some('D') => Ok(Ast::CharClass {
                ranges: vec![('0', '9')],
                negated: true,
            }),
            Some('w') => Ok(Ast::CharClass {
                ranges: vec![('a', 'z'), ('A', 'Z'), ('0', '9'), ('_', '_')],
                negated: false,
            }),
            Some('W') => Ok(Ast::CharClass {
                ranges: vec![('a', 'z'), ('A', 'Z'), ('0', '9'), ('_', '_')],
                negated: true,
            }),
            Some('s') => Ok(Ast::CharClass {
                ranges: vec![(' ', ' '), ('\t', '\t'), ('\n', '\n'), ('\r', '\r')],
                negated: false,
            }),
            Some('S') => Ok(Ast::CharClass {
                ranges: vec![(' ', ' '), ('\t', '\t'), ('\n', '\n'), ('\r', '\r')],
                negated: true,
            }),
            Some('n') => Ok(Ast::Literal('\n')),
            Some('t') => Ok(Ast::Literal('\t')),
            Some('r') => Ok(Ast::Literal('\r')),
            Some(c) => Ok(Ast::Literal(c)), // escaped metacharacters
        }
    }

    fn parse_char_class(&mut self) -> Result<Ast> {
        self.advance(); // consume '['
        let negated = if self.peek() == Some('^') {
            self.advance();
            true
        } else {
            false
        };

        let mut ranges: Vec<(char, char)> = Vec::new();

        // Handle ']' as first character in class (literal)
        if self.peek() == Some(']') {
            self.advance();
            ranges.push((']', ']'));
        }

        loop {
            match self.peek() {
                None => {
                    return Err(TextError::InvalidInput(
                        "Unterminated character class '[' in regex".to_string(),
                    ))
                }
                Some(']') => {
                    self.advance();
                    break;
                }
                Some('\\') => {
                    self.advance(); // consume '\\'
                    let c = self.advance().ok_or_else(|| {
                        TextError::InvalidInput(
                            "Trailing backslash inside character class".to_string(),
                        )
                    })?;
                    let ch = match c {
                        'd' => {
                            ranges.push(('0', '9'));
                            continue;
                        }
                        'w' => {
                            ranges.push(('a', 'z'));
                            ranges.push(('A', 'Z'));
                            ranges.push(('0', '9'));
                            ranges.push(('_', '_'));
                            continue;
                        }
                        's' => {
                            ranges.push((' ', ' '));
                            ranges.push(('\t', '\t'));
                            ranges.push(('\n', '\n'));
                            ranges.push(('\r', '\r'));
                            continue;
                        }
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        other => other,
                    };
                    self.push_range_or_literal(&mut ranges, ch)?;
                }
                Some(c) => {
                    self.advance();
                    self.push_range_or_literal(&mut ranges, c)?;
                }
            }
        }

        Ok(Ast::CharClass { ranges, negated })
    }

    /// If next char is '-' and then another char, make a range; else push literal.
    fn push_range_or_literal(&mut self, ranges: &mut Vec<(char, char)>, start: char) -> Result<()> {
        if self.peek() == Some('-') {
            // peek ahead two: the '-' and the character after
            if self.chars.get(self.pos + 1).copied() == Some(']')
                || self.pos + 1 >= self.chars.len()
            {
                // '-' at end of class is literal
                ranges.push((start, start));
            } else {
                self.advance(); // consume '-'
                let end = self.advance().ok_or_else(|| {
                    TextError::InvalidInput("Unterminated range in character class".to_string())
                })?;
                if end < start {
                    return Err(TextError::InvalidInput(format!(
                        "Invalid character range '{}-{}' in character class",
                        start, end
                    )));
                }
                ranges.push((start, end));
            }
        } else {
            ranges.push((start, start));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// NFA
// ---------------------------------------------------------------------------

/// An NFA state transition.
#[derive(Debug, Clone)]
enum NfaTransition {
    /// Epsilon (empty) transition.
    Epsilon,
    /// Match a specific character.
    Char(char),
    /// Match any character except newline.
    Dot,
    /// Match a character class.
    CharClass {
        ranges: Vec<(char, char)>,
        negated: bool,
    },
    /// Anchor: beginning of string.
    AnchorStart,
    /// Anchor: end of string.
    AnchorEnd,
}

/// An NFA state with up to two outgoing transitions.
#[derive(Debug, Clone)]
struct NfaState {
    transition: NfaTransition,
    out1: Option<usize>,
    out2: Option<usize>,
}

/// The compiled NFA.
struct Nfa {
    states: Vec<NfaState>,
    start: usize,
    accept: usize,
}

/// A fragment used during NFA construction (Thompson construction).
struct Fragment {
    start: usize,
    /// Indices of states whose `out1` or `out2` need to be patched to connect.
    dangling: Vec<(usize, u8)>, // (state_index, which_out: 1 or 2)
}

impl Nfa {
    fn new() -> Self {
        Self {
            states: Vec::new(),
            start: 0,
            accept: 0,
        }
    }

    fn add_state(&mut self, transition: NfaTransition) -> usize {
        let id = self.states.len();
        self.states.push(NfaState {
            transition,
            out1: None,
            out2: None,
        });
        id
    }

    fn patch(&mut self, dangling: &[(usize, u8)], target: usize) {
        for &(state_id, which) in dangling {
            if which == 1 {
                self.states[state_id].out1 = Some(target);
            } else {
                self.states[state_id].out2 = Some(target);
            }
        }
    }

    fn build(ast: &Ast) -> Result<Self> {
        let mut nfa = Nfa::new();
        let frag = nfa.compile(ast)?;
        let accept = nfa.add_state(NfaTransition::Epsilon);
        nfa.patch(&frag.dangling, accept);
        nfa.start = frag.start;
        nfa.accept = accept;
        Ok(nfa)
    }

    fn compile(&mut self, ast: &Ast) -> Result<Fragment> {
        match ast {
            Ast::Literal(c) => {
                let s = self.add_state(NfaTransition::Char(*c));
                Ok(Fragment {
                    start: s,
                    dangling: vec![(s, 1)],
                })
            }
            Ast::Dot => {
                let s = self.add_state(NfaTransition::Dot);
                Ok(Fragment {
                    start: s,
                    dangling: vec![(s, 1)],
                })
            }
            Ast::CharClass { ranges, negated } => {
                let s = self.add_state(NfaTransition::CharClass {
                    ranges: ranges.clone(),
                    negated: *negated,
                });
                Ok(Fragment {
                    start: s,
                    dangling: vec![(s, 1)],
                })
            }
            Ast::AnchorStart => {
                let s = self.add_state(NfaTransition::AnchorStart);
                Ok(Fragment {
                    start: s,
                    dangling: vec![(s, 1)],
                })
            }
            Ast::AnchorEnd => {
                let s = self.add_state(NfaTransition::AnchorEnd);
                Ok(Fragment {
                    start: s,
                    dangling: vec![(s, 1)],
                })
            }
            Ast::Concat(left, right) => {
                let frag1 = self.compile(left)?;
                let frag2 = self.compile(right)?;
                self.patch(&frag1.dangling, frag2.start);
                Ok(Fragment {
                    start: frag1.start,
                    dangling: frag2.dangling,
                })
            }
            Ast::Alt(left, right) => {
                let frag1 = self.compile(left)?;
                let frag2 = self.compile(right)?;
                let split = self.add_state(NfaTransition::Epsilon);
                self.states[split].out1 = Some(frag1.start);
                self.states[split].out2 = Some(frag2.start);
                let mut dangling = frag1.dangling;
                dangling.extend(frag2.dangling);
                Ok(Fragment {
                    start: split,
                    dangling,
                })
            }
            Ast::Star(inner) => {
                let frag = self.compile(inner)?;
                let split = self.add_state(NfaTransition::Epsilon);
                self.states[split].out1 = Some(frag.start);
                self.patch(&frag.dangling, split);
                Ok(Fragment {
                    start: split,
                    dangling: vec![(split, 2)],
                })
            }
            Ast::Plus(inner) => {
                let frag = self.compile(inner)?;
                let split = self.add_state(NfaTransition::Epsilon);
                self.states[split].out1 = Some(frag.start);
                let start = frag.start;
                self.patch(&frag.dangling, split);
                Ok(Fragment {
                    start,
                    dangling: vec![(split, 2)],
                })
            }
            Ast::Quest(inner) => {
                let frag = self.compile(inner)?;
                let split = self.add_state(NfaTransition::Epsilon);
                self.states[split].out1 = Some(frag.start);
                let mut dangling = frag.dangling;
                dangling.push((split, 2));
                Ok(Fragment {
                    start: split,
                    dangling,
                })
            }
            Ast::Group(inner) => self.compile(inner),
        }
    }
}

// ---------------------------------------------------------------------------
// NFA simulation
// ---------------------------------------------------------------------------

/// Add a state (and its epsilon closure) to the state set.
fn add_state_to_set(nfa: &Nfa, state_id: usize, set: &mut Vec<usize>, visited: &mut Vec<bool>) {
    if visited[state_id] {
        return;
    }
    visited[state_id] = true;
    let state = &nfa.states[state_id];
    match &state.transition {
        NfaTransition::Epsilon => {
            // Follow epsilon transitions
            set.push(state_id); // include the epsilon state itself (for accept check)
            if let Some(out1) = state.out1 {
                add_state_to_set(nfa, out1, set, visited);
            }
            if let Some(out2) = state.out2 {
                add_state_to_set(nfa, out2, set, visited);
            }
        }
        _ => {
            set.push(state_id);
        }
    }
}

fn epsilon_closure(nfa: &Nfa, states: &[usize]) -> Vec<usize> {
    let n = nfa.states.len();
    let mut visited = vec![false; n];
    let mut result = Vec::new();
    for &s in states {
        add_state_to_set(nfa, s, &mut result, &mut visited);
    }
    result
}

fn char_matches_class(c: char, ranges: &[(char, char)], negated: bool) -> bool {
    let in_range = ranges.iter().any(|&(lo, hi)| c >= lo && c <= hi);
    if negated {
        !in_range
    } else {
        in_range
    }
}

/// Result of a single NFA simulation starting at a given position.
/// Returns the end position (exclusive) if the NFA accepts.
fn simulate_nfa(
    nfa: &Nfa,
    input: &[char],
    start_pos: usize,
    anchored_start: bool,
    anchored_end: bool,
) -> Option<usize> {
    let mut current = epsilon_closure(nfa, &[nfa.start]);

    // Handle start anchor states
    let mut next_after_anchor: Vec<usize> = Vec::new();
    for &s in &current {
        match &nfa.states[s].transition {
            NfaTransition::AnchorStart => {
                if start_pos == 0 {
                    if let Some(out1) = nfa.states[s].out1 {
                        next_after_anchor.push(out1);
                    }
                }
                // If start_pos != 0, this state is dead
            }
            _ => {
                next_after_anchor.push(s);
            }
        }
    }
    current = epsilon_closure(nfa, &next_after_anchor);

    // Track the longest match position
    let mut last_accept: Option<usize> = None;

    // Check if accept state is reachable before consuming any input
    if current.contains(&nfa.accept) && (!anchored_end || start_pos == input.len()) {
        last_accept = Some(start_pos);
    }

    let mut pos = start_pos;
    while pos < input.len() && !current.is_empty() {
        let ch = input[pos];
        let mut next: Vec<usize> = Vec::new();

        for &s in &current {
            if s == nfa.accept {
                continue; // accept state has no transitions
            }
            let state = &nfa.states[s];
            let matched = match &state.transition {
                NfaTransition::Char(c) => ch == *c,
                NfaTransition::Dot => ch != '\n',
                NfaTransition::CharClass { ranges, negated } => {
                    char_matches_class(ch, ranges, *negated)
                }
                NfaTransition::AnchorEnd => {
                    // End anchor doesn't consume a character
                    // It's satisfied when we're at the end
                    false
                }
                NfaTransition::Epsilon | NfaTransition::AnchorStart => false,
            };
            if matched {
                if let Some(out1) = state.out1 {
                    next.push(out1);
                }
            }
        }

        current = epsilon_closure(nfa, &next);

        // Handle end anchor within the closure
        let mut expanded: Vec<usize> = Vec::new();
        for &s in &current {
            if let NfaTransition::AnchorEnd = &nfa.states[s].transition {
                if pos + 1 == input.len() {
                    if let Some(out1) = nfa.states[s].out1 {
                        expanded.push(out1);
                    }
                }
            } else {
                expanded.push(s);
            }
        }
        if !expanded.is_empty() {
            let extra = epsilon_closure(nfa, &expanded);
            // Merge
            let mut merged = current.clone();
            for s in extra {
                if !merged.contains(&s) {
                    merged.push(s);
                }
            }
            current = merged;
        }

        pos += 1;

        if current.contains(&nfa.accept) && (!anchored_end || pos == input.len()) {
            last_accept = Some(pos);
        }
    }

    // At end of input, check end anchors
    if anchored_end && pos == input.len() {
        for &s in &current {
            if let NfaTransition::AnchorEnd = &nfa.states[s].transition {
                if let Some(out1) = nfa.states[s].out1 {
                    let closure = epsilon_closure(nfa, &[out1]);
                    if closure.contains(&nfa.accept) {
                        last_accept = Some(pos);
                    }
                }
            }
        }
    }

    if anchored_start && start_pos != 0 {
        return None;
    }

    last_accept
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A match result returned by `find` and `find_all`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Match {
    /// Start byte offset in the original string.
    start: usize,
    /// End byte offset (exclusive) in the original string.
    end: usize,
    /// The matched substring.
    text: String,
}

impl Match {
    /// Get the start byte offset.
    pub fn start(&self) -> usize {
        self.start
    }

    /// Get the end byte offset (exclusive).
    pub fn end(&self) -> usize {
        self.end
    }

    /// Get the matched text.
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// Get the length of the match in bytes.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Check if the match is empty.
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// A compiled regular expression backed by an NFA.
///
/// # Example
///
/// ```rust
/// use scirs2_text::regex_lite::RegexLite;
///
/// let re = RegexLite::new(r"[a-z]+\d+").unwrap();
/// assert!(re.is_match("abc123"));
/// assert!(!re.is_match("123"));
/// ```
pub struct RegexLite {
    nfa: Nfa,
    anchored_start: bool,
    anchored_end: bool,
    pattern: String,
}

impl RegexLite {
    /// Compile a regex pattern into an NFA.
    pub fn new(pattern: &str) -> Result<Self> {
        let mut parser = Parser::new(pattern);
        let ast = parser.parse()?;

        // Check for top-level anchors
        let (anchored_start, anchored_end, core_ast) = extract_anchors(ast);

        let nfa = Nfa::build(&core_ast)?;

        Ok(Self {
            nfa,
            anchored_start,
            anchored_end,
            pattern: pattern.to_string(),
        })
    }

    /// Get the original pattern string.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Check whether the pattern matches anywhere in the input string.
    pub fn is_match(&self, input: &str) -> bool {
        let chars: Vec<char> = input.chars().collect();

        if self.anchored_start {
            return simulate_nfa(&self.nfa, &chars, 0, true, self.anchored_end).is_some();
        }

        for start in 0..=chars.len() {
            if simulate_nfa(&self.nfa, &chars, start, false, self.anchored_end).is_some() {
                return true;
            }
        }
        false
    }

    /// Find the first (leftmost-longest) match in the input string.
    pub fn find(&self, input: &str) -> Option<Match> {
        let chars: Vec<char> = input.chars().collect();
        // Build a byte-offset map: char_index -> byte_offset
        let byte_offsets = build_byte_offsets(input, &chars);

        let search_start = 0;
        let search_end = if self.anchored_start {
            1
        } else {
            chars.len() + 1
        };

        for start in search_start..search_end {
            if let Some(end) = simulate_nfa(
                &self.nfa,
                &chars,
                start,
                self.anchored_start,
                self.anchored_end,
            ) {
                let byte_start = byte_offsets[start];
                let byte_end = byte_offsets[end];
                return Some(Match {
                    start: byte_start,
                    end: byte_end,
                    text: input[byte_start..byte_end].to_string(),
                });
            }
        }
        None
    }

    /// Find all non-overlapping matches in the input string.
    pub fn find_all(&self, input: &str) -> Vec<Match> {
        let chars: Vec<char> = input.chars().collect();
        let byte_offsets = build_byte_offsets(input, &chars);
        let mut matches = Vec::new();
        let mut search_from = 0;

        while search_from <= chars.len() {
            let search_end = if self.anchored_start {
                search_from + 1
            } else {
                chars.len() + 1
            };

            let mut next_search_from = search_from + 1;
            for start in search_from..search_end {
                if let Some(end) = simulate_nfa(
                    &self.nfa,
                    &chars,
                    start,
                    self.anchored_start,
                    self.anchored_end,
                ) {
                    let byte_start = byte_offsets[start];
                    let byte_end = byte_offsets[end];
                    matches.push(Match {
                        start: byte_start,
                        end: byte_end,
                        text: input[byte_start..byte_end].to_string(),
                    });
                    // Advance past this match (at least 1 char to avoid infinite loop on empty match)
                    next_search_from = if end > start { end } else { start + 1 };
                    break;
                }
            }
            search_from = next_search_from;

            if self.anchored_start {
                break; // anchored to start, only one match possible
            }
        }
        matches
    }

    /// Replace the first match with a replacement string.
    pub fn replace(&self, input: &str, replacement: &str) -> String {
        match self.find(input) {
            Some(m) => {
                let mut result = String::with_capacity(input.len());
                result.push_str(&input[..m.start()]);
                result.push_str(replacement);
                result.push_str(&input[m.end()..]);
                result
            }
            None => input.to_string(),
        }
    }

    /// Replace all non-overlapping matches with a replacement string.
    pub fn replace_all(&self, input: &str, replacement: &str) -> String {
        let matches = self.find_all(input);
        if matches.is_empty() {
            return input.to_string();
        }

        let mut result = String::with_capacity(input.len());
        let mut last_end = 0;
        for m in &matches {
            result.push_str(&input[last_end..m.start()]);
            result.push_str(replacement);
            last_end = m.end();
        }
        result.push_str(&input[last_end..]);
        result
    }

    /// Split the input string by the pattern.
    pub fn split(&self, input: &str) -> Vec<String> {
        let matches = self.find_all(input);
        if matches.is_empty() {
            return vec![input.to_string()];
        }

        let mut parts = Vec::new();
        let mut last_end = 0;
        for m in &matches {
            parts.push(input[last_end..m.start()].to_string());
            last_end = m.end();
        }
        parts.push(input[last_end..].to_string());
        parts
    }
}

/// Extract top-level anchors from the AST and return (anchored_start, anchored_end, remaining_ast).
fn extract_anchors(ast: Ast) -> (bool, bool, Ast) {
    let mut anchored_start = false;
    let mut anchored_end = false;
    let mut current = ast;

    // Extract leading ^
    if let Ast::Concat(ref left, _) = current {
        if matches!(**left, Ast::AnchorStart) {
            anchored_start = true;
            if let Ast::Concat(_, right) = current {
                current = *right;
            }
        }
    } else if matches!(current, Ast::AnchorStart) {
        // Pattern is just "^"
        anchored_start = true;
        current = Ast::Literal('\0');
    }

    // Extract trailing $
    current = strip_trailing_anchor(current, &mut anchored_end);

    (anchored_start, anchored_end, current)
}

fn strip_trailing_anchor(ast: Ast, anchored_end: &mut bool) -> Ast {
    match ast {
        Ast::AnchorEnd => {
            *anchored_end = true;
            Ast::Literal('\0')
        }
        Ast::Concat(left, right) => {
            if matches!(*right, Ast::AnchorEnd) {
                *anchored_end = true;
                *left
            } else {
                let new_right = strip_trailing_anchor(*right, anchored_end);
                Ast::Concat(left, Box::new(new_right))
            }
        }
        other => other,
    }
}

/// Build a mapping from char index to byte offset.
fn build_byte_offsets(input: &str, chars: &[char]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(chars.len() + 1);
    let mut byte_pos = 0;
    for &c in chars {
        offsets.push(byte_pos);
        byte_pos += c.len_utf8();
    }
    offsets.push(input.len()); // offset for one past the last char
    offsets
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl std::fmt::Debug for RegexLite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegexLite")
            .field("pattern", &self.pattern)
            .field("anchored_start", &self.anchored_start)
            .field("anchored_end", &self.anchored_end)
            .field("states", &self.nfa.states.len())
            .finish()
    }
}

impl std::fmt::Display for RegexLite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RegexLite({})", self.pattern)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_match() {
        let re = RegexLite::new("hello").expect("compile failed");
        assert!(re.is_match("hello world"));
        assert!(re.is_match("say hello"));
        assert!(!re.is_match("helo"));
    }

    #[test]
    fn test_dot() {
        let re = RegexLite::new("h.llo").expect("compile failed");
        assert!(re.is_match("hello"));
        assert!(re.is_match("hxllo"));
        assert!(!re.is_match("hllo"));
    }

    #[test]
    fn test_star() {
        let re = RegexLite::new("ab*c").expect("compile failed");
        assert!(re.is_match("ac"));
        assert!(re.is_match("abc"));
        assert!(re.is_match("abbbbc"));
        assert!(!re.is_match("adc"));
    }

    #[test]
    fn test_plus() {
        let re = RegexLite::new("ab+c").expect("compile failed");
        assert!(!re.is_match("ac"));
        assert!(re.is_match("abc"));
        assert!(re.is_match("abbbbc"));
    }

    #[test]
    fn test_question_mark() {
        let re = RegexLite::new("colou?r").expect("compile failed");
        assert!(re.is_match("color"));
        assert!(re.is_match("colour"));
        assert!(!re.is_match("colouur"));
    }

    #[test]
    fn test_alternation() {
        let re = RegexLite::new("cat|dog").expect("compile failed");
        assert!(re.is_match("I have a cat"));
        assert!(re.is_match("I have a dog"));
        assert!(!re.is_match("I have a fish"));
    }

    #[test]
    fn test_grouping() {
        let re = RegexLite::new("^(ab)+$").expect("compile failed");
        assert!(re.is_match("ab"));
        assert!(re.is_match("ababab"));
        assert!(!re.is_match("aabb"));
    }

    #[test]
    fn test_char_class() {
        let re = RegexLite::new("[aeiou]+").expect("compile failed");
        assert!(re.is_match("hello"));
        assert!(!re.is_match("rhythm"));
    }

    #[test]
    fn test_char_class_range() {
        let re = RegexLite::new("[a-z]+").expect("compile failed");
        assert!(re.is_match("hello"));
        assert!(!re.is_match("12345"));
    }

    #[test]
    fn test_negated_char_class() {
        let re = RegexLite::new("[^0-9]+").expect("compile failed");
        assert!(re.is_match("hello"));
        assert!(!re.is_match("12345"));
    }

    #[test]
    fn test_digit_shortcut() {
        let re = RegexLite::new(r"\d+").expect("compile failed");
        assert!(re.is_match("abc123"));
        assert!(!re.is_match("abcdef"));
    }

    #[test]
    fn test_word_shortcut() {
        let re = RegexLite::new(r"\w+").expect("compile failed");
        assert!(re.is_match("hello_world"));
        assert!(re.is_match("test123"));
    }

    #[test]
    fn test_whitespace_shortcut() {
        let re = RegexLite::new(r"\s+").expect("compile failed");
        assert!(re.is_match("hello world"));
        assert!(!re.is_match("helloworld"));
    }

    #[test]
    fn test_anchor_start() {
        let re = RegexLite::new("^hello").expect("compile failed");
        assert!(re.is_match("hello world"));
        assert!(!re.is_match("say hello"));
    }

    #[test]
    fn test_anchor_end() {
        let re = RegexLite::new("world$").expect("compile failed");
        assert!(re.is_match("hello world"));
        assert!(!re.is_match("world hello"));
    }

    #[test]
    fn test_both_anchors() {
        let re = RegexLite::new("^hello$").expect("compile failed");
        assert!(re.is_match("hello"));
        assert!(!re.is_match("hello world"));
        assert!(!re.is_match("say hello"));
    }

    #[test]
    fn test_find() {
        let re = RegexLite::new(r"\d+").expect("compile failed");
        let m = re.find("price: 99 dollars");
        assert!(m.is_some());
        let m = m.expect("should have match");
        assert_eq!(m.as_str(), "99");
        assert_eq!(m.start(), 7);
        assert_eq!(m.end(), 9);
    }

    #[test]
    fn test_find_all() {
        let re = RegexLite::new(r"\d+").expect("compile failed");
        let matches = re.find_all("1 and 22 and 333");
        let texts: Vec<&str> = matches.iter().map(|m| m.as_str()).collect();
        assert_eq!(texts, vec!["1", "22", "333"]);
    }

    #[test]
    fn test_replace() {
        let re = RegexLite::new(r"\d+").expect("compile failed");
        assert_eq!(re.replace("item 42 costs 99", "NUM"), "item NUM costs 99");
    }

    #[test]
    fn test_replace_all() {
        let re = RegexLite::new(r"\d+").expect("compile failed");
        assert_eq!(
            re.replace_all("item 42 costs 99", "NUM"),
            "item NUM costs NUM"
        );
    }

    #[test]
    fn test_split() {
        let re = RegexLite::new(r"\s+").expect("compile failed");
        let parts = re.split("hello  world   test");
        assert_eq!(parts, vec!["hello", "world", "test"]);
    }

    #[test]
    fn test_escaped_metacharacters() {
        let re = RegexLite::new(r"\.\*\+\?").expect("compile failed");
        assert!(re.is_match(".*+?"));
        assert!(!re.is_match("abcd"));
    }

    #[test]
    fn test_complex_pattern() {
        let re = RegexLite::new(r"[a-zA-Z_]\w*").expect("compile failed");
        let m = re.find("  my_var123  ");
        assert!(m.is_some());
        assert_eq!(m.expect("should match").as_str(), "my_var123");
    }

    #[test]
    fn test_email_like_pattern() {
        let re = RegexLite::new(r"\w+@\w+\.\w+").expect("compile failed");
        assert!(re.is_match("user@example.com"));
        assert!(!re.is_match("not an email"));
    }

    #[test]
    fn test_empty_match() {
        let re = RegexLite::new("a*").expect("compile failed");
        // a* matches empty string at start
        assert!(re.is_match("bbb"));
    }

    #[test]
    fn test_unicode_input() {
        let re = RegexLite::new("hello").expect("compile failed");
        assert!(re.is_match("greetings: hello!"));
    }

    #[test]
    fn test_non_digit_shortcut() {
        let re = RegexLite::new(r"\D+").expect("compile failed");
        assert!(re.is_match("hello"));
        assert!(!re.is_match("123"));
    }

    #[test]
    fn test_non_word_shortcut() {
        let re = RegexLite::new(r"\W+").expect("compile failed");
        assert!(re.is_match("!!??"));
        assert!(!re.is_match("hello"));
    }

    #[test]
    fn test_non_whitespace_shortcut() {
        let re = RegexLite::new(r"\S+").expect("compile failed");
        let m = re.find("  hello  ");
        assert!(m.is_some());
        assert_eq!(m.expect("match").as_str(), "hello");
    }

    #[test]
    fn test_nested_groups() {
        let re = RegexLite::new("((ab)+c)+").expect("compile failed");
        assert!(re.is_match("abcababc"));
    }

    #[test]
    fn test_match_len() {
        let re = RegexLite::new(r"\d+").expect("compile failed");
        let m = re.find("abc12345xyz");
        let m = m.expect("should match");
        assert_eq!(m.len(), 5);
        assert!(!m.is_empty());
    }

    #[test]
    fn test_invalid_pattern_unmatched_paren() {
        let result = RegexLite::new("(abc");
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_pattern_bad_range() {
        let result = RegexLite::new("[z-a]");
        assert!(result.is_err());
    }

    #[test]
    fn test_display() {
        let re = RegexLite::new(r"\d+").expect("compile failed");
        let display = format!("{}", re);
        assert!(display.contains(r"\d+"));
    }

    #[test]
    fn test_debug() {
        let re = RegexLite::new(r"\d+").expect("compile failed");
        let debug = format!("{:?}", re);
        assert!(debug.contains("RegexLite"));
    }

    #[test]
    fn test_find_all_no_matches() {
        let re = RegexLite::new(r"\d+").expect("compile failed");
        let matches = re.find_all("no numbers here");
        assert!(matches.is_empty());
    }

    #[test]
    fn test_alternation_in_group() {
        let re = RegexLite::new("(cat|dog)s?").expect("compile failed");
        assert!(re.is_match("cats"));
        assert!(re.is_match("dog"));
        assert!(re.is_match("dogs"));
    }

    #[test]
    fn test_character_class_with_shortcuts() {
        let re = RegexLite::new(r"[\d\w]+").expect("compile failed");
        assert!(re.is_match("abc123"));
    }

    #[test]
    fn test_escaped_backslash() {
        let re = RegexLite::new(r"a\\b").expect("compile failed");
        assert!(re.is_match(r"a\b"));
    }

    #[test]
    fn test_tab_and_newline() {
        let re = RegexLite::new(r"\t\n").expect("compile failed");
        assert!(re.is_match("\t\n"));
    }

    #[test]
    fn test_pattern_accessor() {
        let re = RegexLite::new(r"\d+").expect("compile failed");
        assert_eq!(re.pattern(), r"\d+");
    }

    #[test]
    fn test_replace_no_match() {
        let re = RegexLite::new(r"\d+").expect("compile failed");
        assert_eq!(re.replace("no numbers", "NUM"), "no numbers");
    }

    #[test]
    fn test_split_no_match() {
        let re = RegexLite::new(r"\d+").expect("compile failed");
        let parts = re.split("hello world");
        assert_eq!(parts, vec!["hello world"]);
    }

    #[test]
    fn test_dot_does_not_match_newline() {
        let re = RegexLite::new("a.b").expect("compile failed");
        assert!(!re.is_match("a\nb"));
        assert!(re.is_match("axb"));
    }

    #[test]
    fn test_complex_alternation() {
        let re = RegexLite::new("a|b|c").expect("compile failed");
        assert!(re.is_match("a"));
        assert!(re.is_match("b"));
        assert!(re.is_match("c"));
        assert!(!re.is_match("d"));
    }
}
