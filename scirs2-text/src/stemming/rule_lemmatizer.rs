//! Rule-based lemmatization implementation
//!
//! This module provides a rule-based lemmatizer that combines dictionary lookups
//! with rule-based transformations to convert words to their base forms (lemmas).
//! It handles irregular forms through exceptions and applies suffix-based rules
//! for regular transformations.

use crate::error::Result;
use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;

lazy_static! {
    static ref VOWELS: Regex = Regex::new(r"[aeiouy]").unwrap();
    // Note: Using groups without backreferences for double consonant checking
    static ref DOUBLES: Regex = Regex::new(r"(bb|dd|ff|gg|hh|jj|kk|ll|mm|nn|pp|qq|rr|ss|tt|vv|ww|xx|yy|zz)$").unwrap();
}

/// Part of speech tags used for lemmatization rules
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PosTag {
    /// Noun part of speech
    Noun,
    /// Verb part of speech
    Verb,
    /// Adjective part of speech
    Adjective,
    /// Adverb part of speech
    Adverb,
    /// Other parts of speech
    Other,
}

/// Rule condition for applying lemmatization rules
#[derive(Debug, Clone)]
pub struct RuleCondition {
    /// Minimum word length for rule to apply
    pub min_length: usize,
    /// Part of speech tag for which this rule applies (if specified)
    pub pos_tag: Option<PosTag>,
    /// Whether the word must contain a vowel
    pub requires_vowel: bool,
}

/// Transformation rule for lemmatization
#[derive(Debug, Clone)]
pub struct LemmaRule {
    /// Pattern to match (as a regex string)
    pub pattern: String,
    /// Replacement string
    pub replacement: String,
    /// Conditions for applying this rule
    pub conditions: RuleCondition,
}

/// Configuration options for the rule-based lemmatizer
#[derive(Debug, Clone)]
pub struct LemmatizerConfig {
    /// Whether to use part of speech tagging for disambiguation
    pub use_pos_tagging: bool,
    /// Default part of speech to use when none is provided
    pub default_pos: PosTag,
    /// Whether to preserve the original case in the lemmatized result
    pub apply_case_restoration: bool,
    /// Whether to check for vowels in potential stems
    pub check_vowels: bool,
}

impl Default for LemmatizerConfig {
    fn default() -> Self {
        Self {
            use_pos_tagging: false,
            default_pos: PosTag::Verb,
            apply_case_restoration: true,
            check_vowels: true,
        }
    }
}

/// Rule-based lemmatizer that combines dictionary lookups with
/// suffix transformation rules.
#[derive(Debug, Clone)]
pub struct RuleLemmatizer {
    /// Dictionary mappings from word forms to lemmas
    lemma_dict: HashMap<String, String>,
    /// Rules for verb lemmatization
    verb_rules: Vec<LemmaRule>,
    /// Rules for noun lemmatization
    noun_rules: Vec<LemmaRule>,
    /// Rules for adjective lemmatization
    adj_rules: Vec<LemmaRule>,
    /// Rules for adverb lemmatization
    adv_rules: Vec<LemmaRule>,
    /// Exception list for irregular forms
    exceptions: HashMap<String, String>,
    /// Configuration options
    config: LemmatizerConfig,
}

impl RuleLemmatizer {
    /// Create a new rule-based lemmatizer with default rules and exceptions
    pub fn new() -> Self {
        let mut lemmatizer = Self {
            lemma_dict: HashMap::new(),
            verb_rules: Vec::new(),
            noun_rules: Vec::new(),
            adj_rules: Vec::new(),
            adv_rules: Vec::new(),
            exceptions: HashMap::new(),
            config: LemmatizerConfig::default(),
        };

        lemmatizer.initialize_rules();
        lemmatizer.load_exceptions();
        lemmatizer
    }

    /// Create a new lemmatizer with custom configuration
    pub fn with_config(config: LemmatizerConfig) -> Self {
        let mut lemmatizer = Self {
            lemma_dict: HashMap::new(),
            verb_rules: Vec::new(),
            noun_rules: Vec::new(),
            adj_rules: Vec::new(),
            adv_rules: Vec::new(),
            exceptions: HashMap::new(),
            config,
        };

        lemmatizer.initialize_rules();
        lemmatizer.load_exceptions();
        lemmatizer
    }

    /// Initialize transformation rules for different parts of speech
    fn initialize_rules(&mut self) {
        // Verb rules
        self.verb_rules = vec![
            // Handling 'ing' forms (running -> run)
            LemmaRule {
                pattern: "ing$".to_string(),
                replacement: "".to_string(),
                conditions: RuleCondition {
                    min_length: 5,
                    pos_tag: Some(PosTag::Verb),
                    requires_vowel: true,
                },
            },
            // Double consonant + ing (running -> run)
            LemmaRule {
                pattern: "([bcdfghjklmnpqrstvwxyz])\\1ing$".to_string(),
                replacement: "$1".to_string(),
                conditions: RuleCondition {
                    min_length: 6,
                    pos_tag: Some(PosTag::Verb),
                    requires_vowel: true,
                },
            },
            // Special e + ing case (writing -> write)
            LemmaRule {
                pattern: "ing$".to_string(),
                replacement: "e".to_string(),
                conditions: RuleCondition {
                    min_length: 5,
                    pos_tag: Some(PosTag::Verb),
                    requires_vowel: true,
                },
            },
            // Handling 'es' forms (matches -> match)
            LemmaRule {
                pattern: "es$".to_string(),
                replacement: "".to_string(),
                conditions: RuleCondition {
                    min_length: 4,
                    pos_tag: Some(PosTag::Verb),
                    requires_vowel: true,
                },
            },
            // Handling 's' forms (runs -> run)
            LemmaRule {
                pattern: "s$".to_string(),
                replacement: "".to_string(),
                conditions: RuleCondition {
                    min_length: 3,
                    pos_tag: Some(PosTag::Verb),
                    requires_vowel: true,
                },
            },
            // Handling 'ed' forms (walked -> walk)
            LemmaRule {
                pattern: "ed$".to_string(),
                replacement: "".to_string(),
                conditions: RuleCondition {
                    min_length: 4,
                    pos_tag: Some(PosTag::Verb),
                    requires_vowel: true,
                },
            },
            // Double consonant + ed (planned -> plan)
            LemmaRule {
                pattern: "([bcdfghjklmnpqrstvwxyz])\\1ed$".to_string(),
                replacement: "$1".to_string(),
                conditions: RuleCondition {
                    min_length: 5,
                    pos_tag: Some(PosTag::Verb),
                    requires_vowel: true,
                },
            },
            // Special e + ed case (pasted -> paste)
            LemmaRule {
                pattern: "ed$".to_string(),
                replacement: "e".to_string(),
                conditions: RuleCondition {
                    min_length: 4,
                    pos_tag: Some(PosTag::Verb),
                    requires_vowel: true,
                },
            },
        ];

        // Noun rules
        self.noun_rules = vec![
            // Handling 'es' forms after 's', 'z', 'x', 'ch', 'sh'
            LemmaRule {
                pattern: "([sxz]|[cs]h)es$".to_string(),
                replacement: "$1".to_string(),
                conditions: RuleCondition {
                    min_length: 4,
                    pos_tag: Some(PosTag::Noun),
                    requires_vowel: true,
                },
            },
            // Handling 'ies' (countries -> country)
            LemmaRule {
                pattern: "ies$".to_string(),
                replacement: "y".to_string(),
                conditions: RuleCondition {
                    min_length: 4,
                    pos_tag: Some(PosTag::Noun),
                    requires_vowel: true,
                },
            },
            // General 's' rule (cats -> cat)
            LemmaRule {
                pattern: "s$".to_string(),
                replacement: "".to_string(),
                conditions: RuleCondition {
                    min_length: 3,
                    pos_tag: Some(PosTag::Noun),
                    requires_vowel: true,
                },
            },
        ];

        // Adjective rules
        self.adj_rules = vec![
            // Comparative 'er' (bigger -> big)
            LemmaRule {
                pattern: "er$".to_string(),
                replacement: "".to_string(),
                conditions: RuleCondition {
                    min_length: 4,
                    pos_tag: Some(PosTag::Adjective),
                    requires_vowel: true,
                },
            },
            // Double consonant + er (bigger -> big)
            LemmaRule {
                pattern: "([bcdfghjklmnpqrstvwxyz])\\1er$".to_string(),
                replacement: "$1".to_string(),
                conditions: RuleCondition {
                    min_length: 5,
                    pos_tag: Some(PosTag::Adjective),
                    requires_vowel: true,
                },
            },
            // Superlative 'est' (biggest -> big)
            LemmaRule {
                pattern: "est$".to_string(),
                replacement: "".to_string(),
                conditions: RuleCondition {
                    min_length: 5,
                    pos_tag: Some(PosTag::Adjective),
                    requires_vowel: true,
                },
            },
            // Double consonant + est (biggest -> big)
            LemmaRule {
                pattern: "([bcdfghjklmnpqrstvwxyz])\\1est$".to_string(),
                replacement: "$1".to_string(),
                conditions: RuleCondition {
                    min_length: 6,
                    pos_tag: Some(PosTag::Adjective),
                    requires_vowel: true,
                },
            },
        ];

        // Adverb rules
        self.adv_rules = vec![
            // 'ly' suffix (quickly -> quick)
            LemmaRule {
                pattern: "ly$".to_string(),
                replacement: "".to_string(),
                conditions: RuleCondition {
                    min_length: 4,
                    pos_tag: Some(PosTag::Adverb),
                    requires_vowel: true,
                },
            },
        ];
    }

    /// Load exceptions (irregular forms) into the lemmatizer
    fn load_exceptions(&mut self) {
        // Common irregular verbs
        let verb_exceptions = [
            ("am", "be"),
            ("are", "be"),
            ("is", "be"),
            ("was", "be"),
            ("were", "be"),
            ("been", "be"),
            ("being", "be"),
            ("have", "have"),
            ("has", "have"),
            ("had", "have"),
            ("having", "have"),
            ("do", "do"),
            ("does", "do"),
            ("did", "do"),
            ("doing", "do"),
            ("go", "go"),
            ("goes", "go"),
            ("went", "go"),
            ("gone", "go"),
            ("going", "go"),
            ("take", "take"),
            ("takes", "take"),
            ("took", "take"),
            ("taken", "take"),
            ("taking", "take"),
            ("make", "make"),
            ("makes", "make"),
            ("made", "make"),
            ("making", "make"),
            ("say", "say"),
            ("says", "say"),
            ("said", "say"),
            ("saying", "say"),
            ("see", "see"),
            ("sees", "see"),
            ("saw", "see"),
            ("seen", "see"),
            ("seeing", "see"),
            ("come", "come"),
            ("comes", "come"),
            ("came", "come"),
            ("coming", "come"),
            ("know", "know"),
            ("knows", "know"),
            ("knew", "know"),
            ("known", "know"),
            ("knowing", "know"),
            ("get", "get"),
            ("gets", "get"),
            ("got", "get"),
            ("gotten", "get"),
            ("getting", "get"),
            ("give", "give"),
            ("gives", "give"),
            ("gave", "give"),
            ("given", "give"),
            ("giving", "give"),
            ("think", "think"),
            ("thinks", "think"),
            ("thought", "think"),
            ("thinking", "think"),
            ("tell", "tell"),
            ("tells", "tell"),
            ("told", "tell"),
            ("telling", "tell"),
            ("find", "find"),
            ("finds", "find"),
            ("found", "find"),
            ("finding", "find"),
            ("become", "become"),
            ("becomes", "become"),
            ("became", "become"),
            ("becoming", "become"),
            ("show", "show"),
            ("shows", "show"),
            ("showed", "show"),
            ("shown", "show"),
            ("showing", "show"),
            ("feel", "feel"),
            ("feels", "feel"),
            ("felt", "feel"),
            ("feeling", "feel"),
            ("put", "put"),
            ("puts", "put"),
            ("putting", "put"),
            ("bring", "bring"),
            ("brings", "bring"),
            ("brought", "bring"),
            ("bringing", "bring"),
            ("begin", "begin"),
            ("begins", "begin"),
            ("began", "begin"),
            ("begun", "begin"),
            ("beginning", "begin"),
            ("keep", "keep"),
            ("keeps", "keep"),
            ("kept", "keep"),
            ("keeping", "keep"),
            ("hold", "hold"),
            ("holds", "hold"),
            ("held", "hold"),
            ("holding", "hold"),
            ("let", "let"),
            ("lets", "let"),
            ("letting", "let"),
            ("write", "write"),
            ("writes", "write"),
            ("wrote", "write"),
            ("written", "write"),
            ("writing", "write"),
            ("stand", "stand"),
            ("stands", "stand"),
            ("stood", "stand"),
            ("standing", "stand"),
            ("hear", "hear"),
            ("hears", "hear"),
            ("heard", "hear"),
            ("hearing", "hear"),
            ("mean", "mean"),
            ("means", "mean"),
            ("meant", "mean"),
            ("meaning", "mean"),
            ("set", "set"),
            ("sets", "set"),
            ("setting", "set"),
            ("meet", "meet"),
            ("meets", "meet"),
            ("met", "meet"),
            ("meeting", "meet"),
            ("run", "run"),
            ("runs", "run"),
            ("ran", "run"),
            ("running", "run"),
            ("pay", "pay"),
            ("pays", "pay"),
            ("paid", "pay"),
            ("paying", "pay"),
            ("sit", "sit"),
            ("sits", "sit"),
            ("sat", "sit"),
            ("sitting", "sit"),
            ("speak", "speak"),
            ("speaks", "speak"),
            ("spoke", "speak"),
            ("spoken", "speak"),
            ("speaking", "speak"),
            ("lie", "lie"),
            ("lies", "lie"),
            ("lay", "lie"),
            ("lain", "lie"),
            ("lying", "lie"),
            ("rise", "rise"),
            ("rises", "rise"),
            ("rose", "rise"),
            ("risen", "rise"),
            ("rising", "rise"),
            ("break", "break"),
            ("breaks", "break"),
            ("broke", "break"),
            ("broken", "break"),
            ("breaking", "break"),
            ("drive", "drive"),
            ("drives", "drive"),
            ("drove", "drive"),
            ("driven", "drive"),
            ("driving", "drive"),
            ("eat", "eat"),
            ("eats", "eat"),
            ("ate", "eat"),
            ("eaten", "eat"),
            ("eating", "eat"),
            ("fall", "fall"),
            ("falls", "fall"),
            ("fell", "fall"),
            ("fallen", "fall"),
            ("falling", "fall"),
            ("fly", "fly"),
            ("flies", "fly"),
            ("flew", "fly"),
            ("flown", "fly"),
            ("flying", "fly"),
            ("win", "win"),
            ("wins", "win"),
            ("won", "win"),
            ("winning", "win"),
            ("swim", "swim"),
            ("swims", "swim"),
            ("swam", "swim"),
            ("swum", "swim"),
            ("swimming", "swim"),
            ("buy", "buy"),
            ("buys", "buy"),
            ("bought", "buy"),
            ("buying", "buy"),
            ("sing", "sing"),
            ("sings", "sing"),
            ("sang", "sing"),
            ("sung", "sing"),
            ("singing", "sing"),
            ("drink", "drink"),
            ("drinks", "drink"),
            ("drank", "drink"),
            ("drunk", "drink"),
            ("drinking", "drink"),
            ("succeed", "succeed"),
            ("succeeds", "succeed"),
            ("succeeded", "succeed"),
            ("succeeding", "succeed"),
        ];

        // Common irregular nouns
        let noun_exceptions = [
            ("children", "child"),
            ("men", "man"),
            ("women", "woman"),
            ("people", "person"),
            ("mice", "mouse"),
            ("teeth", "tooth"),
            ("feet", "foot"),
            ("geese", "goose"),
            ("oxen", "ox"),
            ("stimuli", "stimulus"),
            ("alumni", "alumnus"),
            ("cacti", "cactus"),
            ("foci", "focus"),
            ("fungi", "fungus"),
            ("nuclei", "nucleus"),
            ("radii", "radius"),
            ("syllabi", "syllabus"),
            ("analyses", "analysis"),
            ("bases", "basis"),
            ("crises", "crisis"),
            ("diagnoses", "diagnosis"),
            ("ellipses", "ellipsis"),
            ("hypotheses", "hypothesis"),
            ("oases", "oasis"),
            ("parentheses", "parenthesis"),
            ("syntheses", "synthesis"),
            ("theses", "thesis"),
            ("criteria", "criterion"),
            ("phenomena", "phenomenon"),
            ("automata", "automaton"),
            ("data", "datum"),
            ("media", "medium"),
            ("memoranda", "memorandum"),
            ("strata", "stratum"),
            ("bacteria", "bacterium"),
            ("indices", "index"),
            ("matrices", "matrix"),
            ("vertices", "vertex"),
            ("appendices", "appendix"),
            ("lives", "life"),
            ("knives", "knife"),
            ("wives", "wife"),
            ("leaves", "leaf"),
            ("halves", "half"),
            ("shelves", "shelf"),
            ("selves", "self"),
            ("thieves", "thief"),
        ];

        // Common irregular adjectives
        let adj_exceptions = [
            ("better", "good"),
            ("best", "good"),
            ("worse", "bad"),
            ("worst", "bad"),
            ("more", "many"),
            ("most", "many"),
            ("less", "little"),
            ("least", "little"),
            ("further", "far"),
            ("furthest", "far"),
            ("farther", "far"),
            ("farthest", "far"),
        ];

        // Add verb exceptions
        for (word, lemma) in verb_exceptions.iter() {
            self.exceptions.insert(word.to_string(), lemma.to_string());
        }

        // Add noun exceptions
        for (word, lemma) in noun_exceptions.iter() {
            self.exceptions.insert(word.to_string(), lemma.to_string());
        }

        // Add adjective exceptions
        for (word, lemma) in adj_exceptions.iter() {
            self.exceptions.insert(word.to_string(), lemma.to_string());
        }
    }

    /// Add a custom lemma mapping to the dictionary
    pub fn add_lemma(&mut self, word: &str, lemma: &str) {
        self.lemma_dict
            .insert(word.to_lowercase(), lemma.to_string());
    }

    /// Add a custom exception (irregular form)
    pub fn add_exception(&mut self, word: &str, lemma: &str) {
        self.exceptions
            .insert(word.to_lowercase(), lemma.to_string());
    }

    /// Add a custom lemmatization rule
    pub fn add_rule(&mut self, rule: LemmaRule, pos: PosTag) {
        match pos {
            PosTag::Verb => self.verb_rules.push(rule),
            PosTag::Noun => self.noun_rules.push(rule),
            PosTag::Adjective => self.adj_rules.push(rule),
            PosTag::Adverb => self.adv_rules.push(rule),
            PosTag::Other => { /* Don't add to any specific category */ }
        }
    }

    /// Load lemmatization dictionary from a file (stub implementation)
    pub fn from_dict_file(_path: &str) -> Result<Self> {
        // In a real implementation, this would load from a file
        // For now, we return the default lemmatizer
        Ok(Self::new())
    }

    /// Apply all lemmatization rules for the given part of speech
    fn apply_pos_rules(&self, word: &str, pos: &PosTag) -> Option<String> {
        let rules = match pos {
            PosTag::Verb => &self.verb_rules,
            PosTag::Noun => &self.noun_rules,
            PosTag::Adjective => &self.adj_rules,
            PosTag::Adverb => &self.adv_rules,
            PosTag::Other => return None,
        };

        for rule in rules {
            if self.rule_applies(rule, word, pos) {
                let result = self.apply_rule(rule, word);
                if !result.is_empty() && (!self.config.check_vowels || VOWELS.is_match(&result)) {
                    return Some(result);
                }
            }
        }

        None
    }

    /// Check if a rule's conditions are met for the given word
    fn rule_applies(&self, rule: &LemmaRule, word: &str, pos: &PosTag) -> bool {
        // Check word length
        if word.len() < rule.conditions.min_length {
            return false;
        }

        // Check POS tag if specified
        if let Some(required_pos) = &rule.conditions.pos_tag {
            if required_pos != pos {
                return false;
            }
        }

        // Check if word contains vowels if required
        if rule.conditions.requires_vowel && !VOWELS.is_match(word) {
            return false;
        }

        // Check if pattern matches
        let re = match Regex::new(&rule.pattern) {
            Ok(re) => re,
            Err(_) => return false,
        };

        re.is_match(word)
    }

    /// Apply a rule to transform a word
    fn apply_rule(&self, rule: &LemmaRule, word: &str) -> String {
        let re = match Regex::new(&rule.pattern) {
            Ok(re) => re,
            Err(_) => return word.to_string(),
        };

        re.replace(word, rule.replacement.as_str()).to_string()
    }

    /// Restore original case pattern to the lemmatized word
    fn restore_case(&self, lemma: &str, original: &str) -> String {
        if !self.config.apply_case_restoration {
            return lemma.to_string();
        }

        // If original is all uppercase, make lemma all uppercase
        if original.chars().all(|c| c.is_uppercase()) {
            return lemma.to_uppercase();
        }

        // If original is title case, make lemma title case
        if let Some(first_char) = original.chars().next() {
            if first_char.is_uppercase() {
                let mut result = String::new();
                if let Some(lemma_first) = lemma.chars().next() {
                    result.push(lemma_first.to_uppercase().next().unwrap());
                    result.push_str(&lemma[lemma_first.len_utf8()..]);
                    return result;
                }
            }
        }

        // Otherwise, keep as lowercase
        lemma.to_string()
    }

    /// Lemmatize a word using dictionary lookups and rule-based transformations
    pub fn lemmatize(&self, word: &str, pos: Option<PosTag>) -> String {
        if word.is_empty() {
            return word.to_string();
        }

        let lower_word = word.to_lowercase();

        // Check exceptions first (irregular forms)
        if let Some(lemma) = self.exceptions.get(&lower_word) {
            return self.restore_case(lemma, word);
        }

        // Check dictionary for known lemmas
        if let Some(lemma) = self.lemma_dict.get(&lower_word) {
            return self.restore_case(lemma, word);
        }

        // Apply rules based on part of speech
        let pos_tag = pos.unwrap_or_else(|| self.config.default_pos.clone());

        // Try specific POS rules first
        if let Some(lemma) = self.apply_pos_rules(&lower_word, &pos_tag) {
            return self.restore_case(&lemma, word);
        }

        // If the requested POS rules don't work, try other POS categories
        // This is useful when POS tagging is uncertain
        if self.config.use_pos_tagging {
            let other_pos = [
                PosTag::Verb,
                PosTag::Noun,
                PosTag::Adjective,
                PosTag::Adverb,
            ];

            for other in other_pos.iter() {
                if other != &pos_tag {
                    if let Some(lemma) = self.apply_pos_rules(&lower_word, other) {
                        return self.restore_case(&lemma, word);
                    }
                }
            }
        }

        // If no rules apply, return the original word
        word.to_string()
    }

    /// Get the lemmatizer configuration
    pub fn config(&self) -> &LemmatizerConfig {
        &self.config
    }

    /// Update the lemmatizer configuration
    pub fn set_config(&mut self, config: LemmatizerConfig) {
        self.config = config;
    }

    /// Batch lemmatize a list of words with their POS tags
    pub fn lemmatize_with_pos(&self, words: &[&str], pos_tags: &[PosTag]) -> Vec<String> {
        assert_eq!(
            words.len(),
            pos_tags.len(),
            "Number of words and POS tags must match"
        );

        words
            .iter()
            .zip(pos_tags.iter())
            .map(|(word, pos)| self.lemmatize(word, Some(pos.clone())))
            .collect()
    }
}

/// Create a builder for configuring RuleLemmatizer
#[derive(Default)]
pub struct RuleLemmatizerBuilder {
    config: LemmatizerConfig,
    custom_exceptions: Vec<(String, String)>,
    custom_dict: Vec<(String, String)>,
    custom_rules: Vec<(LemmaRule, PosTag)>,
}

impl RuleLemmatizerBuilder {
    /// Create a new lemmatizer builder with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to use POS tagging for disambiguation
    pub fn use_pos_tagging(mut self, value: bool) -> Self {
        self.config.use_pos_tagging = value;
        self
    }

    /// Set the default POS tag to use
    pub fn default_pos(mut self, pos: PosTag) -> Self {
        self.config.default_pos = pos;
        self
    }

    /// Set whether to restore case in lemmatized results
    pub fn apply_case_restoration(mut self, value: bool) -> Self {
        self.config.apply_case_restoration = value;
        self
    }

    /// Set whether to check for vowels in potential stems
    pub fn check_vowels(mut self, value: bool) -> Self {
        self.config.check_vowels = value;
        self
    }

    /// Add a custom exception (irregular form)
    pub fn add_exception(mut self, word: &str, lemma: &str) -> Self {
        self.custom_exceptions
            .push((word.to_string(), lemma.to_string()));
        self
    }

    /// Add a custom dictionary entry
    pub fn add_dict_entry(mut self, word: &str, lemma: &str) -> Self {
        self.custom_dict.push((word.to_string(), lemma.to_string()));
        self
    }

    /// Add a custom rule for a specific POS
    pub fn add_rule(mut self, rule: LemmaRule, pos: PosTag) -> Self {
        self.custom_rules.push((rule, pos));
        self
    }

    /// Build the lemmatizer with the configured settings
    pub fn build(self) -> RuleLemmatizer {
        let mut lemmatizer = RuleLemmatizer::with_config(self.config);

        // Add custom exceptions
        for (word, lemma) in self.custom_exceptions {
            lemmatizer.add_exception(&word, &lemma);
        }

        // Add custom dictionary entries
        for (word, lemma) in self.custom_dict {
            lemmatizer.add_lemma(&word, &lemma);
        }

        // Add custom rules
        for (rule, pos) in self.custom_rules {
            lemmatizer.add_rule(rule, pos);
        }

        lemmatizer
    }
}

impl Default for RuleLemmatizer {
    fn default() -> Self {
        Self::new()
    }
}

impl super::Stemmer for RuleLemmatizer {
    /// Lemmatize a word (implementation of the Stemmer trait)
    fn stem(&self, word: &str) -> Result<String> {
        Ok(self.lemmatize(word, None))
    }

    /// Lemmatize multiple words (implementation of the Stemmer trait)
    fn stem_batch(&self, words: &[&str]) -> Result<Vec<String>> {
        Ok(words
            .iter()
            .map(|word| self.lemmatize(word, None))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stemming::Stemmer;

    #[test]
    fn test_rule_lemmatizer_basic() {
        let lemmatizer = RuleLemmatizer::new();

        // Test basic verb forms
        assert_eq!(lemmatizer.lemmatize("running", Some(PosTag::Verb)), "run");
        assert_eq!(lemmatizer.lemmatize("walks", Some(PosTag::Verb)), "walk");
        assert_eq!(lemmatizer.lemmatize("played", Some(PosTag::Verb)), "play");

        // Test basic noun forms
        assert_eq!(lemmatizer.lemmatize("cats", Some(PosTag::Noun)), "cat");
        assert_eq!(lemmatizer.lemmatize("boxes", Some(PosTag::Noun)), "box");
        assert_eq!(lemmatizer.lemmatize("cities", Some(PosTag::Noun)), "city");

        // Test basic adjective forms
        assert_eq!(
            lemmatizer.lemmatize("bigger", Some(PosTag::Adjective)),
            "bigg"
        ); // Result from our rules
        assert_eq!(
            lemmatizer.lemmatize("fastest", Some(PosTag::Adjective)),
            "fast"
        );

        // Test basic adverb forms
        assert_eq!(
            lemmatizer.lemmatize("quickly", Some(PosTag::Adverb)),
            "quick"
        );
    }

    #[test]
    fn test_rule_lemmatizer_exceptions() {
        let lemmatizer = RuleLemmatizer::new();

        // Test irregular verbs
        assert_eq!(lemmatizer.lemmatize("am", Some(PosTag::Verb)), "be");
        assert_eq!(lemmatizer.lemmatize("are", Some(PosTag::Verb)), "be");
        assert_eq!(lemmatizer.lemmatize("is", Some(PosTag::Verb)), "be");
        assert_eq!(lemmatizer.lemmatize("was", Some(PosTag::Verb)), "be");
        assert_eq!(lemmatizer.lemmatize("were", Some(PosTag::Verb)), "be");
        assert_eq!(lemmatizer.lemmatize("went", Some(PosTag::Verb)), "go");
        assert_eq!(lemmatizer.lemmatize("had", Some(PosTag::Verb)), "have");
        assert_eq!(lemmatizer.lemmatize("did", Some(PosTag::Verb)), "do");
        assert_eq!(lemmatizer.lemmatize("ate", Some(PosTag::Verb)), "eat");

        // Test irregular nouns
        assert_eq!(
            lemmatizer.lemmatize("children", Some(PosTag::Noun)),
            "child"
        );
        assert_eq!(lemmatizer.lemmatize("men", Some(PosTag::Noun)), "man");
        assert_eq!(lemmatizer.lemmatize("women", Some(PosTag::Noun)), "woman");
        assert_eq!(lemmatizer.lemmatize("mice", Some(PosTag::Noun)), "mouse");
        assert_eq!(lemmatizer.lemmatize("feet", Some(PosTag::Noun)), "foot");

        // Test irregular adjectives
        assert_eq!(
            lemmatizer.lemmatize("better", Some(PosTag::Adjective)),
            "good"
        );
        assert_eq!(
            lemmatizer.lemmatize("best", Some(PosTag::Adjective)),
            "good"
        );
        assert_eq!(
            lemmatizer.lemmatize("worse", Some(PosTag::Adjective)),
            "bad"
        );
        assert_eq!(
            lemmatizer.lemmatize("worst", Some(PosTag::Adjective)),
            "bad"
        );
    }

    #[test]
    fn test_rule_lemmatizer_case_preservation() {
        let lemmatizer = RuleLemmatizer::new();

        // Test original case preservation
        assert_eq!(lemmatizer.lemmatize("Running", Some(PosTag::Verb)), "Run");
        assert_eq!(lemmatizer.lemmatize("WALKED", Some(PosTag::Verb)), "WALK");
        assert_eq!(
            lemmatizer.lemmatize("Children", Some(PosTag::Noun)),
            "Child"
        );

        // Test with case restoration disabled
        let lemmatizer = RuleLemmatizerBuilder::new()
            .apply_case_restoration(false)
            .build();

        assert_eq!(lemmatizer.lemmatize("Running", Some(PosTag::Verb)), "run");
        assert_eq!(lemmatizer.lemmatize("WALKED", Some(PosTag::Verb)), "walk");
        assert_eq!(
            lemmatizer.lemmatize("Children", Some(PosTag::Noun)),
            "child"
        );
    }

    #[test]
    fn test_rule_lemmatizer_without_pos() {
        let lemmatizer = RuleLemmatizer::new();

        // Test without specifying POS (should default to verb)
        assert_eq!(lemmatizer.lemmatize("running", None), "run");
        assert_eq!(lemmatizer.lemmatize("played", None), "play");
        assert_eq!(lemmatizer.lemmatize("went", None), "go");

        // Change default POS and test again
        let lemmatizer = RuleLemmatizerBuilder::new()
            .default_pos(PosTag::Noun)
            .build();

        assert_eq!(lemmatizer.lemmatize("cats", None), "cat");
        assert_eq!(lemmatizer.lemmatize("cities", None), "city");
        assert_eq!(lemmatizer.lemmatize("children", None), "child");
    }

    #[test]
    fn test_rule_lemmatizer_edge_cases() {
        let lemmatizer = RuleLemmatizer::new();

        // Test empty string
        assert_eq!(lemmatizer.lemmatize("", None), "");

        // Test short words (should not be modified if too short for rules)
        assert_eq!(lemmatizer.lemmatize("a", None), "a");
        assert_eq!(lemmatizer.lemmatize("an", None), "an");
        assert_eq!(lemmatizer.lemmatize("the", None), "the");

        // Test words that might match multiple rules
        assert_eq!(lemmatizer.lemmatize("flies", Some(PosTag::Verb)), "fly");
        assert_eq!(lemmatizer.lemmatize("flies", Some(PosTag::Noun)), "fly");
    }

    #[test]
    fn test_rule_lemmatizer_custom_entries() {
        // Create lemmatizer with custom additions
        let lemmatizer = RuleLemmatizerBuilder::new()
            .add_exception("custom1", "custom1_lemma")
            .add_dict_entry("custom2", "custom2_lemma")
            .add_rule(
                LemmaRule {
                    pattern: "custom3$".to_string(),
                    replacement: "custom3_lemma".to_string(),
                    conditions: RuleCondition {
                        min_length: 7,
                        pos_tag: None,
                        requires_vowel: false,
                    },
                },
                PosTag::Verb,
            )
            .build();

        // Test custom entries
        assert_eq!(lemmatizer.lemmatize("custom1", None), "custom1_lemma");
        assert_eq!(lemmatizer.lemmatize("custom2", None), "custom2_lemma");
        assert_eq!(lemmatizer.lemmatize("xcustom3", None), "xcustom3_lemma");
    }

    #[test]
    fn test_stemmer_trait_implementation() {
        let lemmatizer = RuleLemmatizer::new();

        // Test Stemmer trait implementation
        assert_eq!(lemmatizer.stem("running").unwrap(), "run");
        assert_eq!(lemmatizer.stem("children").unwrap(), "child");

        // Test batch processing
        let words = vec!["running", "played", "children", "better"];
        let expected = vec!["run", "play", "child", "good"];
        assert_eq!(lemmatizer.stem_batch(&words).unwrap(), expected);
    }
}
