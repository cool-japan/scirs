//! Dependency-parse based relation extraction.
//!
//! Extracts Subject–Verb–Object and custom relational triples from a
//! dependency parse tree represented as a flat list of arcs.

use std::collections::HashMap;

use crate::error::Result;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A single arc in a dependency tree.
#[derive(Debug, Clone)]
pub struct DependencyRelation {
    /// Head token text.
    pub head: String,
    /// Dependency relation label (e.g. "nsubj", "obj", "dobj").
    pub relation: String,
    /// Dependent token text.
    pub dependent: String,
}

/// A pattern that describes what Subject-Predicate-Object triples to extract.
#[derive(Debug, Clone)]
pub struct RelationPattern {
    /// Optional POS filter on the subject (e.g. "NN").
    pub subject_pos: Option<String>,
    /// Verbs / predicates that trigger extraction (case-insensitive).
    pub predicate: Vec<String>,
    /// Optional POS filter on the object.
    pub object_pos: Option<String>,
    /// Label assigned to extracted triples.
    pub label: String,
}

// ---------------------------------------------------------------------------
// RelationExtractorDep
// ---------------------------------------------------------------------------

/// Dependency-based relation extractor.
pub struct DependencyRelationExtractor {
    patterns: Vec<RelationPattern>,
}

impl Default for DependencyRelationExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl DependencyRelationExtractor {
    /// Create an empty extractor.
    pub fn new() -> DependencyRelationExtractor {
        DependencyRelationExtractor {
            patterns: Vec::new(),
        }
    }

    /// Add a relation pattern.
    pub fn add_pattern(&mut self, pattern: RelationPattern) {
        self.patterns.push(pattern);
    }

    /// Build a simple default extractor that finds SVO triples.
    pub fn with_svo_defaults() -> DependencyRelationExtractor {
        let mut ext = DependencyRelationExtractor::new();
        ext.add_pattern(RelationPattern {
            subject_pos: None,
            predicate: vec![], // any verb
            object_pos: None,
            label: "SVO".to_string(),
        });
        ext
    }

    /// Extract (subject, relation_type, object) triples from a dependency tree.
    ///
    /// # Arguments
    /// * `_text`          – original sentence text (unused; available for future context)
    /// * `dependency_tree` – flat list of dependency arcs
    ///
    /// # Returns
    /// A `Vec<(subject, relation_type, object)>`.
    pub fn extract(
        &self,
        _text: &str,
        dependency_tree: &[DependencyRelation],
    ) -> Result<Vec<(String, String, String)>> {
        // Build head → {relation → [dependents]} index
        let mut head_map: HashMap<&str, Vec<(&str, &str)>> = HashMap::new();
        for arc in dependency_tree {
            head_map
                .entry(arc.head.as_str())
                .or_default()
                .push((arc.relation.as_str(), arc.dependent.as_str()));
        }

        let mut triples = Vec::new();

        // Collect unique head words to avoid duplicate processing
        let mut seen_heads = std::collections::HashSet::new();
        let head_words: Vec<String> = dependency_tree
            .iter()
            .filter_map(|arc| {
                if seen_heads.insert(arc.head.clone()) {
                    Some(arc.head.clone())
                } else {
                    None
                }
            })
            .collect();

        for head in &head_words {
            let Some(deps) = head_map.get(head.as_str()) else {
                continue;
            };

            // Find nsubj / obj pairs under the same head
            let subjects: Vec<&str> = deps
                .iter()
                .filter(|(rel, _)| *rel == "nsubj" || *rel == "nsubjpass")
                .map(|(_, dep)| *dep)
                .collect();

            let objects: Vec<&str> = deps
                .iter()
                .filter(|(rel, _)| {
                    *rel == "obj"
                        || *rel == "dobj"
                        || *rel == "iobj"
                        || *rel == "obl"
                        || *rel == "xobj"
                })
                .map(|(_, dep)| *dep)
                .collect();

            if subjects.is_empty() || objects.is_empty() {
                continue;
            }

            // Check against patterns
            for subj in &subjects {
                for obj in &objects {
                    for pattern in &self.patterns {
                        // Check predicate filter
                        if !pattern.predicate.is_empty() {
                            let head_lower = head.to_lowercase();
                            if !pattern
                                .predicate
                                .iter()
                                .any(|p| p.to_lowercase() == head_lower)
                            {
                                continue;
                            }
                        }
                        triples.push((subj.to_string(), pattern.label.clone(), obj.to_string()));
                    }
                }
            }
        }

        Ok(triples)
    }
}

// ---------------------------------------------------------------------------
// Simple coreference resolver
// ---------------------------------------------------------------------------

/// A minimal pronoun → antecedent resolver using recency heuristics.
///
/// It maintains a short-term memory of recently seen noun phrases and
/// replaces pronouns with their most likely antecedent based on gender/number
/// agreement and distance.
pub struct CorefResolver {
    /// Recently seen noun phrases: `(text, is_plural, gender)`.
    history: Vec<(String, bool, PronounGender)>,
    /// How many candidates to keep in memory.
    window: usize,
}

/// Coarse gender category for pronoun resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PronounGender {
    /// Masculine pronoun (he, him, his).
    Masculine,
    /// Feminine pronoun (she, her, hers).
    Feminine,
    /// Gender-neutral pronoun (it, they singular).
    Neutral,
    /// Plural pronoun (they, them, their).
    Plural,
    /// Gender could not be determined.
    Unknown,
}

impl CorefResolver {
    /// Create a new resolver with a recency window of `window` phrases.
    pub fn new(window: usize) -> CorefResolver {
        CorefResolver {
            history: Vec::new(),
            window,
        }
    }

    /// Register a noun phrase as a potential antecedent.
    pub fn register(
        &mut self,
        noun_phrase: impl Into<String>,
        is_plural: bool,
        gender: PronounGender,
    ) {
        if self.history.len() >= self.window {
            self.history.remove(0);
        }
        self.history.push((noun_phrase.into(), is_plural, gender));
    }

    /// Resolve a pronoun to its most recent compatible antecedent.
    ///
    /// Returns `None` if no compatible antecedent is found in the window.
    pub fn resolve(&self, pronoun: &str) -> Option<&str> {
        let (target_plural, target_gender) = pronoun_attributes(pronoun)?;
        // Search from most recent backwards
        for (np, is_plural, gender) in self.history.iter().rev() {
            if *is_plural != target_plural {
                continue;
            }
            if target_gender != PronounGender::Unknown
                && *gender != PronounGender::Unknown
                && *gender != target_gender
            {
                continue;
            }
            return Some(np.as_str());
        }
        None
    }
}

/// Return (is_plural, gender) for common English pronouns.
fn pronoun_attributes(pronoun: &str) -> Option<(bool, PronounGender)> {
    match pronoun.to_lowercase().as_str() {
        "he" | "him" | "his" | "himself" => Some((false, PronounGender::Masculine)),
        "she" | "her" | "hers" | "herself" => Some((false, PronounGender::Feminine)),
        "it" | "its" | "itself" => Some((false, PronounGender::Neutral)),
        "they" | "them" | "their" | "theirs" | "themselves" => Some((true, PronounGender::Plural)),
        "we" | "us" | "our" | "ours" | "ourselves" => Some((true, PronounGender::Plural)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_svo_tree() -> Vec<DependencyRelation> {
        vec![
            DependencyRelation {
                head: "loves".to_string(),
                relation: "nsubj".to_string(),
                dependent: "John".to_string(),
            },
            DependencyRelation {
                head: "loves".to_string(),
                relation: "obj".to_string(),
                dependent: "Mary".to_string(),
            },
        ]
    }

    #[test]
    fn test_svo_extraction() {
        let extractor = DependencyRelationExtractor::with_svo_defaults();
        let triples = extractor
            .extract("John loves Mary", &make_svo_tree())
            .expect("extract failed");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].0, "John");
        assert_eq!(triples[0].2, "Mary");
    }

    #[test]
    fn test_no_triples_without_subject() {
        let extractor = DependencyRelationExtractor::with_svo_defaults();
        let tree = vec![DependencyRelation {
            head: "runs".to_string(),
            relation: "obj".to_string(),
            dependent: "race".to_string(),
        }];
        let triples = extractor
            .extract("runs race", &tree)
            .expect("extract failed");
        assert!(triples.is_empty());
    }

    #[test]
    fn test_predicate_filter() {
        let mut extractor = DependencyRelationExtractor::new();
        extractor.add_pattern(RelationPattern {
            subject_pos: None,
            predicate: vec!["loves".to_string()],
            object_pos: None,
            label: "LOVE".to_string(),
        });

        // Matching predicate
        let triples = extractor
            .extract("John loves Mary", &make_svo_tree())
            .expect("extract failed");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].1, "LOVE");

        // Non-matching predicate — extractor has only "loves" pattern
        let tree2 = vec![
            DependencyRelation {
                head: "hates".to_string(),
                relation: "nsubj".to_string(),
                dependent: "John".to_string(),
            },
            DependencyRelation {
                head: "hates".to_string(),
                relation: "obj".to_string(),
                dependent: "Mary".to_string(),
            },
        ];
        let triples2 = extractor
            .extract("John hates Mary", &tree2)
            .expect("extract failed");
        assert!(triples2.is_empty());
    }

    #[test]
    fn test_coref_resolver_basic() {
        let mut resolver = CorefResolver::new(5);
        resolver.register("John Smith", false, PronounGender::Masculine);
        let antecedent = resolver.resolve("he");
        assert_eq!(antecedent, Some("John Smith"));
    }

    #[test]
    fn test_coref_resolver_gender_mismatch() {
        let mut resolver = CorefResolver::new(5);
        resolver.register("Alice", false, PronounGender::Feminine);
        // "he" should NOT resolve to Alice
        let antecedent = resolver.resolve("he");
        assert!(antecedent.is_none());
    }

    #[test]
    fn test_coref_resolver_recency() {
        let mut resolver = CorefResolver::new(5);
        resolver.register("Bob", false, PronounGender::Masculine);
        resolver.register("Alice", false, PronounGender::Feminine);
        // "he" should resolve to Bob (most recent masculine)
        let antecedent = resolver.resolve("he");
        assert_eq!(antecedent, Some("Bob"));
    }

    #[test]
    fn test_coref_resolver_window_eviction() {
        let mut resolver = CorefResolver::new(2);
        resolver.register("Old Guy", false, PronounGender::Masculine);
        resolver.register("Middle Person", false, PronounGender::Unknown);
        resolver.register("New Person", false, PronounGender::Unknown);
        // "Old Guy" should have been evicted (window=2)
        let names: Vec<&str> = resolver
            .history
            .iter()
            .map(|(n, _, _)| n.as_str())
            .collect();
        assert!(!names.contains(&"Old Guy"));
    }
}
