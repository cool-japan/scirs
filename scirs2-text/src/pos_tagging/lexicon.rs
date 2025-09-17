//! Built-in lexicon for POS tagging
//!
//! This module contains the default lexicon with common words
//! and their most likely POS tags.

use crate::stemming::PosTag;
use std::collections::HashMap;

/// Initialize the built-in lexicon with common words
pub fn initialize_lexicon() -> HashMap<String, PosTag> {
    let mut lexicon = HashMap::new();

    // Common determiners and articles
    let determiners = [
        "the", "a", "an", "this", "that", "these", "those", "some", "any", "each", "every", "all",
        "both", "few", "many", "much", "several",
    ];
    for word in &determiners {
        lexicon.insert(word.to_string(), PosTag::Other);
    }

    // Common prepositions
    let prepositions = [
        "in", "on", "at", "by", "for", "with", "without", "to", "from", "of", "about", "over",
        "under", "through", "during", "before", "after", "above", "below", "between", "among",
    ];
    for word in &prepositions {
        lexicon.insert(word.to_string(), PosTag::Other);
    }

    // Common conjunctions
    let conjunctions = [
        "and", "or", "but", "nor", "for", "so", "yet", "although", "because", "since", "when",
        "where", "while", "if", "unless", "until",
    ];
    for word in &conjunctions {
        lexicon.insert(word.to_string(), PosTag::Other);
    }

    // Common pronouns
    let pronouns = [
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
        "your", "his", "her", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
    ];
    for word in &pronouns {
        lexicon.insert(word.to_string(), PosTag::Other);
    }

    // Common auxiliary verbs
    let auxiliaries = [
        "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "must",
    ];
    for word in &auxiliaries {
        lexicon.insert(word.to_string(), PosTag::Verb);
    }

    // Add common verbs
    add_common_verbs(&mut lexicon);

    // Add common nouns
    add_common_nouns(&mut lexicon);

    // Add common adjectives
    add_common_adjectives(&mut lexicon);

    // Add common adverbs
    add_common_adverbs(&mut lexicon);

    lexicon
}

fn add_common_verbs(lexicon: &mut HashMap<String, PosTag>) {
    let verb_forms = [
        // Go forms
        ("go", PosTag::Verb),
        ("went", PosTag::Verb),
        ("goes", PosTag::Verb),
        ("going", PosTag::Verb),
        ("gone", PosTag::Verb),
        // Make forms
        ("make", PosTag::Verb),
        ("made", PosTag::Verb),
        ("makes", PosTag::Verb),
        ("making", PosTag::Verb),
        // Take forms
        ("take", PosTag::Verb),
        ("took", PosTag::Verb),
        ("takes", PosTag::Verb),
        ("taking", PosTag::Verb),
        ("taken", PosTag::Verb),
        // Come forms
        ("come", PosTag::Verb),
        ("came", PosTag::Verb),
        ("comes", PosTag::Verb),
        ("coming", PosTag::Verb),
        // See forms
        ("see", PosTag::Verb),
        ("saw", PosTag::Verb),
        ("sees", PosTag::Verb),
        ("seeing", PosTag::Verb),
        ("seen", PosTag::Verb),
        // Get forms
        ("get", PosTag::Verb),
        ("got", PosTag::Verb),
        ("gets", PosTag::Verb),
        ("getting", PosTag::Verb),
        ("gotten", PosTag::Verb),
        // Know forms
        ("know", PosTag::Verb),
        ("knew", PosTag::Verb),
        ("knows", PosTag::Verb),
        ("knowing", PosTag::Verb),
        ("known", PosTag::Verb),
        // Additional common verbs
        ("think", PosTag::Verb),
        ("thought", PosTag::Verb),
        ("thinks", PosTag::Verb),
        ("thinking", PosTag::Verb),
        ("say", PosTag::Verb),
        ("said", PosTag::Verb),
        ("says", PosTag::Verb),
        ("saying", PosTag::Verb),
        ("tell", PosTag::Verb),
        ("told", PosTag::Verb),
        ("tells", PosTag::Verb),
        ("telling", PosTag::Verb),
        ("give", PosTag::Verb),
        ("gave", PosTag::Verb),
        ("gives", PosTag::Verb),
        ("giving", PosTag::Verb),
        ("given", PosTag::Verb),
        ("find", PosTag::Verb),
        ("found", PosTag::Verb),
        ("finds", PosTag::Verb),
        ("finding", PosTag::Verb),
        ("work", PosTag::Verb),
        ("worked", PosTag::Verb),
        ("works", PosTag::Verb),
        ("working", PosTag::Verb),
    ];

    for (word, tag) in &verb_forms {
        lexicon.insert(word.to_string(), tag.clone());
    }
}

fn add_common_nouns(lexicon: &mut HashMap<String, PosTag>) {
    let nouns = [
        "time",
        "person",
        "year",
        "way",
        "day",
        "thing",
        "man",
        "world",
        "life",
        "hand",
        "part",
        "child",
        "eye",
        "woman",
        "place",
        "work",
        "week",
        "case",
        "point",
        "government",
        "company",
        "number",
        "group",
        "problem",
        "fact",
        "be",
        "have",
        "do",
        "say",
        "get",
        "make",
        "go",
        "know",
        "take",
        "see",
        "come",
        "think",
        "look",
        "want",
        "give",
        "use",
        "find",
        "tell",
        "ask",
        "work",
        "seem",
        "feel",
        "try",
        "leave",
        "call",
    ];

    for noun in &nouns {
        lexicon.insert(noun.to_string(), PosTag::Noun);
    }
}

fn add_common_adjectives(lexicon: &mut HashMap<String, PosTag>) {
    let adjectives = [
        "good",
        "new",
        "first",
        "last",
        "long",
        "great",
        "little",
        "own",
        "other",
        "old",
        "right",
        "big",
        "high",
        "different",
        "small",
        "large",
        "next",
        "early",
        "young",
        "important",
        "few",
        "public",
        "bad",
        "same",
        "able",
        "available",
        "likely",
        "free",
        "political",
        "special",
        "certain",
        "personal",
        "open",
        "red",
        "difficult",
        "far",
        "local",
        "sure",
        "cold",
        "clear",
        "recent",
        "international",
        "full",
    ];

    for adj in &adjectives {
        lexicon.insert(adj.to_string(), PosTag::Adjective);
    }
}

fn add_common_adverbs(lexicon: &mut HashMap<String, PosTag>) {
    let adverbs = [
        "not",
        "so",
        "out",
        "up",
        "only",
        "just",
        "now",
        "how",
        "then",
        "more",
        "also",
        "here",
        "well",
        "where",
        "why",
        "back",
        "down",
        "off",
        "over",
        "again",
        "still",
        "in",
        "on",
        "when",
        "much",
        "very",
        "too",
        "really",
        "quite",
        "rather",
        "pretty",
        "almost",
        "always",
        "never",
        "often",
        "sometimes",
        "usually",
        "probably",
        "maybe",
        "perhaps",
        "definitely",
        "certainly",
        "clearly",
        "obviously",
        "exactly",
        "actually",
    ];

    for adv in &adverbs {
        lexicon.insert(adv.to_string(), PosTag::Adverb);
    }
}
