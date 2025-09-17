//! Translation pipeline implementation
//!
//! This module provides functionality for text translation between languages.

use super::TranslationResult;
use crate::error::Result;
use std::collections::HashMap;

/// Translation pipeline
#[derive(Debug)]
pub struct TranslationPipeline {
    /// Source language
    src_lang: String,
    /// Target language
    tgt_lang: String,
}

impl Default for TranslationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl TranslationPipeline {
    /// Create new translation pipeline
    pub fn new() -> Self {
        Self {
            src_lang: "en".to_string(),
            tgt_lang: "fr".to_string(),
        }
    }

    /// Translate text
    pub fn translate(&self, text: &str) -> Result<TranslationResult> {
        // Simple dictionary-based translation for demonstration
        let translation_text = self.simple_translate(text);

        Ok(TranslationResult { translation_text })
    }

    fn simple_translate(&self, text: &str) -> String {
        // Very basic word-by-word translation dictionary
        let mut dictionary = HashMap::new();

        // English to French basic dictionary
        if self.src_lang == "en" && self.tgt_lang == "fr" {
            dictionary.insert("hello", "bonjour");
            dictionary.insert("world", "monde");
            dictionary.insert("the", "le");
            dictionary.insert("cat", "chat");
            dictionary.insert("dog", "chien");
            dictionary.insert("house", "maison");
            dictionary.insert("car", "voiture");
            dictionary.insert("good", "bon");
            dictionary.insert("bad", "mauvais");
            dictionary.insert("big", "grand");
            dictionary.insert("small", "petit");
            dictionary.insert("red", "rouge");
            dictionary.insert("blue", "bleu");
            dictionary.insert("green", "vert");
            dictionary.insert("and", "et");
            dictionary.insert("or", "ou");
            dictionary.insert("but", "mais");
            dictionary.insert("yes", "oui");
            dictionary.insert("no", "non");
        } else if self.src_lang == "fr" && self.tgt_lang == "en" {
            dictionary.insert("bonjour", "hello");
            dictionary.insert("monde", "world");
            dictionary.insert("le", "the");
            dictionary.insert("chat", "cat");
            dictionary.insert("chien", "dog");
            dictionary.insert("maison", "house");
            dictionary.insert("voiture", "car");
            dictionary.insert("bon", "good");
            dictionary.insert("mauvais", "bad");
            dictionary.insert("grand", "big");
            dictionary.insert("petit", "small");
            dictionary.insert("rouge", "red");
            dictionary.insert("bleu", "blue");
            dictionary.insert("vert", "green");
            dictionary.insert("et", "and");
            dictionary.insert("ou", "or");
            dictionary.insert("mais", "but");
            dictionary.insert("oui", "yes");
            dictionary.insert("non", "no");
        }

        // Translate word by word
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower.split_whitespace().collect();
        let translated_words: Vec<String> = words
            .iter()
            .map(|word| {
                let clean_word = word.trim_matches(|c: char| !c.is_alphabetic());
                dictionary
                    .get(clean_word)
                    .map(|&translation| translation.to_string())
                    .unwrap_or_else(|| format!("({})", word))
            })
            .collect();

        translated_words.join(" ")
    }

    /// Set translation languages
    pub fn set_languages(&mut self, src_lang: String, tgt_lang: String) {
        self.src_lang = src_lang;
        self.tgt_lang = tgt_lang;
    }
}
