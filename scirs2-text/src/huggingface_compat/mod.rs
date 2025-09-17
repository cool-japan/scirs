//! Hugging Face compatibility layer for interoperability
//!
//! This module provides compatibility interfaces and adapters to work with
//! Hugging Face model formats, tokenizers, and APIs, enabling seamless
//! integration with the broader ML ecosystem.

pub mod adapter;
pub mod config;
pub mod conversion;
pub mod hub;
pub mod manager;
pub mod pipelines;
pub mod tokenizer;

// Re-export main components for convenience
pub use adapter::HfModelAdapter;
pub use config::{HfConfig, HfTokenizerConfig};
pub use conversion::FormatConverter;
pub use hub::{HfHub, HfModelInfo};
pub use manager::HfModelManager;
pub use pipelines::{
    ClassificationResult,
    // Pipeline implementations
    FeatureExtractionPipeline,
    FillMaskPipeline,
    FillMaskResult,
    HfPipeline,
    QuestionAnsweringPipeline,
    QuestionAnsweringResult,
    SummarizationPipeline,
    SummarizationResult,
    TextClassificationPipeline,
    TextGenerationPipeline,
    TextGenerationResult,
    TokenClassificationPipeline,
    TokenClassificationResult,
    TranslationPipeline,
    TranslationResult,
    ZeroShotClassificationPipeline,
};
pub use tokenizer::{HfEncodedInput, HfTokenizer};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;

    #[test]
    fn test_hf_config_default() {
        let config = HfConfig::default();
        assert_eq!(config.model_type, "bert");
        assert_eq!(config.hidden_size, Some(768));
        assert_eq!(config.num_attention_heads, Some(12));
    }

    #[test]
    fn test_hf_tokenizer_config_default() {
        let config = HfTokenizerConfig::default();
        assert_eq!(config.tokenizer_type, "WordPiece");
        assert_eq!(config.max_len, 512);
        assert_eq!(config.pad_token, "[PAD]");
        assert_eq!(config.unk_token, "[UNK]");
    }

    #[test]
    fn test_text_classification_pipeline() {
        let pipeline = TextClassificationPipeline::new();
        let results = pipeline.predict("This is a great movie!").unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].score >= 0.0 && results[0].score <= 1.0);
    }

    #[test]
    fn test_zero_shot_classification() {
        let pipeline = ZeroShotClassificationPipeline::new();
        let labels = ["positive", "negative", "neutral"];
        let results = pipeline
            .classify("This is a wonderful day", &labels)
            .unwrap();
        assert_eq!(results.len(), 3);
        assert!(results[0].score >= results[1].score);
        assert!(results[1].score >= results[2].score);
    }

    #[test]
    fn test_question_answering() {
        let pipeline = QuestionAnsweringPipeline::new();
        let context = "The quick brown fox jumps over the lazy dog.";
        let question = "What jumps over the dog?";

        let result = pipeline.answer(question, context).unwrap();
        assert!(!result.answer.is_empty());
        assert!(result.score > 0.0);
        assert!(result.start < result.end);
    }

    #[test]
    fn test_hf_model_adapter_pipeline_creation() {
        let config = HfConfig::default();
        let adapter = HfModelAdapter::new(config);

        let text_class_pipeline = adapter.create_pipeline("text-classification").unwrap();
        assert!(matches!(
            text_class_pipeline,
            HfPipeline::TextClassification(_)
        ));

        let zero_shot_pipeline = adapter.create_pipeline("zero-shot-classification").unwrap();
        assert!(matches!(
            zero_shot_pipeline,
            HfPipeline::ZeroShotClassification(_)
        ));

        let qa_pipeline = adapter.create_pipeline("question-answering").unwrap();
        assert!(matches!(qa_pipeline, HfPipeline::QuestionAnswering(_)));
    }

    #[test]
    fn test_hub_list_models() {
        let hub = HfHub::new();
        let models = hub.list_models(None).unwrap();
        assert!(!models.is_empty());

        let filtered = hub.list_models(Some("bert")).unwrap();
        assert!(filtered.iter().any(|m| m.contains("bert")));
    }

    #[test]
    fn test_model_manager() {
        let mut manager = HfModelManager::new();

        // Test loading model (will use mock/default config)
        let config = manager.load_model("test-model");
        assert!(config.is_ok());

        // Test caching
        let cached_models = manager.list_cached_models();
        assert!(cached_models.contains(&"test-model".to_string()));

        // Test unloading
        let unloaded = manager.unload_model("test-model");
        assert!(unloaded);

        let cached_after_unload = manager.list_cached_models();
        assert!(!cached_after_unload.contains(&"test-model".to_string()));
    }

    #[test]
    fn test_feature_extraction() {
        let pipeline = FeatureExtractionPipeline::new();
        let features = pipeline.extract_features("Hello world").unwrap();
        assert_eq!(features.shape()[1], 768); // Feature dimension
        assert!(features.shape()[0] > 0); // Sequence length
    }

    #[test]
    fn test_fill_mask() {
        let pipeline = FillMaskPipeline::new();
        let results = pipeline.fill_mask("The quick [MASK] fox").unwrap();
        assert!(!results.is_empty());
        assert!(!results[0].sequence.contains("[MASK]"));
    }

    #[test]
    fn test_summarization() {
        let pipeline = SummarizationPipeline::new();
        let text = "This is a long text that needs to be summarized. It contains multiple sentences with various information.";
        let result = pipeline.summarize(text).unwrap();
        assert!(!result.summary_text.is_empty());
        assert!(result.summary_text.len() <= text.len());
    }

    #[test]
    fn test_translation() {
        let pipeline = TranslationPipeline::new();
        let result = pipeline.translate("hello world").unwrap();
        assert!(!result.translation_text.is_empty());
    }

    #[test]
    fn test_token_classification() {
        let pipeline = TokenClassificationPipeline::new();
        let results = pipeline
            .classify_tokens("John works at Microsoft in Seattle")
            .unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_text_generation() {
        let pipeline = TextGenerationPipeline::new();
        let results = pipeline.generate("The weather today is").unwrap();
        assert!(!results.is_empty());
        assert!(results[0]
            .generated_text
            .starts_with("The weather today is"));
    }
}
