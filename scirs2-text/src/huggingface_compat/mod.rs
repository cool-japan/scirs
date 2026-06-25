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
        let results = pipeline
            .predict("This is a great movie!")
            .expect("Operation failed");
        assert_eq!(results.len(), 2);
        assert!(results[0].score >= 0.0 && results[0].score <= 1.0);
    }

    #[test]
    fn test_zero_shot_classification() {
        let pipeline = ZeroShotClassificationPipeline::new();
        let labels = ["positive", "negative", "neutral"];
        let results = pipeline
            .classify("This is a wonderful day", &labels)
            .expect("Operation failed");
        assert_eq!(results.len(), 3);
        assert!(results[0].score >= results[1].score);
        assert!(results[1].score >= results[2].score);
    }

    #[test]
    fn test_question_answering() {
        let pipeline = QuestionAnsweringPipeline::new();
        let context = "The quick brown fox jumps over the lazy dog.";
        let question = "What jumps over the dog?";

        let result = pipeline
            .answer(question, context)
            .expect("Operation failed");
        assert!(!result.answer.is_empty());
        assert!(result.score > 0.0);
        assert!(result.start < result.end);
    }

    #[test]
    fn test_hf_model_adapter_pipeline_creation() {
        let config = HfConfig::default();
        let adapter = HfModelAdapter::new(config);

        let text_class_pipeline = adapter
            .create_pipeline("text-classification")
            .expect("Operation failed");
        assert!(matches!(
            text_class_pipeline,
            HfPipeline::TextClassification(_)
        ));

        let zero_shot_pipeline = adapter
            .create_pipeline("zero-shot-classification")
            .expect("Operation failed");
        assert!(matches!(
            zero_shot_pipeline,
            HfPipeline::ZeroShotClassification(_)
        ));

        let qa_pipeline = adapter
            .create_pipeline("question-answering")
            .expect("Operation failed");
        assert!(matches!(qa_pipeline, HfPipeline::QuestionAnswering(_)));
    }

    #[test]
    fn test_hub_list_models_requires_network() {
        let hub = HfHub::new();
        // Without a networking backend, listing models must return an honest
        // error rather than a fabricated catalogue.
        assert!(hub.list_models(None).is_err());
        assert!(hub.list_models(Some("bert")).is_err());
    }

    #[test]
    fn test_hub_model_info_cache_roundtrip() {
        let mut hub = HfHub::new();

        // Uncached lookups require the network and must error honestly.
        assert!(hub.model_info("bert-base-uncased").is_err());

        // Explicitly cached metadata is returned without any network access.
        let info = HfModelInfo {
            model_id: "bert-base-uncased".to_string(),
            tags: vec!["pytorch".to_string()],
            pipeline_tag: Some("fill-mask".to_string()),
            downloads: 0,
            likes: 0,
            library_name: Some("transformers".to_string()),
        };
        hub.cache_model_info(info.clone());

        let fetched = hub
            .model_info("bert-base-uncased")
            .expect("cached model info should be returned");
        assert_eq!(fetched.model_id, "bert-base-uncased");
        assert_eq!(fetched.pipeline_tag.as_deref(), Some("fill-mask"));
    }

    #[test]
    fn test_model_manager_load_without_local_model_errors() {
        let mut manager = HfModelManager::new();

        // No local model files exist and downloading requires the network, so
        // loading must fail honestly instead of fabricating a config.
        let config = manager.load_model("definitely-not-a-real-local-model");
        assert!(config.is_err());

        // The failed load must not have populated the cache.
        let cached_models = manager.list_cached_models();
        assert!(cached_models.is_empty());

        // Unloading a model that was never loaded reports false.
        assert!(!manager.unload_model("definitely-not-a-real-local-model"));
    }

    #[test]
    fn test_feature_extraction() {
        let pipeline = FeatureExtractionPipeline::new();
        let features = pipeline
            .extract_features("Hello world")
            .expect("Operation failed");
        assert_eq!(features.shape()[1], 768); // Feature dimension
        assert!(features.shape()[0] > 0); // Sequence length
    }

    #[test]
    fn test_fill_mask() {
        let pipeline = FillMaskPipeline::new();
        let results = pipeline
            .fill_mask("The quick [MASK] fox")
            .expect("Operation failed");
        assert!(!results.is_empty());
        assert!(!results[0].sequence.contains("[MASK]"));
    }

    #[test]
    fn test_summarization() {
        let pipeline = SummarizationPipeline::new();
        let text = "This is a long text that needs to be summarized. It contains multiple sentences with various information.";
        let result = pipeline.summarize(text).expect("Operation failed");
        assert!(!result.summary_text.is_empty());
        assert!(result.summary_text.len() <= text.len());
    }

    #[test]
    fn test_translation() {
        let pipeline = TranslationPipeline::new();
        let result = pipeline.translate("hello world").expect("Operation failed");
        assert!(!result.translation_text.is_empty());
    }

    #[test]
    fn test_token_classification() {
        let pipeline = TokenClassificationPipeline::new();
        let results = pipeline
            .classify_tokens("John works at Microsoft in Seattle")
            .expect("Operation failed");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_text_generation() {
        let pipeline = TextGenerationPipeline::new();
        let results = pipeline
            .generate("The weather today is")
            .expect("Operation failed");
        assert!(!results.is_empty());
        assert!(results[0]
            .generated_text
            .starts_with("The weather today is"));
    }
}
