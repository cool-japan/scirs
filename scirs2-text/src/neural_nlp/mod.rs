//! Neural NLP integration: transformer encoder, attention visualization,
//! BERT-style fine-tuning, and neural NER.
//!
//! All implementations are pure-ndarray (f32) and do not depend on any
//! external ML runtime.

pub mod attention_viz;
pub mod bert_classifier;
pub mod neural_ner;
pub mod transformer_encoder;

pub use attention_viz::{AttentionHeatmap, AttentionVisualization};
pub use bert_classifier::{BertClassifier, BertClassifierConfig};
pub use neural_ner::{NerTag, NeuralNer, NeuralNerConfig};
pub use transformer_encoder::{TransformerEncoderConfig, TransformerTextEncoder};
