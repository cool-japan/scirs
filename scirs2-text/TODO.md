# scirs2-text TODO

## Status: v0.4.3 Released (May 3, 2026)

All v0.4.3 features are complete and production-ready. Highlights since v0.4.2:
- `TransformerTextEncoder` (multi-head self-attention, sinusoidal positional encoding, GELU FFN, pure-ndarray)
- `BertClassifier` (frozen encoder + SGD classification head) and `NeuralNer` (BIO-9 tagging with token-level F1)
- Hierarchical Dirichlet Process via Chinese Restaurant Franchise Gibbs sampler (`topic/hdp.rs`)
- `UniversalSentenceEncoder` with six pooling strategies (CLS / Mean / Max / MeanSqrt / AttentionPooling / WeightedMean)
- Language-agnostic Unicode tokenizer (`tokenization/language_agnostic.rs`)
- SimCSE contrastive trainer — note: the projection head deviates from the original `scirs2-autograd` plan and uses pure-ndarray analytic backpropagation due to broken upstream gradient paths (`extract_diag`, `ScalarMulOp`, Adam's `Variable::compute`). The pure-ndarray MLP delivers identical training semantics (Glorot init, Bernoulli dropout, SGD with InfoNCE loss, L2 penalty) and will switch back once the upstream autograd bugs are fixed.

## Status: v0.3.4 Released (March 18, 2026)

## v0.3.3 Completed

### Core Tokenization
- [x] `WordTokenizer` - Unicode-aware word tokenization, configurable lowercase
- [x] `SentenceTokenizer` - Rule-based sentence boundary detection
- [x] `CharTokenizer` - Character and Unicode grapheme cluster tokenization
- [x] `NgramTokenizer` - N-grams with fixed n and range support
- [x] `RegexTokenizer` - Pattern-based and gap-based tokenization
- [x] `WhitespaceTokenizer` - Simple whitespace splitting
- [x] `BpeTokenizer` - Byte Pair Encoding with vocabulary learning and save/load
- [x] WordPiece tokenizer (BERT-style subword)
- [x] `Tokenizer` trait for interchangeable backends

### Text Preprocessing
- [x] `BasicNormalizer` - Unicode normalization, case folding, accent removal
- [x] `BasicTextCleaner` - HTML/XML stripping, URL/email normalization, stopwords
- [x] Contraction expansion
- [x] Number normalization (dates, currencies, percentages, ordinals)
- [x] `TextPreprocessor` - Composable normalizer + cleaner pipeline

### Stemming and Lemmatization
- [x] `PorterStemmer` - Classic Porter algorithm
- [x] `SnowballStemmer` - Snowball algorithm (English)
- [x] `LancasterStemmer` - Aggressive Lancaster stemming
- [x] `SimpleLemmatizer` - Dictionary-based lemmatization with morphological analysis
- [x] `Stemmer` trait for interchangeable backends

### Text Vectorization
- [x] `CountVectorizer` - Bag-of-words with N-gram support and vocabulary management
- [x] `TfidfVectorizer` - TF-IDF with smoothing, sublinear TF, L1/L2 normalization
- [x] `BinaryVectorizer` - Binary occurrence representation
- [x] `EnhancedCountVectorizer` - Max features, min/max document frequency
- [x] `EnhancedTfidfVectorizer` - All enhanced options + advanced IDF weighting
- [x] Sparse matrix output for memory efficiency
- [x] Vocabulary persistence (save/load)

### Word Embeddings
- [x] `Word2Vec` - Skip-gram and CBOW with negative sampling
- [x] Configurable: vector size, window, min_count, iterations, negative samples
- [x] `most_similar()` cosine similarity lookup
- [x] Binary and text format save/load
- [x] GloVe vector loading
- [x] `FastText` (pure Rust subword embeddings with character n-grams)

### Sequence Labelling
- [x] `CrfTagger` - CRF with Viterbi decoding and custom feature functions
- [x] `HmmTagger` - HMM for POS tagging (forward-backward, Viterbi)
- [x] Feature engineering utilities for NER, POS, chunking

### Named Entity Recognition (NER)
- [x] Rule-based NER with regex patterns
- [x] Dictionary/gazetteer-based NER
- [x] CRF-based NER with feature engineering
- [x] Standard types: PER, ORG, LOC, DATE, TIME, MONEY, PERCENT
- [x] Entity span detection with start/end offsets

### Advanced NLP (New in v0.3.1)
- [x] `coreference` - Mention detection and coreference clustering
- [x] `dependency` - Arc-factored dependency graph construction
- [x] `discourse` - Discourse analysis and RST primitives
- [x] `event_extraction` - Event trigger and argument extraction
- [x] `question_answering` - Extractive span detection
- [x] `knowledge_graph` - Entity-relation-entity triple extraction
- [x] `semantic_parsing` - Logical form generation
- [x] `temporal` - Date/time expression normalization (TIMEX3-style)
- [x] `grammar` - Rule-based grammar error detection
- [x] `annotation` - Annotation layer management

### Topic Modeling
- [x] `LatentDirichletAllocation` (LDA) - variational inference
- [x] Coherence metrics: CV, UMass, UCI
- [x] NMF-based topic modeling
- [x] `TopicModel` trait

### Summarization
- [x] Extractive: TextRank, centroid-based, keyword-based sentence scoring
- [x] `abstractive_summary.rs` - Abstractive summarization primitives

### Sentiment Analysis
- [x] `LexiconSentimentAnalyzer` - VADER-style with negation and intensifiers
- [x] Rule-based sentiment with modifier handling
- [x] Compound sentiment score
- [x] ML-based classifier adapter

### Text Classification
- [x] Feature extraction pipeline (bag-of-words, TF-IDF, n-gram combos)
- [x] `MultinomialNaiveBayes` (text-optimized with Laplace smoothing)
- [x] Classification dataset handling and evaluation
- [x] `text_classification.rs` - Classification workflow

### String Metrics and Phonetics
- [x] `LevenshteinMetric` (basic edit distance)
- [x] `DamerauLevenshteinMetric` - with transpositions, restricted/unrestricted modes
- [x] Jaro-Winkler similarity
- [x] `WeightedLevenshtein` - per-operation and per-character-pair costs
- [x] `WeightedDamerauLevenshtein` - with weighted transpositions
- [x] Cosine similarity, Jaccard similarity (set-based and n-gram)
- [x] `Soundex` phonetic encoding
- [x] `Metaphone` phonetic algorithm
- [x] `NYSIIS` phonetic algorithm
- [x] `advanced_distance.rs` - Word Mover's Distance, Soft Cosine, Conceptual Similarity

### Language Models
- [x] N-gram language model with Kneser-Ney smoothing
- [x] Character-level language model
- [x] Perplexity computation
- [x] `language_models` module

### Text Statistics and Readability
- [x] Flesch Reading Ease, Flesch-Kincaid Grade Level
- [x] Gunning Fog Index, SMOG Index, Coleman-Liau Index
- [x] Lexical diversity, type-token ratio
- [x] Word count, sentence count, average sentence length
- [x] `ReadabilityMetrics` struct with all common formulas

### Performance and Infrastructure
- [x] Rayon-based parallel tokenization and vectorization
- [x] `simd_ops.rs` - SIMD-accelerated string operations and distance computation
- [x] Memory-mapped corpus for large-scale processing
- [x] Sparse matrix output from all vectorizers
- [x] `parallel.rs` - Parallel corpus processing utilities
- [x] `information_theory` - Entropy, mutual information, KL divergence for text
- [x] `multilingual_ext.rs` - Language detection and multilingual utilities

### Testing and Quality
- [x] 160+ unit tests
- [x] 8 doctest examples
- [x] Zero-warning builds
- [x] All public APIs documented

## v0.4.0 Roadmap

### Transformer Tokenizers
- [x] SentencePiece tokenizer (Unigram LM-based, used by T5/LLaMA) — Implemented in v0.4.0
- [x] BERT/RoBERTa tokenizer (WordPiece with special tokens: [CLS], [SEP], [MASK]) — Implemented in v0.4.0 (`tokenizers/bert.rs`, `tokenizers/roberta.rs`)
- [x] GPT-2/GPT-4 tokenizer (BPE with byte-level encoding) — Implemented in v0.4.0 (`tokenization/byte_level_bpe.rs`, `gpt_bpe.rs`)
- [x] LLaMA tokenizer (SentencePiece + BPE hybrid) — Implemented in v0.4.0 (`tokenization/llama.rs`)
- [x] Tokenizer serialization compatible with HuggingFace `tokenizers` JSON format — Implemented in v0.4.0 (`tokenizers/hf_json.rs`)
- [x] Batch tokenization with padding and truncation — Implemented in v0.4.0

### Sentence Embeddings
- [x] Sentence-BERT-style aggregation (mean pooling of token embeddings) — Implemented in v0.4.0 (`embeddings/sentence.rs`)
- [x] Universal Sentence Encoder-style (transformer + pooling) (completed 2026-04-17)
  - **Goal:** `UniversalSentenceEncoder` with `PoolingStrategy::{ClsToken, Mean, Max, MeanSqrt, AttentionPooling, WeightedMean}`. Wraps token embeddings → sentence vector.
  - **Design:** `sentence_embeddings/universal.rs`. `encode(&self, tokens: &[usize]) -> Array1<f32>`. AttentionPooling: learnable query via 1-epoch pure-ndarray SGD. WeightedMean: log-IDF weights.
  - **Files:** `scirs2-text/src/sentence_embeddings/universal.rs` (new), `scirs2-text/src/sentence_embeddings/mod.rs`, `scirs2-text/tests/universal_encoder_tests.rs` (new).
  - **Tests:** `use_mean_pool_equals_average_of_token_vectors`, `use_cls_token_returns_first_token_embedding`, `use_attention_pool_weights_sum_to_one`, `use_max_pool_componentwise_max`, `use_normalized_output_has_unit_norm`.
  - **Risk:** AttentionPooling training via pure-ndarray SGD (no scirs2-autograd).
- [x] Contrastive sentence representation learning (SimCSE-style) (partial: 2026-04-17)
  - **Implemented:** `infonce.rs` (InfoNCE/NT-Xent loss, cosine similarity matrix, top-1 accuracy), `autograd_projection.rs` (pure-ndarray two-layer MLP with analytic backprop), `trainer.rs` (SimcseTrainer with unsupervised_step/supervised_step/fit_unsupervised/encode/encode_batch), integration test suite (`tests/simcse_tests.rs`, 14 tests).
  - **Deviation from design:** The projection head was implemented with pure-ndarray analytic backpropagation (closed-form gradient derivation) rather than `scirs2-autograd` as originally planned. Root cause: `scirs2-autograd` has multiple broken gradient paths — `extract_diag`, `ScalarMulOp`, and Adam's `Variable::compute` all call `.eval()` without a placeholder feeder, silently returning None gradients. Nine separate graph formulations were attempted; all hit the same class of upstream bug. The pure-ndarray MLP delivers identical training semantics (Glorot init, Bernoulli dropout, SGD with InfoNCE loss, L2 penalty) without the broken autograd layer.
  - **Test results (2026-04-17):** 26/26 tests PASS (`cargo nextest run -p scirs2-text -E 'test(simcse) | test(infonce)'`), zero clippy warnings.
  - **Remaining work:** Switch to `scirs2-autograd` once upstream gradient bugs are fixed (tracked in `scirs2-autograd/TODO.md`). Semantic similarity and cross-lingual sentence embeddings still pending.
- [x] Semantic similarity via sentence embeddings (completed 2026-04-17)
  - **Goal:** `semantic_similarity(s1, s2, encoder) -> f32` + `semantic_similarity_matrix(sentences, ...) -> Array2<f32>`. `PairwiseSimilarityMetric::{Cosine, Euclidean, DotProduct, Manhattan, Pearson}`.
  - **Design:** `sentence_embeddings/similarity.rs`. `SentenceEncoderLike` trait: `encode(&str) -> Array1<f32>` + `d_model() -> usize`. Implement for `SentenceEncoder`, `UniversalSentenceEncoder`.
  - **Files:** `scirs2-text/src/sentence_embeddings/similarity.rs` (new), `scirs2-text/src/sentence_embeddings/mod.rs`, `scirs2-text/tests/semantic_similarity_tests.rs` (new).
  - **Tests:** `cosine_of_identical_sentences_is_one`, `semantic_sim_matrix_is_symmetric_and_diag_one`, `euclidean_strictly_positive_except_identical`.
  - **Risk:** Low — pure arithmetic over encoder output.
- [x] Cross-lingual sentence embeddings (completed 2026-04-17)
  - **Goal:** `CrossLingualAligner` learning projection W: R^{d_src} → R^{d_tgt} from parallel sentences. `procrustes_align(x, y) -> Array2<f32>`. `AlignedEncoder` = monolingual encoder + projection.
  - **Design:** `sentence_embeddings/cross_lingual.rs`. Orthogonal Procrustes via `scirs2-linalg` SVD: SVD(YᵀX) = UΣVᵀ → W = UVᵀ. Optional contrastive refinement via existing `infonce.rs`.
  - **Files:** `scirs2-text/src/sentence_embeddings/cross_lingual.rs` (new), `scirs2-text/src/sentence_embeddings/mod.rs`, `scirs2-text/tests/cross_lingual_tests.rs` (new).
  - **Tests:** `procrustes_aligns_rotated_copies_exactly`, `aligned_encoder_preserves_norm`, `procrustes_identity_when_input_equals_output`.
  - **Risk:** Low — closed-form SVD, no training loop.

### Multilingual Models
- [x] Language-agnostic tokenization (Unicode-based, no language assumptions) (planned 2026-04-17)
  - **Goal:** Tokenizer following UAX #29 word boundaries (with script-aware tweaks for CJK), producing consistent output for mixed-language text, supporting NFC/NFKC normalization and configurable casing/punctuation policies.
  - **Design:** `tokenization/language_agnostic.rs` (new). Use `unicode-segmentation` (already a workspace dep) for grapheme + word boundaries. For CJK runs (detected by script property), emit per-character tokens instead of whole runs. Normalization via `unicode-normalization` (also a workspace dep). Provide `LanguageAgnosticTokenizer { normalize, lowercase, split_cjk_by_char, preserve_punctuation, max_token_len }`. Implement `Tokenizer` trait.
  - **Files:** `scirs2-text/src/tokenization/language_agnostic.rs` (new), `scirs2-text/src/tokenization/mod.rs` (re-export), `scirs2-text/Cargo.toml` (add `unicode-segmentation`, `unicode-normalization` as workspace deps if not already in crate deps), `scirs2-text/tests/language_agnostic_tokenizer_tests.rs` (new), `scirs2-text/TODO.md`.
  - **Prerequisites:** `unicode-segmentation` and `unicode-normalization` are confirmed workspace deps (root Cargo.toml lines 192-193).
  - **Tests:** `tok_mixed_latin_cjk`, `tok_nfc_normalization_idempotent`, `tok_cjk_split_by_char`, `tok_preserves_emoji_grapheme_clusters`, `tok_punctuation_policy_roundtrip`.
  - **Risk:** Small — UAX #29 is well-tested; pure Rust crates.
- [x] Multilingual vocabulary (shared BPE across 50+ languages) — Implemented in v0.4.0 (`tokenization/multilingual_bpe.rs`)
- [x] Cross-lingual NER transfer — Implemented in v0.4.0 (`crosslingual/mod.rs`)
- [x] Transliteration utilities for CJK and Cyrillic scripts (planned 2026-04-17)
  - **Goal:** Pure-Rust transliteration delivering (a) Hanyu Pinyin for simplified Chinese with tone-mark and numbered-tone styles, (b) Hepburn romaji for Japanese with hiragana/katakana coverage, (c) Cyrillic→Latin via GOST 2005 + BGN/PCGN + ALA-LC schemes.
  - **Design:** `transliteration/mod.rs` grows per-script submodules. Tables: hiragana+katakana → Hepburn (complete 46+dakuten+yoon set); Cyrillic 33 letters × 3 schemes; pinyin via CC-CEDICT subset (common-character table ≤5k entries, fallback to passthrough for unknowns). Tone logic: numeric or diacritic. Handle long vowels in Hepburn (ou, oo, macron). Provide a `Transliterator` trait with `transliterate(&self, s: &str) -> String` plus per-scheme configuration.
  - **Files:** `scirs2-text/src/transliteration/mod.rs`, `scirs2-text/src/transliteration/{pinyin.rs,hepburn.rs,cyrillic.rs,tables.rs}` (new), `scirs2-text/src/lib.rs` (re-export), `scirs2-text/tests/transliteration_tests.rs` (new), `scirs2-text/TODO.md`.
  - **Prerequisites:** none.
  - **Tests:** `pinyin_ni_hao_with_tone_marks`, `pinyin_numeric_tones`, `hepburn_konnichiwa`, `hepburn_long_vowel_macron`, `cyrillic_gost_moskva_yields_moskva`, `cyrillic_bgn_matches_published_examples`, `mixed_script_passthrough_unchanged`.
  - **Risk:** Pinyin requires a Chinese-character-to-pinyin dictionary; keep common-character table ≤5k entries and fall back to character passthrough for unknowns. Document the limitation. Pure-Rust data, no C deps.

### Enhanced Topic Modeling
- [x] Online LDA for streaming corpora — Implemented in v0.4.0
- [x] Hierarchical Dirichlet Process (HDP) for automatic topic number selection (completed 2026-04-17)
  - **Goal:** Production HDP topic model with automatic topic count selection via Chinese Restaurant Franchise (CRF) Gibbs sampler (Teh et al. 2006). Expose `fit`, `transform`, `topics`, `num_topics_inferred()`.
  - **Design:** Two-level DP: global DP(γ, H) with Dirichlet(η) base; per-document DP(α, G₀). Stick-breaking truncation T_max=50. `HdpConfig { alpha, gamma, eta, t_max, n_iter, burn_in, seed }`. RNG via `scirs2_core::random`.
  - **Files:** `scirs2-text/src/topic/hdp.rs` (new), `scirs2-text/src/topic/mod.rs`, `scirs2-text/tests/hdp_tests.rs` (new), `scirs2-text/TODO.md`.
  - **Tests:** `hdp_recovers_3_topics_on_synthetic_3_topic_corpus`, `hdp_convergence_log_likelihood_monotone_in_expectation`, `hdp_transform_respects_doc_length`, `hdp_deterministic_with_seed`, `hdp_empty_corpus_errors_cleanly`.
  - **Risk:** Gibbs burn-in slow on tiny corpora; use n_iter=100 default.
- [x] Correlated Topic Model (CTM) with logistic-normal prior — Implemented in v0.4.0 (`ctm/` module)
- [x] Dynamic Topic Model (DTM) for temporal analysis — Implemented in v0.4.0 (`dtm/` module)

### Neural NLP Integration
- [x] Bridge to `scirs2-neural` for transformer-based NLP
- [x] Attention visualization for transformer token attribution
- [x] BERT-style fine-tuning API for classification and NER
- [x] Named entity recognition via neural sequence labeler

### Evaluation and Benchmarks
- [x] CoNLL-2003 NER evaluation protocol (span-level F1) — Implemented in v0.4.0 (`evaluation/ner.rs`)
- [x] BLEU, ROUGE, METEOR for generation/summarization — Implemented in v0.4.0
- [x] STS benchmark integration (semantic textual similarity) (implemented 2026-04-17)
  - **Goal:** `sts_evaluate(encoder, pairs: &[(String, String, f32)]) -> StsReport { pearson, spearman, mse, predictions }`. Protocol only — no dataset download.
  - **Design:** `evaluation/sts.rs`. Cosine similarity → Pearson + Spearman correlation with gold labels. `load_sts_from_tsv(path)` helper. `StsDatasetFormat::{StsB, SICK, Sts12to16}` enum.
  - **Files:** `scirs2-text/src/evaluation/sts.rs` (new), `scirs2-text/src/evaluation/mod.rs`, `scirs2-text/tests/sts_evaluation_tests.rs` (new, 8 tests).
  - **Tests:** `sts_pearson_of_identical_predictions_is_one`, `sts_spearman_is_scale_invariant`, `sts_evaluate_on_toy_pairs_matches_hand_computed`, `sts_loads_tsv_correctly`, `sts_handles_empty_dataset_returns_error`, `sts_report_fields_consistent`, `sts_zero_vector_handled_without_panic`, `sts_mse_is_zero_for_perfect_match`.
  - **Test results (2026-04-17):** 8/8 PASS, zero clippy warnings.
- [x] Perplexity benchmarks on standard corpora (PTB, WikiText) (implemented 2026-04-17)
  - **Goal:** `perplexity_evaluate(model: &dyn LanguageModelLike, corpus: &[Vec<&str>]) -> PerplexityReport { per_sentence_perplexity, corpus_perplexity, total_tokens, total_log_prob }`.
  - **Design:** `evaluation/perplexity.rs`. PPL = exp(-1/N Σ log p(wᵢ|w<ᵢ)). `LanguageModelLike` trait. Implemented for `language_models::NgramLM` (Kneser-Ney) and `language_model::NgramModel` (Laplace/KN). `load_token_corpus` file helper (no network fetch).
  - **Files:** `scirs2-text/src/evaluation/perplexity.rs` (new), `scirs2-text/src/evaluation/mod.rs`, `scirs2-text/tests/perplexity_tests.rs` (new, 11 tests). Also wired `src/language_models/` into `lib.rs`.
  - **Tests:** `perplexity_uniform_model_equals_vocab_size`, `perplexity_of_perfect_predictor_is_one`, `perplexity_corpus_aggregates_token_log_probs`, `perplexity_empty_corpus_returns_error`, `perplexity_per_sentence_are_positive`, `perplexity_all_empty_sentences_returns_error`, `perplexity_per_sentence_count_equals_corpus_length`, `perplexity_higher_vocab_gives_higher_ppl`, `perplexity_fixed_prob_model_matches_formula`, `perplexity_empty_sentence_yields_nan_in_per_sentence`, `perplexity_with_ngram_model_from_language_models`, `perplexity_with_ngram_model_from_language_model`.
  - **Test results (2026-04-17):** 11/11 PASS, zero clippy warnings.

## v0.4.2 Additions (Wave 44)

### LLM-Compatible Tokenizer Features
- [x] LLM-compatible BPE tokenizer (byte-level, special tokens, chat templates) — `tokenizers/bpe_enhanced.rs`
- [x] `SpecialTokens` — GPT-2, LLaMA/Mistral, ChatML presets; custom token map
- [x] `BpeVocab` (enhanced) — u32 token IDs, special-token-aware, collision-safe registration
- [x] `ByteLevelBpe` — encode/decode with BOS/EOS injection and `skip_special_tokens`
- [x] Chat template formatting (ChatML, LLaMA-2, Alpaca, Simple styles) — `ChatTemplate` / `ChatStyle`
- [x] `Message` struct for multi-turn conversation representation
- [x] Approximate token counting via `ChatTemplate::count_tokens`

## Known Issues

- The `MultinomialNaiveBayes` import was previously duplicated in `text_classification.rs`; resolved in v0.3.1.
- LDA coherence computation uses the corpus vocabulary; very small corpora may produce unreliable scores — document minimum corpus size recommendations.
- `abstractive_summary.rs` provides primitives only; full abstractive summarization requires a neural sequence-to-sequence model from `scirs2-neural`.
- Word2Vec training convergence depends heavily on `min_count` and corpus size; add validation warnings for very small corpora.
- FastText character n-gram support may increase memory significantly for large vocabulary sizes; document memory tradeoffs.
