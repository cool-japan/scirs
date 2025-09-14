//! Advanced semantic similarity measures for text analysis
//!
//! This module provides advanced similarity metrics that go beyond simple
//! lexical matching to capture semantic relationships between texts.

use crate::embeddings::Word2Vec;
use crate::error::{Result, TextError};
use crate::tokenize::Tokenizer;
use ndarray::{Array1, Array2, ArrayView1};
use std::cmp;
use std::collections::{HashMap, HashSet};

/// Word Mover's Distance (WMD) for measuring semantic distance between documents
pub struct WordMoversDistance {
    embeddings: HashMap<String, Array1<f64>>,
}

impl WordMoversDistance {
    /// Create a new WMD calculator from pre-computed embeddings
    pub fn fromembeddings(embeddings: HashMap<String, Array1<f64>>) -> Self {
        Self { embeddings }
    }

    /// Create from a trained Word2Vec model
    pub fn from_word2vec(model: &Word2Vec, vocabulary: &[String]) -> Result<Self> {
        let mut embeddings = HashMap::new();

        for word in vocabulary {
            if let Ok(vector) = model.get_word_vector(word) {
                embeddings.insert(word.clone(), vector);
            }
        }

        if embeddings.is_empty() {
            return Err(TextError::EmbeddingError(
                "No embeddings could be extracted from the _model".into(),
            ));
        }

        Ok(Self { embeddings })
    }

    /// Calculate Word Mover's Distance between two texts
    pub fn distance(&self, text1: &str, text2: &str, tokenizer: &dyn Tokenizer) -> Result<f64> {
        let tokens1 = tokenizer.tokenize(text1)?;
        let tokens2 = tokenizer.tokenize(text2)?;

        // Filter tokens to only those with embeddings
        let tokens1: Vec<&str> = tokens1
            .iter()
            .map(|s| s.as_str())
            .filter(|t| self.embeddings.contains_key(*t))
            .collect();

        let tokens2: Vec<&str> = tokens2
            .iter()
            .map(|s| s.as_str())
            .filter(|t| self.embeddings.contains_key(*t))
            .collect();

        if tokens1.is_empty() || tokens2.is_empty() {
            return Err(TextError::InvalidInput(
                "No tokens with embeddings found in one or both texts".into(),
            ));
        }

        // Calculate normalized frequencies
        let freq1 = Self::calculate_frequencies(&tokens1);
        let freq2 = Self::calculate_frequencies(&tokens2);

        // Build distance matrix between all word pairs
        let n1 = freq1.len();
        let n2 = freq2.len();
        let mut distance_matrix = Array2::zeros((n1, n2));

        let words1: Vec<&String> = freq1.keys().collect();
        let words2: Vec<&String> = freq2.keys().collect();

        for (i, word1) in words1.iter().enumerate() {
            for (j, word2) in words2.iter().enumerate() {
                let embed1 = &self.embeddings[*word1];
                let embed2 = &self.embeddings[*word2];
                distance_matrix[[i, j]] = Self::euclidean_distance(embed1.view(), embed2.view());
            }
        }

        // Solve optimal transport problem (simplified greedy approach)
        // For a full implementation, you would use a linear programming solver
        self.solve_transport_greedy(&freq1, &freq2, &distance_matrix, &words1, &words2)
    }

    /// Calculate word frequencies (normalized)
    fn calculate_frequencies(tokens: &[&str]) -> HashMap<String, f64> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for &token in tokens {
            *counts.entry(token.to_string()).or_insert(0) += 1;
        }

        let total = tokens.len() as f64;
        counts
            .into_iter()
            .map(|(word, count)| (word, count as f64 / total))
            .collect()
    }

    /// Calculate Euclidean distance between two vectors
    fn euclidean_distance(v1: ArrayView1<f64>, v2: ArrayView1<f64>) -> f64 {
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Greedy approximation of optimal transport
    fn solve_transport_greedy(
        &self,
        freq1: &HashMap<String, f64>,
        freq2: &HashMap<String, f64>,
        distances: &Array2<f64>,
        words1: &[&String],
        words2: &[&String],
    ) -> Result<f64> {
        let mut remaining1: HashMap<String, f64> = freq1.clone();
        let mut remaining2: HashMap<String, f64> = freq2.clone();
        let mut total_cost = 0.0;

        // Create a list of all edges sorted by distance
        let mut edges = Vec::new();
        for (i, word1) in words1.iter().enumerate() {
            for (j, word2) in words2.iter().enumerate() {
                edges.push((distances[[i, j]], word1.to_string(), word2.to_string()));
            }
        }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Greedily assign mass
        for (distance, word1, word2) in edges {
            let mass1 = remaining1.get(&word1).copied().unwrap_or(0.0);
            let mass2 = remaining2.get(&word2).copied().unwrap_or(0.0);

            if mass1 > 0.0 && mass2 > 0.0 {
                let transported = mass1.min(mass2);
                total_cost += transported * distance;

                remaining1.insert(word1.clone(), mass1 - transported);
                remaining2.insert(word2.clone(), mass2 - transported);
            }
        }

        Ok(total_cost)
    }
}

/// Soft Cosine Similarity using word similarities
pub struct SoftCosineSimilarity {
    similarity_matrix: HashMap<(String, String), f64>,
}

impl SoftCosineSimilarity {
    /// Create from pre-computed word similarities
    pub fn new(_similaritymatrix: HashMap<(String, String), f64>) -> Self {
        Self {
            similarity_matrix: _similaritymatrix,
        }
    }

    /// Create from word embeddings by computing cosine similarities
    pub fn fromembeddings(embeddings: &HashMap<String, Array1<f64>>) -> Self {
        let mut similarity_matrix = HashMap::new();

        let words: Vec<&String> = embeddings.keys().collect();
        for (i, word1) in words.iter().enumerate() {
            for word2 in words.iter().skip(i) {
                let sim =
                    Self::cosine_similarity(embeddings[*word1].view(), embeddings[*word2].view());
                similarity_matrix.insert(((*word1).clone(), (*word2).clone()), sim);
                if word1 != word2 {
                    similarity_matrix.insert(((*word2).clone(), (*word1).clone()), sim);
                }
            }
        }

        Self { similarity_matrix }
    }

    /// Calculate soft cosine similarity between two texts
    pub fn similarity(&self, text1: &str, text2: &str, tokenizer: &dyn Tokenizer) -> Result<f64> {
        let tokens1 = tokenizer.tokenize(text1)?;
        let tokens2 = tokenizer.tokenize(text2)?;

        // Calculate TF vectors
        let tf1 = Self::calculate_tf(&tokens1);
        let tf2 = Self::calculate_tf(&tokens2);

        // Get all unique words
        let mut all_words: HashSet<String> = HashSet::new();
        all_words.extend(tf1.keys().cloned());
        all_words.extend(tf2.keys().cloned());

        // Calculate soft cosine similarity
        let mut numerator = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        for word1 in &all_words {
            for word2 in &all_words {
                let similarity = self.get_similarity(word1, word2);

                // Calculate numerator (cross-product between text1 and text2)
                let w1_from_tf1 = tf1.get(word1).copied().unwrap_or(0.0);
                let w2_from_tf2 = tf2.get(word2).copied().unwrap_or(0.0);
                numerator += w1_from_tf1 * w2_from_tf2 * similarity;
            }
        }

        // Calculate norms separately
        for word1 in &all_words {
            let weight1 = tf1.get(word1).copied().unwrap_or(0.0);
            for word2 in &all_words {
                let weight2 = tf1.get(word2).copied().unwrap_or(0.0);
                let similarity = self.get_similarity(word1, word2);
                norm1 += weight1 * weight2 * similarity;
            }
        }

        for word1 in &all_words {
            let weight1 = tf2.get(word1).copied().unwrap_or(0.0);
            for word2 in &all_words {
                let weight2 = tf2.get(word2).copied().unwrap_or(0.0);
                let similarity = self.get_similarity(word1, word2);
                norm2 += weight1 * weight2 * similarity;
            }
        }

        if norm1 > 0.0 && norm2 > 0.0 {
            Ok(numerator / (norm1.sqrt() * norm2.sqrt()))
        } else {
            Ok(0.0)
        }
    }

    /// Get similarity between two words
    fn get_similarity(&self, word1: &str, word2: &str) -> f64 {
        if word1 == word2 {
            return 1.0;
        }

        self.similarity_matrix
            .get(&(word1.to_string(), word2.to_string()))
            .copied()
            .unwrap_or(0.0)
    }

    /// Calculate term frequencies
    fn calculate_tf(tokens: &[String]) -> HashMap<String, f64> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for token in tokens {
            *counts.entry(token.clone()).or_insert(0) += 1;
        }

        let max_count = counts.values().max().copied().unwrap_or(1) as f64;
        counts
            .into_iter()
            .map(|(word, count)| (word, count as f64 / max_count))
            .collect()
    }

    /// Calculate cosine similarity between vectors
    fn cosine_similarity(v1: ArrayView1<f64>, v2: ArrayView1<f64>) -> f64 {
        let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot / (norm1 * norm2)
        } else {
            0.0
        }
    }
}

/// Weighted Jaccard similarity with custom term weights
pub struct WeightedJaccard {
    weights: HashMap<String, f64>,
}

impl Default for WeightedJaccard {
    fn default() -> Self {
        Self::new()
    }
}

impl WeightedJaccard {
    /// Create with uniform weights
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
        }
    }

    /// Create with custom term weights (e.g., IDF weights)
    pub fn with_weights(weights: HashMap<String, f64>) -> Self {
        Self { weights }
    }

    /// Calculate weighted Jaccard similarity
    pub fn similarity(&self, text1: &str, text2: &str, tokenizer: &dyn Tokenizer) -> Result<f64> {
        let tokens1 = tokenizer.tokenize(text1)?;
        let tokens2 = tokenizer.tokenize(text2)?;

        let set1: HashSet<String> = tokens1.into_iter().collect();
        let set2: HashSet<String> = tokens2.into_iter().collect();

        let intersection: HashSet<&String> = set1.intersection(&set2).collect();
        let union: HashSet<&String> = set1.union(&set2).collect();

        if union.is_empty() {
            return Ok(1.0);
        }

        // Calculate weighted intersection and union
        let weighted_intersection: f64 =
            intersection.iter().map(|term| self.get_weight(term)).sum();

        let weighted_union: f64 = union.iter().map(|term| self.get_weight(term)).sum();

        Ok(weighted_intersection / weighted_union)
    }

    /// Get weight for a term
    fn get_weight(&self, term: &str) -> f64 {
        self.weights.get(term).copied().unwrap_or(1.0)
    }
}

/// Longest Common Subsequence (LCS) based similarity
pub struct LcsSimilarity;

impl LcsSimilarity {
    /// Calculate LCS-based similarity between two texts
    pub fn similarity(text1: &str, text2: &str, tokenizer: &dyn Tokenizer) -> Result<f64> {
        let tokens1 = tokenizer.tokenize(text1)?;
        let tokens2 = tokenizer.tokenize(text2)?;

        let lcs_length = Self::lcs_length(&tokens1, &tokens2);
        let max_length = cmp::max(tokens1.len(), tokens2.len()) as f64;

        if max_length == 0.0 {
            Ok(1.0)
        } else {
            Ok(lcs_length as f64 / max_length)
        }
    }

    /// Calculate length of longest common subsequence
    fn lcs_length(seq1: &[String], seq2: &[String]) -> usize {
        let m = seq1.len();
        let n = seq2.len();
        let mut dp = vec![vec![0; n + 1]; m + 1];

        for i in 1..=m {
            for j in 1..=n {
                if seq1[i - 1] == seq2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = cmp::max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        dp[m][n]
    }
}

/// Semantic edit distance for measuring text similarity with semantic operations
pub struct SemanticEditDistance {
    embeddings: HashMap<String, Array1<f64>>,
    synonymthreshold: f64,
}

impl SemanticEditDistance {
    /// Create new semantic edit distance calculator
    pub fn new(embeddings: HashMap<String, Array1<f64>>, synonymthreshold: f64) -> Self {
        Self {
            embeddings,
            synonymthreshold,
        }
    }

    /// Calculate semantic edit distance between two texts
    pub fn distance(&self, text1: &str, text2: &str, tokenizer: &dyn Tokenizer) -> Result<f64> {
        let tokens1 = tokenizer.tokenize(text1)?;
        let tokens2 = tokenizer.tokenize(text2)?;

        let m = tokens1.len();
        let n = tokens2.len();
        let mut dp = Array2::zeros((m + 1, n + 1));

        // Initialize base cases
        for i in 0..=m {
            dp[[i, 0]] = i as f64;
        }
        for j in 0..=n {
            dp[[0, j]] = j as f64;
        }

        // Fill DP table with semantic costs
        for i in 1..=m {
            for j in 1..=n {
                let substitution_cost = if tokens1[i - 1] == tokens2[j - 1] {
                    0.0
                } else {
                    self.semantic_substitution_cost(&tokens1[i - 1], &tokens2[j - 1])
                };

                dp[[i, j]] = (dp[[i - 1, j]] + 1.0) // deletion
                    .min(dp[[i, j - 1]] + 1.0) // insertion
                    .min(dp[[i - 1, j - 1]] + substitution_cost); // substitution
            }
        }

        Ok(dp[[m, n]])
    }

    /// Calculate semantic substitution cost between two words
    fn semantic_substitution_cost(&self, word1: &str, word2: &str) -> f64 {
        if let (Some(embed1), Some(embed2)) =
            (self.embeddings.get(word1), self.embeddings.get(word2))
        {
            let similarity = Self::cosine_similarity(embed1.view(), embed2.view());
            if similarity >= self.synonymthreshold {
                0.5 // Reduced cost for similar words
            } else {
                1.0 - similarity // Cost inversely related to similarity
            }
        } else {
            1.0 // Full substitution cost for unknown words
        }
    }

    /// Calculate cosine similarity between vectors
    fn cosine_similarity(v1: ArrayView1<f64>, v2: ArrayView1<f64>) -> f64 {
        let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot / (norm1 * norm2)
        } else {
            0.0
        }
    }

    /// Convert distance to normalized similarity (0-1)
    pub fn similarity(&self, text1: &str, text2: &str, tokenizer: &dyn Tokenizer) -> Result<f64> {
        let distance = self.distance(text1, text2, tokenizer)?;
        let tokens1 = tokenizer.tokenize(text1)?;
        let tokens2 = tokenizer.tokenize(text2)?;
        let max_length = cmp::max(tokens1.len(), tokens2.len()) as f64;

        if max_length == 0.0 {
            Ok(1.0)
        } else {
            Ok(1.0 - (distance / max_length))
        }
    }
}

/// Sentence embedding-based similarity for document-level comparisons
pub struct SentenceEmbeddingSimilarity {
    embeddings: HashMap<String, Array1<f64>>,
    pooling_strategy: PoolingStrategy,
}

/// Strategy for pooling word embeddings into sentence embeddings
#[derive(Debug, Clone)]
pub enum PoolingStrategy {
    /// Average all word embeddings
    Mean,
    /// Maximum across all dimensions
    Max,
    /// Weighted average using IDF weights
    WeightedMean(HashMap<String, f64>),
}

impl SentenceEmbeddingSimilarity {
    /// Create new sentence embedding similarity calculator
    pub fn new(
        embeddings: HashMap<String, Array1<f64>>,
        pooling_strategy: PoolingStrategy,
    ) -> Self {
        Self {
            embeddings,
            pooling_strategy,
        }
    }

    /// Calculate similarity between two texts using sentence embeddings
    pub fn similarity(&self, text1: &str, text2: &str, tokenizer: &dyn Tokenizer) -> Result<f64> {
        let embed1 = self.get_sentence_embedding(text1, tokenizer)?;
        let embed2 = self.get_sentence_embedding(text2, tokenizer)?;

        Ok(Self::cosine_similarity(embed1.view(), embed2.view()))
    }

    /// Generate sentence embedding from text
    fn get_sentence_embedding(&self, text: &str, tokenizer: &dyn Tokenizer) -> Result<Array1<f64>> {
        let tokens = tokenizer.tokenize(text)?;

        // Filter tokens that have embeddings
        let validembeddings: Vec<&Array1<f64>> = tokens
            .iter()
            .filter_map(|token| self.embeddings.get(token))
            .collect();

        if validembeddings.is_empty() {
            return Err(TextError::InvalidInput(
                "No valid embeddings found for tokens".into(),
            ));
        }

        let embed_dim = validembeddings[0].len();

        match &self.pooling_strategy {
            PoolingStrategy::Mean => {
                let mut result = Array1::zeros(embed_dim);
                for embedding in &validembeddings {
                    result += *embedding;
                }
                Ok(result / validembeddings.len() as f64)
            }
            PoolingStrategy::Max => {
                let mut result = Array1::from_elem(embed_dim, f64::NEG_INFINITY);
                for embedding in &validembeddings {
                    for (i, &val) in embedding.iter().enumerate() {
                        if val > result[i] {
                            result[i] = val;
                        }
                    }
                }
                Ok(result)
            }
            PoolingStrategy::WeightedMean(weights) => {
                let mut result = Array1::zeros(embed_dim);
                let mut total_weight = 0.0;

                for (token, embedding) in tokens.iter().zip(&validembeddings) {
                    let weight = weights.get(token).copied().unwrap_or(1.0);
                    result = result + *embedding * weight;
                    total_weight += weight;
                }

                if total_weight > 0.0 {
                    Ok(result / total_weight)
                } else {
                    Ok(result / validembeddings.len() as f64)
                }
            }
        }
    }

    /// Calculate cosine similarity between vectors
    fn cosine_similarity(v1: ArrayView1<f64>, v2: ArrayView1<f64>) -> f64 {
        let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot / (norm1 * norm2)
        } else {
            0.0
        }
    }
}

/// N-gram based semantic similarity with skip-grams and weighted matching
pub struct NGramSemanticSimilarity {
    n: usize,
    skipdistance: usize,
    embeddings: HashMap<String, Array1<f64>>,
    ngramweights: HashMap<usize, f64>,
}

impl NGramSemanticSimilarity {
    /// Create new N-gram semantic similarity calculator
    pub fn new(n: usize, skipdistance: usize, embeddings: HashMap<String, Array1<f64>>) -> Self {
        let mut ngram_weights = HashMap::new();
        // Higher order n-grams get higher weights
        for i in 1..=n {
            ngram_weights.insert(i, i as f64);
        }

        Self {
            n,
            skipdistance,
            embeddings,
            ngramweights: ngram_weights,
        }
    }

    /// Calculate similarity using semantic n-grams
    pub fn similarity(&self, text1: &str, text2: &str, tokenizer: &dyn Tokenizer) -> Result<f64> {
        let tokens1 = tokenizer.tokenize(text1)?;
        let tokens2 = tokenizer.tokenize(text2)?;

        let mut total_similarity = 0.0;
        let mut total_weight = 0.0;

        // Calculate similarity for each n-gram size
        for ngram_size in 1..=self.n {
            let ngrams1 = self.extract_ngrams(&tokens1, ngram_size);
            let ngrams2 = self.extract_ngrams(&tokens2, ngram_size);

            let similarity = self.calculate_ngram_similarity(&ngrams1, &ngrams2);
            let weight = self.ngramweights.get(&ngram_size).copied().unwrap_or(1.0);

            total_similarity += similarity * weight;
            total_weight += weight;
        }

        Ok(if total_weight > 0.0 {
            total_similarity / total_weight
        } else {
            0.0
        })
    }

    /// Extract n-grams with skip-grams
    fn extract_ngrams(&self, tokens: &[String], n: usize) -> Vec<Vec<String>> {
        let mut ngrams = Vec::new();

        if tokens.len() < n {
            return ngrams;
        }

        // Standard n-grams
        for i in 0..=tokens.len() - n {
            ngrams.push(tokens[i..i + n].to_vec());
        }

        // Skip-grams (if skipdistance > 0)
        if self.skipdistance > 0 && n > 1 {
            for i in 0..tokens.len() {
                for skip in 1..=self.skipdistance {
                    if i + skip + n - 1 < tokens.len() {
                        let mut skipgram = vec![tokens[i].clone()];
                        for j in 1..n {
                            skipgram.push(tokens[i + skip + j - 1].clone());
                        }
                        ngrams.push(skipgram);
                    }
                }
            }
        }

        ngrams
    }

    /// Calculate similarity between two sets of n-grams
    fn calculate_ngram_similarity(&self, ngrams1: &[Vec<String>], ngrams2: &[Vec<String>]) -> f64 {
        if ngrams1.is_empty() || ngrams2.is_empty() {
            return 0.0;
        }

        let mut total_similarity = 0.0;
        let mut count = 0;

        for ngram1 in ngrams1 {
            let mut max_similarity = 0.0f64;
            for ngram2 in ngrams2 {
                let similarity = self.ngram_semantic_similarity(ngram1, ngram2);
                max_similarity = max_similarity.max(similarity);
            }
            total_similarity += max_similarity;
            count += 1;
        }

        total_similarity / count as f64
    }

    /// Calculate semantic similarity between two n-grams
    fn ngram_semantic_similarity(&self, ngram1: &[String], ngram2: &[String]) -> f64 {
        if ngram1.len() != ngram2.len() {
            return 0.0;
        }

        let mut total_similarity = 0.0;
        for (word1, word2) in ngram1.iter().zip(ngram2.iter()) {
            if word1 == word2 {
                total_similarity += 1.0;
            } else if let (Some(embed1), Some(embed2)) =
                (self.embeddings.get(word1), self.embeddings.get(word2))
            {
                total_similarity += Self::cosine_similarity(embed1.view(), embed2.view()).max(0.0);
            }
        }

        total_similarity / ngram1.len() as f64
    }

    /// Calculate cosine similarity between vectors
    fn cosine_similarity(v1: ArrayView1<f64>, v2: ArrayView1<f64>) -> f64 {
        let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot / (norm1 * norm2)
        } else {
            0.0
        }
    }
}

/// Semantic similarity ensemble that combines multiple metrics
pub struct SemanticSimilarityEnsemble {
    wmd: Option<WordMoversDistance>,
    soft_cosine: Option<SoftCosineSimilarity>,
    weighted_jaccard: Option<WeightedJaccard>,
    semantic_edit: Option<SemanticEditDistance>,
    sentence_embedding: Option<SentenceEmbeddingSimilarity>,
    ngram_semantic: Option<NGramSemanticSimilarity>,
    weights: HashMap<String, f64>,
}

impl Default for SemanticSimilarityEnsemble {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticSimilarityEnsemble {
    /// Create a new ensemble
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("wmd".to_string(), 0.15);
        weights.insert("soft_cosine".to_string(), 0.20);
        weights.insert("weighted_jaccard".to_string(), 0.15);
        weights.insert("lcs".to_string(), 0.10);
        weights.insert("semantic_edit".to_string(), 0.15);
        weights.insert("sentence_embedding".to_string(), 0.15);
        weights.insert("ngram_semantic".to_string(), 0.10);

        Self {
            wmd: None,
            soft_cosine: None,
            weighted_jaccard: None,
            semantic_edit: None,
            sentence_embedding: None,
            ngram_semantic: None,
            weights,
        }
    }

    /// Set Word Mover's Distance component
    pub fn with_wmd(mut self, wmd: WordMoversDistance) -> Self {
        self.wmd = Some(wmd);
        self
    }

    /// Set Soft Cosine Similarity component
    pub fn with_soft_cosine(mut self, softcosine: SoftCosineSimilarity) -> Self {
        self.soft_cosine = Some(softcosine);
        self
    }

    /// Set Weighted Jaccard component
    pub fn with_weighted_jaccard(mut self, weightedjaccard: WeightedJaccard) -> Self {
        self.weighted_jaccard = Some(weightedjaccard);
        self
    }

    /// Set Semantic Edit Distance component
    pub fn with_semantic_edit(mut self, semanticedit: SemanticEditDistance) -> Self {
        self.semantic_edit = Some(semanticedit);
        self
    }

    /// Set Sentence Embedding Similarity component
    pub fn with_sentence_embedding(
        mut self,
        sentence_embedding: SentenceEmbeddingSimilarity,
    ) -> Self {
        self.sentence_embedding = Some(sentence_embedding);
        self
    }

    /// Set N-gram Semantic Similarity component
    pub fn with_ngram_semantic(mut self, ngramsemantic: NGramSemanticSimilarity) -> Self {
        self.ngram_semantic = Some(ngramsemantic);
        self
    }

    /// Set custom weights for components
    pub fn with_weights(mut self, weights: HashMap<String, f64>) -> Self {
        self.weights = weights;
        self
    }

    /// Calculate ensemble similarity
    pub fn similarity(&self, text1: &str, text2: &str, tokenizer: &dyn Tokenizer) -> Result<f64> {
        let mut scores = HashMap::new();
        let mut total_weight = 0.0;

        // Calculate WMD similarity (converted from distance)
        if let Some(ref wmd) = self.wmd {
            if let Ok(distance) = wmd.distance(text1, text2, tokenizer) {
                // Convert distance to similarity (simple inverse)
                let similarity = 1.0 / (1.0 + distance);
                scores.insert("wmd".to_string(), similarity);
                total_weight += self.weights.get("wmd").copied().unwrap_or(0.0);
            }
        }

        // Calculate soft cosine similarity
        if let Some(ref soft_cosine) = self.soft_cosine {
            if let Ok(similarity) = soft_cosine.similarity(text1, text2, tokenizer) {
                scores.insert("soft_cosine".to_string(), similarity);
                total_weight += self.weights.get("soft_cosine").copied().unwrap_or(0.0);
            }
        }

        // Calculate weighted Jaccard similarity
        if let Some(ref weighted_jaccard) = self.weighted_jaccard {
            if let Ok(similarity) = weighted_jaccard.similarity(text1, text2, tokenizer) {
                scores.insert("weighted_jaccard".to_string(), similarity);
                total_weight += self.weights.get("weighted_jaccard").copied().unwrap_or(0.0);
            }
        }

        // Calculate LCS similarity
        if let Ok(lcs_sim) = LcsSimilarity::similarity(text1, text2, tokenizer) {
            scores.insert("lcs".to_string(), lcs_sim);
            total_weight += self.weights.get("lcs").copied().unwrap_or(0.0);
        }

        // Calculate semantic edit distance similarity
        if let Some(ref semantic_edit) = self.semantic_edit {
            if let Ok(similarity) = semantic_edit.similarity(text1, text2, tokenizer) {
                scores.insert("semantic_edit".to_string(), similarity);
                total_weight += self.weights.get("semantic_edit").copied().unwrap_or(0.0);
            }
        }

        // Calculate sentence embedding similarity
        if let Some(ref sentence_embedding) = self.sentence_embedding {
            if let Ok(similarity) = sentence_embedding.similarity(text1, text2, tokenizer) {
                scores.insert("sentence_embedding".to_string(), similarity);
                total_weight += self
                    .weights
                    .get("sentence_embedding")
                    .copied()
                    .unwrap_or(0.0);
            }
        }

        // Calculate n-gram semantic similarity
        if let Some(ref ngram_semantic) = self.ngram_semantic {
            if let Ok(similarity) = ngram_semantic.similarity(text1, text2, tokenizer) {
                scores.insert("ngram_semantic".to_string(), similarity);
                total_weight += self.weights.get("ngram_semantic").copied().unwrap_or(0.0);
            }
        }

        if scores.is_empty() {
            return Err(TextError::InvalidInput(
                "No similarity metrics could be calculated".into(),
            ));
        }

        // Calculate weighted average
        let weighted_sum: f64 = scores
            .iter()
            .map(|(name, &score)| score * self.weights.get(name).copied().unwrap_or(0.0))
            .sum();

        Ok(weighted_sum / total_weight)
    }
}

/// Conceptual similarity based on hierarchical relationships (WordNet-style)
pub struct ConceptualSimilarity {
    concept_hierarchy: HashMap<String, ConceptNode>,
    #[allow(dead_code)]
    _maxdepth: usize,
}

/// Node in a concept hierarchy for semantic similarity
#[derive(Debug, Clone)]
pub struct ConceptNode {
    /// Name of the concept
    pub concept: String,
    /// Parent concepts
    pub parents: Vec<String>,
    /// Child concepts
    pub children: Vec<String>,
    /// Depth in the hierarchy
    pub depth: usize,
    /// Information content score
    pub information_content: f64,
}

impl ConceptualSimilarity {
    /// Create new conceptual similarity calculator
    pub fn new(_maxdepth: usize) -> Self {
        Self {
            concept_hierarchy: HashMap::new(),
            _maxdepth,
        }
    }

    /// Add a concept to the hierarchy
    pub fn add_concept(&mut self, concept: ConceptNode) {
        self.concept_hierarchy
            .insert(concept.concept.clone(), concept);
    }

    /// Build a simple concept hierarchy from word relationships
    pub fn build_simple_hierarchy(_wordrelations: Vec<(String, String)>) -> Self {
        let mut similarity = Self::new(10);
        let mut concept_map = HashMap::new();

        // Build nodes from _relations
        for (child, parent) in _wordrelations {
            let child_node = concept_map
                .entry(child.clone())
                .or_insert_with(|| ConceptNode {
                    concept: child.clone(),
                    parents: Vec::new(),
                    children: Vec::new(),
                    depth: 0,
                    information_content: 1.0,
                });
            child_node.parents.push(parent.clone());

            let parent_node = concept_map
                .entry(parent.clone())
                .or_insert_with(|| ConceptNode {
                    concept: parent.clone(),
                    parents: Vec::new(),
                    children: Vec::new(),
                    depth: 0,
                    information_content: 1.0,
                });
            parent_node.children.push(child.clone());
        }

        // Calculate depths and information content
        similarity.concept_hierarchy = concept_map;
        similarity.calculate_depths();
        similarity
    }

    /// Calculate path-based similarity between two concepts
    pub fn path_similarity(&self, concept1: &str, concept2: &str) -> f64 {
        if concept1 == concept2 {
            return 1.0;
        }

        if let Some(lcs) = self.lowest_common_subsumer(concept1, concept2) {
            let depth1 = self.get_depth(concept1).unwrap_or(0);
            let depth2 = self.get_depth(concept2).unwrap_or(0);
            let lcs_depth = self.get_depth(&lcs).unwrap_or(0);

            let path_length = (depth1 - lcs_depth) + (depth2 - lcs_depth);
            1.0 / (1.0 + path_length as f64)
        } else {
            0.0
        }
    }

    /// Calculate Wu-Palmer similarity
    pub fn wu_palmer_similarity(&self, concept1: &str, concept2: &str) -> f64 {
        if concept1 == concept2 {
            return 1.0;
        }

        if let Some(lcs) = self.lowest_common_subsumer(concept1, concept2) {
            let depth1 = self.get_depth(concept1).unwrap_or(0);
            let depth2 = self.get_depth(concept2).unwrap_or(0);
            let lcs_depth = self.get_depth(&lcs).unwrap_or(0);

            (2.0 * lcs_depth as f64) / (depth1 + depth2) as f64
        } else {
            0.0
        }
    }

    /// Calculate Resnik similarity (information content based)
    pub fn resnik_similarity(&self, concept1: &str, concept2: &str) -> f64 {
        if concept1 == concept2 {
            return self.get_information_content(concept1);
        }

        if let Some(lcs) = self.lowest_common_subsumer(concept1, concept2) {
            self.get_information_content(&lcs)
        } else {
            0.0
        }
    }

    /// Find the lowest common subsumer (most specific common ancestor)
    fn lowest_common_subsumer(&self, concept1: &str, concept2: &str) -> Option<String> {
        let ancestors1 = self.get_ancestors(concept1);
        let ancestors2 = self.get_ancestors(concept2);

        // Find common ancestors
        let common: Vec<&String> = ancestors1.intersection(&ancestors2).collect();

        // Return the most specific (deepest) common ancestor
        common
            .into_iter()
            .max_by_key(|concept| self.get_depth(concept).unwrap_or(0))
            .cloned()
    }

    /// Get all ancestors of a concept
    fn get_ancestors(&self, concept: &str) -> HashSet<String> {
        let mut ancestors = HashSet::new();
        self.collect_ancestors(concept, &mut ancestors);
        ancestors
    }

    /// Recursively collect ancestors
    fn collect_ancestors(&self, concept: &str, ancestors: &mut HashSet<String>) {
        ancestors.insert(concept.to_string());

        if let Some(node) = self.concept_hierarchy.get(concept) {
            for parent in &node.parents {
                if !ancestors.contains(parent) {
                    self.collect_ancestors(parent, ancestors);
                }
            }
        }
    }

    /// Get depth of a concept
    fn get_depth(&self, concept: &str) -> Option<usize> {
        self.concept_hierarchy.get(concept).map(|node| node.depth)
    }

    /// Get information content of a concept
    fn get_information_content(&self, concept: &str) -> f64 {
        self.concept_hierarchy
            .get(concept)
            .map(|node| node.information_content)
            .unwrap_or(0.0)
    }

    /// Calculate depths for all concepts
    fn calculate_depths(&mut self) {
        // Find root concepts (no parents)
        let roots: Vec<String> = self
            .concept_hierarchy
            .values()
            .filter(|node| node.parents.is_empty())
            .map(|node| node.concept.clone())
            .collect();

        // BFS to calculate depths
        for root in roots {
            self.set_depth_recursive(&root, 1);
        }
    }

    /// Recursively set depths
    fn set_depth_recursive(&mut self, concept: &str, depth: usize) {
        if let Some(node) = self.concept_hierarchy.get_mut(concept) {
            if node.depth < depth {
                node.depth = depth;
                let children = node.children.clone();
                for child in children {
                    self.set_depth_recursive(&child, depth + 1);
                }
            }
        }
    }

    /// Calculate conceptual similarity between two texts
    pub fn text_similarity(
        &self,
        text1: &str,
        text2: &str,
        tokenizer: &dyn Tokenizer,
    ) -> Result<f64> {
        let tokens1 = tokenizer.tokenize(text1)?;
        let tokens2 = tokenizer.tokenize(text2)?;

        if tokens1.is_empty() || tokens2.is_empty() {
            return Ok(0.0);
        }

        let mut total_similarity = 0.0;
        let mut count = 0;

        for token1 in &tokens1 {
            for token2 in &tokens2 {
                if self.concept_hierarchy.contains_key(token1)
                    && self.concept_hierarchy.contains_key(token2)
                {
                    total_similarity += self.wu_palmer_similarity(token1, token2);
                    count += 1;
                }
            }
        }

        Ok(if count > 0 {
            total_similarity / count as f64
        } else {
            0.0
        })
    }
}

/// Distributional semantic similarity based on co-occurrence statistics
pub struct DistributionalSimilarity {
    cooccurrence_matrix: HashMap<(String, String), f64>,
    word_frequencies: HashMap<String, f64>,
    total_words: f64,
}

impl DistributionalSimilarity {
    /// Create new distributional similarity calculator
    pub fn new() -> Self {
        Self {
            cooccurrence_matrix: HashMap::new(),
            word_frequencies: HashMap::new(),
            total_words: 0.0,
        }
    }

    /// Build co-occurrence matrix from a corpus
    pub fn from_corpus(
        corpus: &[String],
        tokenizer: &dyn Tokenizer,
        windowsize: usize,
    ) -> Result<Self> {
        let mut similarity = Self::new();

        for document in corpus {
            let tokens = tokenizer.tokenize(document)?;
            similarity.update_counts(&tokens, windowsize);
        }

        similarity.calculate_pmis();
        Ok(similarity)
    }

    /// Update co-occurrence counts
    fn update_counts(&mut self, tokens: &[String], windowsize: usize) {
        for (i, target) in tokens.iter().enumerate() {
            // Update word frequency
            *self.word_frequencies.entry(target.clone()).or_insert(0.0) += 1.0;
            self.total_words += 1.0;

            // Update co-occurrence counts within window
            let start = i.saturating_sub(windowsize);
            let end = (i + windowsize + 1).min(tokens.len());

            #[allow(clippy::needless_range_loop)]
            for j in start..end {
                if i != j {
                    let context = &tokens[j];
                    let key = if target <= context {
                        (target.clone(), context.clone())
                    } else {
                        (context.clone(), target.clone())
                    };
                    *self.cooccurrence_matrix.entry(key).or_insert(0.0) += 1.0;
                }
            }
        }
    }

    /// Calculate PMI (Pointwise Mutual Information) values
    fn calculate_pmis(&mut self) {
        let mut pmi_matrix = HashMap::new();

        for ((word1, word2), &cooccur_count) in &self.cooccurrence_matrix {
            let p_word1 = self.word_frequencies.get(word1).unwrap_or(&0.0) / self.total_words;
            let p_word2 = self.word_frequencies.get(word2).unwrap_or(&0.0) / self.total_words;
            let p_together = cooccur_count / self.total_words;

            if p_word1 > 0.0 && p_word2 > 0.0 && p_together > 0.0 {
                let pmi = (p_together / (p_word1 * p_word2)).ln();
                let ppmi = pmi.max(0.0); // Positive PMI
                pmi_matrix.insert((word1.clone(), word2.clone()), ppmi);
            }
        }

        self.cooccurrence_matrix = pmi_matrix;
    }

    /// Calculate distributional similarity between two words
    pub fn word_similarity(&self, word1: &str, word2: &str) -> f64 {
        if word1 == word2 {
            return 1.0;
        }

        let key = if word1 <= word2 {
            (word1.to_string(), word2.to_string())
        } else {
            (word2.to_string(), word1.to_string())
        };

        self.cooccurrence_matrix.get(&key).copied().unwrap_or(0.0)
    }

    /// Calculate distributional similarity between two texts
    pub fn text_similarity(
        &self,
        text1: &str,
        text2: &str,
        tokenizer: &dyn Tokenizer,
    ) -> Result<f64> {
        let tokens1 = tokenizer.tokenize(text1)?;
        let tokens2 = tokenizer.tokenize(text2)?;

        if tokens1.is_empty() || tokens2.is_empty() {
            return Ok(0.0);
        }

        let mut similarities = Vec::new();

        for token1 in &tokens1 {
            for token2 in &tokens2 {
                let sim = self.word_similarity(token1, token2);
                if sim > 0.0 {
                    similarities.push(sim);
                }
            }
        }

        Ok(if similarities.is_empty() {
            0.0
        } else {
            similarities.iter().sum::<f64>() / similarities.len() as f64
        })
    }

    /// Get the most similar words to a given word
    pub fn most_similar(&self, word: &str, topk: usize) -> Vec<(String, f64)> {
        let mut similarities: Vec<(String, f64)> = self
            .cooccurrence_matrix
            .iter()
            .filter_map(|((w1, w2), &score)| {
                if w1 == word {
                    Some((w2.clone(), score))
                } else if w2 == word {
                    Some((w1.clone(), score))
                } else {
                    None
                }
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(topk);
        similarities
    }
}

impl Default for DistributionalSimilarity {
    fn default() -> Self {
        Self::new()
    }
}

/// Topic-based similarity using document topic distributions
pub struct TopicBasedSimilarity {
    topicdistributions: HashMap<String, Array1<f64>>,
    _similaritymetric: TopicSimilarityMetric,
}

/// Metrics for comparing topic distributions
#[derive(Debug, Clone)]
pub enum TopicSimilarityMetric {
    /// Cosine similarity between topic distributions
    Cosine,
    /// Jensen-Shannon divergence
    JensenShannon,
    /// Hellinger distance
    Hellinger,
}

impl TopicBasedSimilarity {
    /// Create new topic-based similarity calculator
    pub fn new(_similaritymetric: TopicSimilarityMetric) -> Self {
        Self {
            topicdistributions: HashMap::new(),
            _similaritymetric,
        }
    }

    /// Add topic distribution for a document
    pub fn add_document_topics(&mut self, doc_id: String, topicdistribution: Array1<f64>) {
        self.topicdistributions.insert(doc_id, topicdistribution);
    }

    /// Calculate topic-based similarity between two documents
    pub fn similarity(&self, doc_id1: &str, docid2: &str) -> Result<f64> {
        let topics1 = self
            .topicdistributions
            .get(doc_id1)
            .ok_or_else(|| TextError::InvalidInput(format!("Document {doc_id1} not found")))?;
        let topics2 = self
            .topicdistributions
            .get(docid2)
            .ok_or_else(|| TextError::InvalidInput(format!("Document {docid2} not found")))?;

        match self._similaritymetric {
            TopicSimilarityMetric::Cosine => {
                Ok(self.cosine_similarity(topics1.view(), topics2.view()))
            }
            TopicSimilarityMetric::JensenShannon => {
                Ok(1.0 - self.jensen_shannon_divergence(topics1.view(), topics2.view()))
            }
            TopicSimilarityMetric::Hellinger => {
                Ok(1.0 - self.hellinger_distance(topics1.view(), topics2.view()))
            }
        }
    }

    /// Calculate cosine similarity between topic distributions
    fn cosine_similarity(&self, v1: ArrayView1<f64>, v2: ArrayView1<f64>) -> f64 {
        let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot / (norm1 * norm2)
        } else {
            0.0
        }
    }

    /// Calculate Jensen-Shannon divergence
    fn jensen_shannon_divergence(&self, v1: ArrayView1<f64>, v2: ArrayView1<f64>) -> f64 {
        let m: Array1<f64> = (&v1.to_owned() + &v2.to_owned()) / 2.0;

        let kl1 = self.kl_divergence(v1, m.view());
        let kl2 = self.kl_divergence(v2, m.view());

        (kl1 + kl2) / 2.0
    }

    /// Calculate KL divergence
    fn kl_divergence(&self, p: ArrayView1<f64>, q: ArrayView1<f64>) -> f64 {
        p.iter()
            .zip(q.iter())
            .filter(|(pi, qi)| **pi > 0.0 && **qi > 0.0)
            .map(|(pi, qi)| pi * (pi / qi).ln())
            .sum()
    }

    /// Calculate Hellinger distance
    fn hellinger_distance(&self, v1: ArrayView1<f64>, v2: ArrayView1<f64>) -> f64 {
        let sum_sq_diff: f64 = v1
            .iter()
            .zip(v2.iter())
            .map(|(a, b)| (a.sqrt() - b.sqrt()).powi(2))
            .sum();

        (sum_sq_diff / 2.0).sqrt()
    }

    /// Find most similar documents to a given document
    pub fn most_similar_documents(&self, doc_id: &str, topk: usize) -> Result<Vec<(String, f64)>> {
        if !self.topicdistributions.contains_key(doc_id) {
            return Err(TextError::InvalidInput(format!(
                "Document {doc_id} not found"
            )));
        }

        let mut similarities: Vec<(String, f64)> = self
            .topicdistributions
            .keys()
            .filter(|&_id| _id != doc_id)
            .filter_map(|_id| {
                self.similarity(doc_id, _id)
                    .ok()
                    .map(|sim| (_id.clone(), sim))
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(topk);
        Ok(similarities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenize::WordTokenizer;

    #[test]
    fn test_lcs_similarity() {
        let tokenizer = WordTokenizer::default();

        let sim1 =
            LcsSimilarity::similarity("the quick brown fox", "the fast brown fox", &tokenizer)
                .unwrap();
        assert!(sim1 > 0.5); // Should have high similarity

        let sim2 = LcsSimilarity::similarity("hello world", "goodbye moon", &tokenizer).unwrap();
        assert!(sim2 < 0.3); // Should have low similarity
    }

    #[test]
    fn test_weighted_jaccard() {
        let tokenizer = WordTokenizer::default();
        let mut weights = HashMap::new();
        weights.insert("important".to_string(), 5.0);
        weights.insert("the".to_string(), 0.1);

        let weighted_jaccard = WeightedJaccard::with_weights(weights);

        let sim = weighted_jaccard
            .similarity("the important document", "the important paper", &tokenizer)
            .unwrap();

        // Should give high weight to "important" which is common
        assert!(sim > 0.7);
    }

    #[test]
    fn test_soft_cosine_similarity() {
        // Create mock embeddings
        let mut embeddings = HashMap::new();
        embeddings.insert("cat".to_string(), arr1(&[1.0, 0.0]));
        embeddings.insert("dog".to_string(), arr1(&[0.9, 0.1]));
        embeddings.insert("car".to_string(), arr1(&[0.0, 1.0]));

        let soft_cosine = SoftCosineSimilarity::fromembeddings(&embeddings);
        let tokenizer = WordTokenizer::default();

        let sim = soft_cosine
            .similarity("cat dog", "dog cat", &tokenizer)
            .unwrap();

        // Should be very high as they contain the same words
        assert!(sim > 0.9);
    }

    fn arr1(data: &[f64]) -> Array1<f64> {
        Array1::from_vec(data.to_vec())
    }

    #[test]
    fn test_semantic_edit_distance() {
        let mut embeddings = HashMap::new();
        embeddings.insert("cat".to_string(), arr1(&[1.0, 0.0]));
        embeddings.insert("kitten".to_string(), arr1(&[0.9, 0.1]));
        embeddings.insert("dog".to_string(), arr1(&[0.0, 1.0]));
        embeddings.insert("puppy".to_string(), arr1(&[0.1, 0.9]));

        let semantic_edit = SemanticEditDistance::new(embeddings, 0.8);
        let tokenizer = WordTokenizer::default();

        // Similar words should have lower distance
        let sim1 = semantic_edit
            .similarity("cat", "kitten", &tokenizer)
            .unwrap();
        let sim2 = semantic_edit.similarity("cat", "dog", &tokenizer).unwrap();

        assert!(sim1 > sim2); // Cat and kitten are more similar than cat and dog
    }

    #[test]
    fn test_sentence_embedding_similarity() {
        let mut embeddings = HashMap::new();
        embeddings.insert("the".to_string(), arr1(&[0.1, 0.1, 0.1]));
        embeddings.insert("quick".to_string(), arr1(&[1.0, 0.0, 0.0]));
        embeddings.insert("brown".to_string(), arr1(&[0.0, 1.0, 0.0]));
        embeddings.insert("fox".to_string(), arr1(&[0.0, 0.0, 1.0]));
        embeddings.insert("fast".to_string(), arr1(&[0.9, 0.1, 0.0]));

        let sentence_sim = SentenceEmbeddingSimilarity::new(embeddings, PoolingStrategy::Mean);
        let tokenizer = WordTokenizer::default();

        let sim = sentence_sim
            .similarity("the quick brown fox", "the fast brown fox", &tokenizer)
            .unwrap();

        assert!(sim > 0.7); // Should be high due to similar embeddings for quick/fast
    }

    #[test]
    fn test_sentence_embedding_pooling_strategies() {
        let mut embeddings = HashMap::new();
        embeddings.insert("high".to_string(), arr1(&[1.0, 0.0]));
        embeddings.insert("low".to_string(), arr1(&[0.0, 1.0]));

        // Test Mean pooling
        let mean_sim = SentenceEmbeddingSimilarity::new(embeddings.clone(), PoolingStrategy::Mean);
        let tokenizer = WordTokenizer::default();

        // Test Max pooling
        let max_sim = SentenceEmbeddingSimilarity::new(embeddings.clone(), PoolingStrategy::Max);

        // Test Weighted Mean pooling
        let mut weights = HashMap::new();
        weights.insert("high".to_string(), 2.0);
        weights.insert("low".to_string(), 1.0);
        let weighted_sim =
            SentenceEmbeddingSimilarity::new(embeddings, PoolingStrategy::WeightedMean(weights));

        let text1 = "high low";
        let text2 = "high high";

        let mean_result = mean_sim.similarity(text1, text2, &tokenizer).unwrap();
        let max_result = max_sim.similarity(text1, text2, &tokenizer).unwrap();
        let weighted_result = weighted_sim.similarity(text1, text2, &tokenizer).unwrap();

        // All should be valid similarities
        assert!((0.0..=1.0).contains(&mean_result));
        assert!((0.0..=1.0).contains(&max_result));
        assert!((0.0..=1.0).contains(&weighted_result));
    }

    #[test]
    fn test_ngram_semantic_similarity() {
        let mut embeddings = HashMap::new();
        embeddings.insert("machine".to_string(), arr1(&[1.0, 0.0, 0.0]));
        embeddings.insert("learning".to_string(), arr1(&[0.0, 1.0, 0.0]));
        embeddings.insert("artificial".to_string(), arr1(&[0.9, 0.1, 0.0]));
        embeddings.insert("intelligence".to_string(), arr1(&[0.1, 0.9, 0.0]));
        embeddings.insert("computer".to_string(), arr1(&[0.8, 0.0, 0.2]));
        embeddings.insert("science".to_string(), arr1(&[0.0, 0.8, 0.2]));

        let ngram_sim = NGramSemanticSimilarity::new(2, 1, embeddings);
        let tokenizer = WordTokenizer::default();

        let sim = ngram_sim
            .similarity("machine learning", "artificial intelligence", &tokenizer)
            .unwrap();

        assert!(sim > 0.0); // Should have some similarity due to related concepts
    }

    #[test]
    fn test_enhanced_ensemble() {
        let mut embeddings = HashMap::new();
        embeddings.insert("cat".to_string(), arr1(&[1.0, 0.0]));
        embeddings.insert("kitten".to_string(), arr1(&[0.9, 0.1]));
        embeddings.insert("dog".to_string(), arr1(&[0.0, 1.0]));

        let semantic_edit = SemanticEditDistance::new(embeddings.clone(), 0.8);
        let sentence_embedding =
            SentenceEmbeddingSimilarity::new(embeddings.clone(), PoolingStrategy::Mean);
        let ngram_semantic = NGramSemanticSimilarity::new(1, 0, embeddings);

        let ensemble = SemanticSimilarityEnsemble::new()
            .with_semantic_edit(semantic_edit)
            .with_sentence_embedding(sentence_embedding)
            .with_ngram_semantic(ngram_semantic);

        let tokenizer = WordTokenizer::default();

        let sim = ensemble.similarity("cat", "kitten", &tokenizer).unwrap();
        assert!(sim > 0.0 && sim <= 1.0);
    }

    #[test]
    fn test_ngram_skip_grams() {
        let mut embeddings = HashMap::new();
        embeddings.insert("the".to_string(), arr1(&[0.1, 0.1]));
        embeddings.insert("quick".to_string(), arr1(&[1.0, 0.0]));
        embeddings.insert("brown".to_string(), arr1(&[0.0, 1.0]));
        embeddings.insert("fox".to_string(), arr1(&[0.5, 0.5]));

        // Test with skip-grams enabled
        let ngram_sim = NGramSemanticSimilarity::new(2, 1, embeddings);
        let tokenizer = WordTokenizer::default();

        let sim = ngram_sim
            .similarity(
                "the quick brown fox",
                "the brown quick fox", // Different order but same words
                &tokenizer,
            )
            .unwrap();

        // Should still have some similarity due to skip-grams
        assert!(sim > 0.3);
    }

    #[test]
    fn test_conceptual_similarity() {
        // Build a simple concept hierarchy
        let relations = vec![
            ("cat".to_string(), "animal".to_string()),
            ("dog".to_string(), "animal".to_string()),
            ("animal".to_string(), "living_thing".to_string()),
            ("plant".to_string(), "living_thing".to_string()),
            ("rose".to_string(), "plant".to_string()),
        ];

        let conceptual_sim = ConceptualSimilarity::build_simple_hierarchy(relations);

        // Test path similarity
        let sim_cat_dog = conceptual_sim.path_similarity("cat", "dog");
        let sim_cat_rose = conceptual_sim.path_similarity("cat", "rose");

        // Cat and dog should be more similar than cat and rose (closer in hierarchy)
        assert!(sim_cat_dog > sim_cat_rose);
        assert!(sim_cat_dog > 0.0);

        // Test Wu-Palmer similarity
        let wu_palmer_sim = conceptual_sim.wu_palmer_similarity("cat", "dog");
        assert!(wu_palmer_sim > 0.0 && wu_palmer_sim <= 1.0);

        // Test Resnik similarity
        let resnik_sim = conceptual_sim.resnik_similarity("cat", "dog");
        assert!(resnik_sim >= 0.0);

        // Test text similarity
        let tokenizer = WordTokenizer::default();
        let text_sim = conceptual_sim
            .text_similarity("cat runs", "dog runs", &tokenizer)
            .unwrap();
        assert!(text_sim > 0.0);
    }

    #[test]
    fn test_distributional_similarity() {
        let tokenizer = WordTokenizer::default();

        // Create a small corpus
        let corpus = vec![
            "cat sits on mat".to_string(),
            "dog runs in park".to_string(),
            "cat plays with dog".to_string(),
            "dog sits on mat".to_string(),
            "cat runs in park".to_string(),
        ];

        let dist_sim = DistributionalSimilarity::from_corpus(&corpus, &tokenizer, 2).unwrap();

        // Test word similarity
        let sim_cat_dog = dist_sim.word_similarity("cat", "dog");
        let sim_cat_mat = dist_sim.word_similarity("cat", "mat");

        // Words that appear in similar contexts should have higher similarity
        assert!(sim_cat_dog >= 0.0);
        assert!(sim_cat_mat >= 0.0);

        // Test text similarity
        let text_sim = dist_sim
            .text_similarity("cat runs", "dog runs", &tokenizer)
            .unwrap();
        assert!(text_sim >= 0.0);

        // Test most similar words
        let similar_words = dist_sim.most_similar("cat", 3);
        assert!(similar_words.len() <= 3);
        for (word, score) in similar_words {
            assert!(score >= 0.0);
            assert!(!word.is_empty());
        }
    }

    #[test]
    fn test_topic_based_similarity() {
        let mut topic_sim = TopicBasedSimilarity::new(TopicSimilarityMetric::Cosine);

        // Add some mock topic distributions
        let topics1 = arr1(&[0.8, 0.1, 0.1]); // Primarily topic 0
        let topics2 = arr1(&[0.7, 0.2, 0.1]); // Also primarily topic 0
        let topics3 = arr1(&[0.1, 0.8, 0.1]); // Primarily topic 1

        topic_sim.add_document_topics("doc1".to_string(), topics1);
        topic_sim.add_document_topics("doc2".to_string(), topics2);
        topic_sim.add_document_topics("doc3".to_string(), topics3);

        // Test cosine similarity
        let sim_1_2 = topic_sim.similarity("doc1", "doc2").unwrap();
        let sim_1_3 = topic_sim.similarity("doc1", "doc3").unwrap();

        // Documents with similar topic distributions should be more similar
        assert!(sim_1_2 > sim_1_3);
        assert!(sim_1_2 > 0.5);
        assert!(sim_1_3 >= 0.0);

        // Test most similar documents
        let similar_docs = topic_sim.most_similar_documents("doc1", 2).unwrap();
        assert_eq!(similar_docs.len(), 2);
        assert_eq!(similar_docs[0].0, "doc2"); // Should be most similar
        assert!(similar_docs[0].1 > similar_docs[1].1);
    }

    #[test]
    fn test_topic_similaritymetrics() {
        // Test different similarity metrics
        let mut cosine_sim = TopicBasedSimilarity::new(TopicSimilarityMetric::Cosine);
        let mut js_sim = TopicBasedSimilarity::new(TopicSimilarityMetric::JensenShannon);
        let mut hellinger_sim = TopicBasedSimilarity::new(TopicSimilarityMetric::Hellinger);

        let topics1 = arr1(&[0.6, 0.3, 0.1]);
        let topics2 = arr1(&[0.5, 0.4, 0.1]);

        cosine_sim.add_document_topics("doc1".to_string(), topics1.clone());
        cosine_sim.add_document_topics("doc2".to_string(), topics2.clone());

        js_sim.add_document_topics("doc1".to_string(), topics1.clone());
        js_sim.add_document_topics("doc2".to_string(), topics2.clone());

        hellinger_sim.add_document_topics("doc1".to_string(), topics1);
        hellinger_sim.add_document_topics("doc2".to_string(), topics2);

        let cosine_result = cosine_sim.similarity("doc1", "doc2").unwrap();
        let js_result = js_sim.similarity("doc1", "doc2").unwrap();
        let hellinger_result = hellinger_sim.similarity("doc1", "doc2").unwrap();

        // All similarity measures should return valid values
        assert!((0.0..=1.0).contains(&cosine_result));
        assert!((0.0..=1.0).contains(&js_result));
        assert!((0.0..=1.0).contains(&hellinger_result));
    }

    #[test]
    fn test_conceptual_similarity_edge_cases() {
        let conceptual_sim = ConceptualSimilarity::new(5);

        // Test with unknown concepts
        let sim = conceptual_sim.path_similarity("unknown1", "unknown2");
        assert_eq!(sim, 0.0);

        // Test identical concepts
        let relations = vec![("cat".to_string(), "animal".to_string())];
        let conceptual_sim = ConceptualSimilarity::build_simple_hierarchy(relations);
        let sim = conceptual_sim.path_similarity("cat", "cat");
        assert_eq!(sim, 1.0);
    }

    #[test]
    fn test_distributional_similarity_edge_cases() {
        let tokenizer = WordTokenizer::default();

        // Empty corpus
        let empty_corpus = vec![];
        let dist_sim = DistributionalSimilarity::from_corpus(&empty_corpus, &tokenizer, 2).unwrap();

        let sim = dist_sim.word_similarity("word1", "word2");
        assert_eq!(sim, 0.0);

        // Test with identical words
        let sim_same = dist_sim.word_similarity("word", "word");
        assert_eq!(sim_same, 1.0);

        // Test text similarity with empty texts
        let text_sim = dist_sim.text_similarity("", "", &tokenizer).unwrap();
        assert_eq!(text_sim, 0.0);
    }

    #[test]
    fn test_enhanced_semantic_similarity_ensemble() {
        // Test ensemble with new similarity measures
        let mut embeddings = HashMap::new();
        embeddings.insert("cat".to_string(), arr1(&[1.0, 0.0]));
        embeddings.insert("dog".to_string(), arr1(&[0.8, 0.2]));
        embeddings.insert("animal".to_string(), arr1(&[0.9, 0.1]));

        // Create conceptual similarity
        let relations = vec![
            ("cat".to_string(), "animal".to_string()),
            ("dog".to_string(), "animal".to_string()),
        ];
        let conceptual_sim = ConceptualSimilarity::build_simple_hierarchy(relations);

        // Create distributional similarity
        let corpus = vec![
            "cat animal runs".to_string(),
            "dog animal plays".to_string(),
        ];
        let tokenizer = WordTokenizer::default();
        let dist_sim = DistributionalSimilarity::from_corpus(&corpus, &tokenizer, 2).unwrap();

        // Test individual components
        let conceptual_result = conceptual_sim
            .text_similarity("cat", "dog", &tokenizer)
            .unwrap();
        let distributional_result = dist_sim.text_similarity("cat", "dog", &tokenizer).unwrap();

        assert!(conceptual_result >= 0.0);
        assert!(distributional_result >= 0.0);

        // Both should show some similarity since cat and dog are related concepts
        assert!(conceptual_result > 0.0 || distributional_result > 0.0);
    }
}
