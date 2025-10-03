//! Memory-efficient streaming and memory-mapped text processing
//!
//! This module provides utilities for processing large text corpora that don't fit in memory
//! using streaming and memory-mapped file techniques.

use crate::error::{Result, TextError};
use crate::sparse::{CsrMatrix, SparseMatrixBuilder, SparseVector};
use crate::tokenize::{Tokenizer, WordTokenizer};
use crate::vocabulary::Vocabulary;
use memmap2::{Mmap, MmapOptions};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// Advanced-advanced streaming metrics for performance monitoring
#[derive(Debug, Clone, Default)]
pub struct AdvancedStreamingMetrics {
    /// Total documents processed
    pub documents_processed: usize,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Peak memory usage in bytes
    pub peak_memory_usage: usize,
    /// Current throughput (documents per second)
    pub throughput: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Memory efficiency score
    pub memory_efficiency: f64,
}

/// Memory usage tracking for streaming operations
#[derive(Debug, Default)]
#[allow(dead_code)]
struct MemoryUsageTracker {
    current_usage: usize,
    peak_usage: usize,
}

impl MemoryUsageTracker {
    #[allow(dead_code)]
    fn update_usage(&mut self, current: usize) {
        self.current_usage = current;
        if current > self.peak_usage {
            self.peak_usage = current;
        }
    }
}

/// Memory-mapped corpus for efficient large file processing
pub struct MemoryMappedCorpus {
    mmap: Arc<Mmap>,
    line_offsets: Vec<usize>,
}

impl MemoryMappedCorpus {
    /// Create a new memory-mapped corpus from a file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| TextError::IoError(format!("Failed to open file: {e}")))?;

        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| TextError::IoError(format!("Failed to memory map file: {e}")))?
        };

        // Build line offset index
        let line_offsets = Self::build_line_index(&mmap);

        Ok(Self {
            mmap: Arc::new(mmap),
            line_offsets,
        })
    }

    /// Build an index of line offsets for fast access
    fn build_line_index(mmap: &Mmap) -> Vec<usize> {
        let mut offsets = vec![0];
        let data = mmap.as_ref();

        for (i, &byte) in data.iter().enumerate() {
            if byte == b'\n' {
                offsets.push(i + 1);
            }
        }

        offsets
    }

    /// Get the number of documents (lines) in the corpus
    pub fn num_documents(&self) -> usize {
        self.line_offsets.len().saturating_sub(1)
    }

    /// Get a specific document by index
    pub fn get_document(&self, index: usize) -> Result<&str> {
        if index >= self.num_documents() {
            return Err(TextError::InvalidInput(format!(
                "Document index {index} out of range"
            )));
        }

        let start = self.line_offsets[index];
        let end = if index + 1 < self.line_offsets.len() {
            self.line_offsets[index + 1].saturating_sub(1) // Remove newline
        } else {
            self.mmap.len()
        };

        let data = &self.mmap[start..end];
        std::str::from_utf8(data)
            .map_err(|e| TextError::IoError(format!("Invalid UTF-8 in document: {e}")))
    }

    /// Iterate over all documents
    pub fn iter(&self) -> CorpusIterator {
        CorpusIterator {
            corpus: self,
            current: 0,
        }
    }

    /// Process documents in parallel chunks
    pub fn parallel_process<F, R>(&self, chunksize: usize, processor: F) -> Result<Vec<R>>
    where
        F: Fn(&[&str]) -> Result<R> + Send + Sync,
        R: Send,
    {
        use scirs2_core::parallel_ops::*;

        let num_docs = self.num_documents();
        let num_chunks = num_docs.div_ceil(chunksize);

        (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunksize;
                let end = ((chunk_idx + 1) * chunksize).min(num_docs);

                let mut docs = Vec::with_capacity(end - start);
                for i in start..end {
                    docs.push(self.get_document(i)?);
                }

                processor(&docs)
            })
            .collect()
    }
}

/// Iterator over documents in a memory-mapped corpus
pub struct CorpusIterator<'a> {
    corpus: &'a MemoryMappedCorpus,
    current: usize,
}

impl<'a> Iterator for CorpusIterator<'a> {
    type Item = Result<&'a str>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.corpus.num_documents() {
            return None;
        }

        let doc = self.corpus.get_document(self.current);
        self.current += 1;
        Some(doc)
    }
}

/// Streaming text processor for handling arbitrarily large files
pub struct StreamingTextProcessor<T: Tokenizer> {
    tokenizer: T,
    buffer_size: usize,
}

impl<T: Tokenizer> StreamingTextProcessor<T> {
    /// Create a new streaming processor
    pub fn new(tokenizer: T) -> Self {
        Self {
            tokenizer,
            buffer_size: 1024 * 1024, // 1MB default buffer
        }
    }

    /// Set custom buffer size
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Process a file line by line
    pub fn process_lines<P, F, R>(&self, path: P, processor: F) -> Result<Vec<R>>
    where
        P: AsRef<Path>,
        F: FnMut(&str, usize) -> Result<Option<R>>,
    {
        let file = File::open(path)
            .map_err(|e| TextError::IoError(format!("Failed to open file: {e}")))?;

        let reader = BufReader::with_capacity(self.buffer_size, file);
        self.process_reader_lines(reader, processor)
    }

    /// Process lines from any reader
    pub fn process_reader_lines<R: BufRead, F, U>(
        &self,
        reader: R,
        mut processor: F,
    ) -> Result<Vec<U>>
    where
        F: FnMut(&str, usize) -> Result<Option<U>>,
    {
        let mut results = Vec::new();

        for (line_num, line_result) in reader.lines().enumerate() {
            let line =
                line_result.map_err(|e| TextError::IoError(format!("Error reading line: {e}")))?;

            if let Some(result) = processor(&line, line_num)? {
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Build vocabulary from a streaming corpus
    pub fn build_vocabulary_streaming<P: AsRef<Path>>(
        &self,
        path: P,
        min_count: usize,
    ) -> Result<Vocabulary> {
        let mut token_counts = HashMap::<String, usize>::new();

        // First pass: _count tokens
        self.process_lines(&path, |line, _line_num| {
            let tokens = self.tokenizer.tokenize(line)?;
            for token in tokens {
                *token_counts.entry(token).or_insert(0) += 1;
            }
            Ok(None::<()>)
        })?;

        // Build vocabulary with high-frequency tokens
        let mut vocab = Vocabulary::new();
        for (token, count) in &token_counts {
            if *count >= min_count {
                vocab.add_token(token);
            }
        }

        Ok(vocab)
    }
}

impl StreamingTextProcessor<WordTokenizer> {
    /// Create a streaming processor with default word tokenizer
    pub fn with_default_tokenizer() -> Self {
        Self::new(WordTokenizer::default())
    }
}

/// Streaming vectorizer for creating sparse matrices from large corpora
pub struct StreamingVectorizer {
    vocabulary: Vocabulary,
    chunksize: usize,
}

impl StreamingVectorizer {
    /// Create a new streaming vectorizer
    pub fn new(vocabulary: Vocabulary) -> Self {
        Self {
            vocabulary,
            chunksize: 1000, // Process 1000 documents at a time
        }
    }

    /// Set chunk size for processing
    pub fn with_chunksize(mut self, size: usize) -> Self {
        self.chunksize = size;
        self
    }

    /// Transform a streaming corpus into a sparse matrix
    pub fn transform_streaming<P, T>(&self, path: P, tokenizer: &T) -> Result<CsrMatrix>
    where
        P: AsRef<Path>,
        T: Tokenizer,
    {
        let mut builder = SparseMatrixBuilder::new(self.vocabulary.len());

        let file = std::fs::File::open(path)
            .map_err(|e| TextError::IoError(format!("Failed to open file: {e}")))?;
        let reader = std::io::BufReader::new(file);

        for line in reader.lines() {
            let line = line.map_err(|e| TextError::IoError(format!("Error reading line: {e}")))?;
            let tokens = tokenizer.tokenize(&line)?;
            let sparse_vec = self.tokens_to_sparse_vector(&tokens)?;
            builder.add_row(sparse_vec)?;
        }

        Ok(builder.build())
    }

    /// Convert tokens to sparse vector
    fn tokens_to_sparse_vector(&self, tokens: &[String]) -> Result<SparseVector> {
        let mut counts = std::collections::HashMap::new();

        for token in tokens {
            if let Some(idx) = self.vocabulary.get_index(token) {
                *counts.entry(idx).or_insert(0.0) += 1.0;
            }
        }

        let mut indices: Vec<usize> = counts.keys().copied().collect();
        indices.sort_unstable();

        let values: Vec<f64> = indices.iter().map(|&idx| counts[&idx]).collect();

        let sparse_vec = SparseVector::fromindices_values(indices, values, self.vocabulary.len());

        Ok(sparse_vec)
    }
}

/// Chunked corpus reader for processing files in manageable chunks
pub struct ChunkedCorpusReader {
    file: File,
    chunksize: usize,
    position: u64,
    file_size: u64,
}

impl ChunkedCorpusReader {
    /// Create a new chunked reader
    pub fn new<P: AsRef<Path>>(path: P, chunksize: usize) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| TextError::IoError(format!("Failed to open file: {e}")))?;

        let file_size = file
            .metadata()
            .map_err(|e| TextError::IoError(format!("Failed to get file metadata: {e}")))?
            .len();

        Ok(Self {
            file,
            chunksize,
            position: 0,
            file_size,
        })
    }

    /// Read the next chunk of complete lines
    pub fn next_chunk(&mut self) -> Result<Option<Vec<String>>> {
        if self.position >= self.file_size {
            return Ok(None);
        }

        self.file
            .seek(SeekFrom::Start(self.position))
            .map_err(|e| TextError::IoError(format!("Failed to seek: {e}")))?;

        let mut buffer = vec![0u8; self.chunksize];
        let bytes_read = self
            .file
            .read(&mut buffer)
            .map_err(|e| TextError::IoError(format!("Failed to read chunk: {e}")))?;

        if bytes_read == 0 {
            return Ok(None);
        }

        buffer.truncate(bytes_read);

        // Find the last newline to ensure complete lines
        let last_newline = buffer.iter().rposition(|&b| b == b'\n');

        let chunk_end = if let Some(pos) = last_newline {
            pos + 1
        } else if self.position + bytes_read as u64 >= self.file_size {
            bytes_read
        } else {
            // No newline found and not at end of file, need to read more
            return Err(TextError::IoError(
                "Chunk size too small to contain a complete line".into(),
            ));
        };

        let chunk_str = std::str::from_utf8(&buffer[..chunk_end])
            .map_err(|e| TextError::IoError(format!("Invalid UTF-8: {e}")))?;

        let lines: Vec<String> = chunk_str.lines().map(|s| s.to_string()).collect();

        self.position += chunk_end as u64;

        Ok(Some(lines))
    }

    /// Reset to the beginning of the file
    pub fn reset(&mut self) -> Result<()> {
        self.position = 0;
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|e| TextError::IoError(format!("Failed to seek: {e}")))?;
        Ok(())
    }
}

/// Multi-file corpus for handling large distributed corpora
pub struct MultiFileCorpus {
    files: Vec<MemoryMappedCorpus>,
    file_boundaries: Vec<usize>, // Cumulative document counts
    total_documents: usize,
}

impl MultiFileCorpus {
    /// Create corpus from multiple files
    pub fn from_files<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        let mut files = Vec::new();
        let mut file_boundaries = vec![0];
        let mut total_documents = 0;

        for path in paths {
            let corpus = MemoryMappedCorpus::from_file(path)?;
            let doc_count = corpus.num_documents();
            total_documents += doc_count;

            files.push(corpus);
            file_boundaries.push(total_documents);
        }

        Ok(Self {
            files,
            file_boundaries,
            total_documents,
        })
    }

    /// Get total number of documents across all files
    pub fn num_documents(&self) -> usize {
        self.total_documents
    }

    /// Get document by global index
    pub fn get_document(&self, globalindex: usize) -> Result<&str> {
        if globalindex >= self.total_documents {
            return Err(TextError::InvalidInput(format!(
                "Document _index {globalindex} out of range"
            )));
        }

        // Find which file contains this document
        let file_idx = match self.file_boundaries.binary_search(&(globalindex + 1)) {
            Ok(idx) => {
                // Found exact match, means we're at a boundary
                // The document belongs to the previous file
                if idx == 0 {
                    0
                } else {
                    idx - 1
                }
            }
            Err(idx) => {
                // Not found, idx is insertion point
                if idx == 0 {
                    0
                } else {
                    idx - 1
                }
            }
        };

        let local_index = globalindex.saturating_sub(self.file_boundaries[file_idx]);
        self.files[file_idx].get_document(local_index)
    }

    /// Iterate over all documents across files
    pub fn iter(&self) -> MultiFileIterator {
        MultiFileIterator {
            corpus: self,
            current: 0,
        }
    }

    /// Get random sample of documents
    pub fn random_sample(&self, samplesize: usize, seed: u64) -> Result<Vec<&str>> {
        use std::collections::HashSet;

        if samplesize > self.total_documents {
            return Err(TextError::InvalidInput(
                "Sample _size larger than corpus _size".into(),
            ));
        }

        let mut rng = seed;
        let mut selected = HashSet::new();
        let mut samples = Vec::new();

        while samples.len() < samplesize {
            // Simple LCG for deterministic random numbers
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let index = (rng % self.total_documents as u64) as usize;

            if selected.insert(index) {
                samples.push(self.get_document(index)?);
            }
        }

        Ok(samples)
    }
}

/// Iterator for multi-file corpus
pub struct MultiFileIterator<'a> {
    corpus: &'a MultiFileCorpus,
    current: usize,
}

impl<'a> Iterator for MultiFileIterator<'a> {
    type Item = Result<&'a str>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.corpus.num_documents() {
            return None;
        }

        let doc = self.corpus.get_document(self.current);
        self.current += 1;
        Some(doc)
    }
}

/// Cached memory-mapped corpus with LRU caching for frequently accessed documents
pub struct CachedCorpus {
    corpus: MemoryMappedCorpus,
    cache: std::collections::HashMap<usize, String>,
    access_order: std::collections::VecDeque<usize>,
    cache_size: usize,
}

impl CachedCorpus {
    /// Create cached corpus with specified cache size
    pub fn new(_corpus: MemoryMappedCorpus, cachesize: usize) -> Self {
        Self {
            corpus: _corpus,
            cache: std::collections::HashMap::new(),
            access_order: std::collections::VecDeque::new(),
            cache_size: cachesize,
        }
    }

    /// Get document with caching
    pub fn get_document(&mut self, index: usize) -> Result<String> {
        // Check cache first
        if let Some(doc) = self.cache.get(&index) {
            // Move to front of access order
            if let Some(pos) = self.access_order.iter().position(|&x| x == index) {
                self.access_order.remove(pos);
            }
            self.access_order.push_front(index);
            return Ok(doc.clone());
        }

        // Not in cache, get from corpus
        let doc = self.corpus.get_document(index)?.to_string();

        // Add to cache
        if self.cache.len() >= self.cache_size {
            // Remove least recently used
            if let Some(lru_index) = self.access_order.pop_back() {
                self.cache.remove(&lru_index);
            }
        }

        let doc_clone = doc.clone();
        self.cache.insert(index, doc);
        self.access_order.push_front(index);

        Ok(doc_clone)
    }

    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        if self.access_order.is_empty() {
            0.0
        } else {
            self.cache.len() as f64 / self.access_order.len() as f64
        }
    }
}

/// Advanced indexing for fast text search in large corpora
pub struct CorpusIndex {
    word_to_docs: std::collections::HashMap<String, Vec<usize>>,
    #[allow(dead_code)]
    doc_to_words: Vec<std::collections::HashSet<String>>,
}

impl CorpusIndex {
    /// Build index from corpus
    pub fn build<T: Tokenizer>(corpus: &MemoryMappedCorpus, tokenizer: &T) -> Result<Self> {
        let mut word_to_docs = std::collections::HashMap::new();
        let mut doc_to_words = Vec::new();

        for doc_idx in 0..corpus.num_documents() {
            let doc = corpus.get_document(doc_idx)?;
            let tokens = tokenizer.tokenize(doc)?;
            let unique_tokens: std::collections::HashSet<String> = tokens.into_iter().collect();

            for token in &unique_tokens {
                word_to_docs
                    .entry(token.clone())
                    .or_insert_with(Vec::new)
                    .push(doc_idx);
            }

            doc_to_words.push(unique_tokens);
        }

        Ok(Self {
            word_to_docs,
            doc_to_words,
        })
    }

    /// Find documents containing a specific word
    pub fn find_documents_with_word(&self, word: &str) -> Option<&[usize]> {
        self.word_to_docs.get(word).map(|v| v.as_slice())
    }

    /// Find documents containing all words
    pub fn find_documents_with_all_words(&self, words: &[&str]) -> Vec<usize> {
        if words.is_empty() {
            return Vec::new();
        }

        let mut result: Option<std::collections::HashSet<usize>> = None;

        for &word in words {
            if let Some(docs) = self.word_to_docs.get(word) {
                let doc_set: std::collections::HashSet<usize> = docs.iter().copied().collect();

                result = match result {
                    None => Some(doc_set),
                    Some(mut existing) => {
                        existing.retain(|doc| doc_set.contains(doc));
                        Some(existing)
                    }
                };
            } else {
                // Word not found, no documents match
                return Vec::new();
            }
        }

        result
            .map(|set| set.into_iter().collect())
            .unwrap_or_default()
    }

    /// Get vocabulary size
    pub fn vocabulary_size(&self) -> usize {
        self.word_to_docs.len()
    }
}

/// Memory usage monitor for large corpus operations
pub struct MemoryMonitor {
    peak_usage: usize,
    current_usage: usize,
    warnings_enabled: bool,
    warning_threshold: usize,
}

impl MemoryMonitor {
    /// Create new memory monitor
    pub fn new() -> Self {
        Self {
            peak_usage: 0,
            current_usage: 0,
            warnings_enabled: true,
            warning_threshold: 1024 * 1024 * 1024, // 1GB default
        }
    }

    /// Set warning threshold in bytes
    pub fn with_warning_threshold(mut self, threshold: usize) -> Self {
        self.warning_threshold = threshold;
        self
    }

    /// Track memory allocation
    pub fn allocate(&mut self, size: usize) {
        self.current_usage += size;
        self.peak_usage = self.peak_usage.max(self.current_usage);

        if self.warnings_enabled && self.current_usage > self.warning_threshold {
            eprintln!(
                "Memory warning: Current usage {} MB exceeds threshold {} MB",
                self.current_usage / (1024 * 1024),
                self.warning_threshold / (1024 * 1024)
            );
        }
    }

    /// Track memory deallocation
    pub fn deallocate(&mut self, size: usize) {
        self.current_usage = self.current_usage.saturating_sub(size);
    }

    /// Get current memory usage in bytes
    pub fn current_usage(&self) -> usize {
        self.current_usage
    }

    /// Get peak memory usage in bytes
    pub fn peak_usage(&self) -> usize {
        self.peak_usage
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.peak_usage = 0;
        self.current_usage = 0;
    }
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced streaming processor with parallel processing and monitoring
pub struct AdvancedStreamingProcessor<T: Tokenizer> {
    tokenizer: T,
    buffer_size: usize,
    parallel_chunks: usize,
    memory_monitor: MemoryMonitor,
}

impl<T: Tokenizer + Send + Sync> AdvancedStreamingProcessor<T> {
    /// Create new advanced streaming processor
    pub fn new(tokenizer: T) -> Self {
        Self {
            tokenizer,
            buffer_size: 1024 * 1024, // 1MB
            parallel_chunks: num_cpus::get(),
            memory_monitor: MemoryMonitor::new(),
        }
    }

    /// Set parallel processing parameters
    pub fn with_parallelism(mut self, chunks: usize, buffersize: usize) -> Self {
        self.parallel_chunks = chunks;
        self.buffer_size = buffersize;
        self
    }

    /// Process corpus with parallel memory-mapped chunks
    pub fn process_corpus_parallel<F, R>(
        &mut self,
        corpus: &MemoryMappedCorpus,
        processor: F,
    ) -> Result<Vec<R>>
    where
        F: Fn(&str, usize) -> Result<R> + Send + Sync,
        R: Send,
    {
        use scirs2_core::parallel_ops::*;

        let num_docs = corpus.num_documents();
        let chunksize = num_docs.div_ceil(self.parallel_chunks);

        // Track memory usage
        let estimated_memory = num_docs * 100; // Rough estimate
        self.memory_monitor.allocate(estimated_memory);

        let results: Vec<R> = (0..self.parallel_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunksize;
                let end = ((chunk_idx + 1) * chunksize).min(num_docs);

                let mut chunk_results = Vec::new();
                for doc_idx in start..end {
                    let doc = corpus.get_document(doc_idx)?;
                    let result = processor(doc, doc_idx)?;
                    chunk_results.push(result);
                }
                Ok(chunk_results)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();

        self.memory_monitor.deallocate(estimated_memory);
        Ok(results)
    }

    /// Build advanced statistics from corpus
    pub fn build_corpus_statistics(
        &mut self,
        corpus: &MemoryMappedCorpus,
    ) -> Result<CorpusStatistics> {
        let mut stats = CorpusStatistics::new();

        // Clone tokenizer to avoid borrow conflict
        let tokenizer = self.tokenizer.clone_box();

        self.process_corpus_parallel(corpus, move |doc, _doc_idx| {
            let tokens = tokenizer.tokenize(doc)?;
            let char_count = doc.chars().count();
            let word_count = tokens.len();
            let line_count = doc.lines().count();

            Ok(DocumentStats {
                char_count,
                word_count,
                line_count,
                unique_words: tokens
                    .into_iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
            })
        })?
        .into_iter()
        .for_each(|doc_stats| stats.add_document(doc_stats));

        Ok(stats)
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> (usize, usize) {
        (
            self.memory_monitor.current_usage(),
            self.memory_monitor.peak_usage(),
        )
    }
}

/// Statistics for corpus analysis
#[derive(Debug, Clone)]
pub struct CorpusStatistics {
    /// Total number of documents in the corpus
    pub total_documents: usize,
    /// Total number of words across all documents
    pub total_words: usize,
    /// Total number of characters across all documents
    pub total_chars: usize,
    /// Total number of lines across all documents
    pub total_lines: usize,
    /// Size of the vocabulary (unique words)
    pub vocabulary_size: usize,
    /// Average document length in words
    pub avg_doc_length: f64,
    /// Average words per line
    pub avg_words_per_line: f64,
}

impl CorpusStatistics {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self {
            total_documents: 0,
            total_words: 0,
            total_chars: 0,
            total_lines: 0,
            vocabulary_size: 0,
            avg_doc_length: 0.0,
            avg_words_per_line: 0.0,
        }
    }

    /// Add document statistics
    pub fn add_document(&mut self, docstats: DocumentStats) {
        self.total_documents += 1;
        self.total_words += docstats.word_count;
        self.total_chars += docstats.char_count;
        self.total_lines += docstats.line_count;
        self.vocabulary_size += docstats.unique_words;

        // Recalculate averages
        self.avg_doc_length = self.total_words as f64 / self.total_documents as f64;
        self.avg_words_per_line = if self.total_lines > 0 {
            self.total_words as f64 / self.total_lines as f64
        } else {
            0.0
        };
    }
}

impl Default for CorpusStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for individual documents
#[derive(Debug, Clone)]
pub struct DocumentStats {
    /// Number of characters in the document
    pub char_count: usize,
    /// Number of words in the document
    pub word_count: usize,
    /// Number of lines in the document
    pub line_count: usize,
    /// Number of unique words in the document
    pub unique_words: usize,
}

/// Progress tracker for long-running operations
pub struct ProgressTracker {
    total: usize,
    current: usize,
    report_interval: usize,
}

impl ProgressTracker {
    /// Create a new progress tracker
    pub fn new(total: usize) -> Self {
        Self {
            total,
            current: 0,
            report_interval: total / 100, // Report every 1%
        }
    }

    /// Update progress
    pub fn update(&mut self, count: usize) {
        self.current += count;

        if self.current.is_multiple_of(self.report_interval) || self.current >= self.total {
            let percentage = (self.current as f64 / self.total as f64) * 100.0;
            println!(
                "Progress: {:.1}% ({}/{})",
                percentage, self.current, self.total
            );
        }
    }

    /// Check if complete
    pub fn is_complete(&self) -> bool {
        self.current >= self.total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_memory_mapped_corpus() {
        // Create a temporary file with test data
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "First document").unwrap();
        writeln!(file, "Second document").unwrap();
        writeln!(file, "Third document").unwrap();
        file.flush().unwrap();

        let corpus = MemoryMappedCorpus::from_file(file.path()).unwrap();

        assert_eq!(corpus.num_documents(), 3);
        assert_eq!(corpus.get_document(0).unwrap(), "First document");
        assert_eq!(corpus.get_document(1).unwrap(), "Second document");
        assert_eq!(corpus.get_document(2).unwrap(), "Third document");
    }

    #[test]
    fn test_streaming_processor() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "hello world").unwrap();
        writeln!(file, "foo bar baz").unwrap();
        file.flush().unwrap();

        let processor = StreamingTextProcessor::with_default_tokenizer();
        let mut line_count = 0;

        processor
            .process_lines(file.path(), |_line, _line_num| {
                line_count += 1;
                Ok(None::<()>)
            })
            .unwrap();

        assert_eq!(line_count, 2);
    }

    #[test]
    fn test_chunked_reader() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 0..100 {
            writeln!(file, "Line {i}").unwrap();
        }
        file.flush().unwrap();

        let mut reader = ChunkedCorpusReader::new(file.path(), 256).unwrap();
        let mut total_lines = 0;

        while let Some(lines) = reader.next_chunk().unwrap() {
            total_lines += lines.len();
        }

        assert_eq!(total_lines, 100);
    }

    #[test]
    fn test_streaming_vocabulary_building() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "the quick brown fox").unwrap();
        writeln!(file, "the lazy dog").unwrap();
        writeln!(file, "the brown dog").unwrap();
        file.flush().unwrap();

        let processor = StreamingTextProcessor::with_default_tokenizer();
        let vocab = processor
            .build_vocabulary_streaming(file.path(), 2)
            .unwrap();

        // "the" appears 3 times, "brown" and "dog" appear 2 times each
        assert!(vocab.get_index("the").is_some());
        assert!(vocab.get_index("brown").is_some());
        assert!(vocab.get_index("dog").is_some());

        // These appear only once and should be pruned
        assert!(vocab.get_index("quick").is_none());
        assert!(vocab.get_index("fox").is_none());
        assert!(vocab.get_index("lazy").is_none());
    }

    #[test]
    fn test_multi_file_corpus() {
        // Create multiple test files
        let mut file1 = NamedTempFile::new().unwrap();
        writeln!(file1, "Document 1 line 1").unwrap();
        writeln!(file1, "Document 1 line 2").unwrap();
        file1.flush().unwrap();

        let mut file2 = NamedTempFile::new().unwrap();
        writeln!(file2, "Document 2 line 1").unwrap();
        writeln!(file2, "Document 2 line 2").unwrap();
        writeln!(file2, "Document 2 line 3").unwrap();
        file2.flush().unwrap();

        let paths = vec![file1.path(), file2.path()];
        let multi_corpus = MultiFileCorpus::from_files(&paths).unwrap();

        assert_eq!(multi_corpus.num_documents(), 5); // 2 + 3 documents
        assert_eq!(multi_corpus.get_document(0).unwrap(), "Document 1 line 1");
        assert_eq!(multi_corpus.get_document(2).unwrap(), "Document 2 line 1");
        assert_eq!(multi_corpus.get_document(4).unwrap(), "Document 2 line 3");
    }

    #[test]
    fn test_multi_file_random_sampling() {
        let mut file1 = NamedTempFile::new().unwrap();
        for i in 0..10 {
            writeln!(file1, "File1 Doc {i}").unwrap();
        }
        file1.flush().unwrap();

        let mut file2 = NamedTempFile::new().unwrap();
        for i in 0..10 {
            writeln!(file2, "File2 Doc {i}").unwrap();
        }
        file2.flush().unwrap();

        let paths = vec![file1.path(), file2.path()];
        let multi_corpus = MultiFileCorpus::from_files(&paths).unwrap();

        let sample = multi_corpus.random_sample(5, 12345).unwrap();
        assert_eq!(sample.len(), 5);

        // Should be deterministic with same seed
        let sample2 = multi_corpus.random_sample(5, 12345).unwrap();
        assert_eq!(sample, sample2);
    }

    #[test]
    fn test_cached_corpus() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 0..10 {
            writeln!(file, "Document {i}").unwrap();
        }
        file.flush().unwrap();

        let base_corpus = MemoryMappedCorpus::from_file(file.path()).unwrap();
        let mut cached_corpus = CachedCorpus::new(base_corpus, 3);

        // Access documents
        let doc0 = cached_corpus.get_document(0).unwrap();
        let doc1 = cached_corpus.get_document(1).unwrap();
        let doc2 = cached_corpus.get_document(2).unwrap();

        assert_eq!(doc0, "Document 0");
        assert_eq!(doc1, "Document 1");
        assert_eq!(doc2, "Document 2");

        // Access doc0 again - should be cached
        let doc0_again = cached_corpus.get_document(0).unwrap();
        assert_eq!(doc0_again, "Document 0");

        // Cache should have good hit rate for repeated access
        let hit_rate = cached_corpus.cache_hit_rate();
        assert!(hit_rate > 0.0);
    }

    #[test]
    fn test_corpus_index() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "the quick brown fox").unwrap();
        writeln!(file, "the lazy dog").unwrap();
        writeln!(file, "quick brown animal").unwrap();
        file.flush().unwrap();

        let corpus = MemoryMappedCorpus::from_file(file.path()).unwrap();
        let tokenizer = WordTokenizer::default();
        let index = CorpusIndex::build(&corpus, &tokenizer).unwrap();

        // Test single word search
        let docs_with_quick = index.find_documents_with_word("quick").unwrap();
        assert_eq!(docs_with_quick.len(), 2); // Documents 0 and 2

        // Test multi-word search
        let docs_with_all = index.find_documents_with_all_words(&["the", "brown"]);
        assert_eq!(docs_with_all.len(), 1); // Only document 0

        // Test vocabulary size
        assert!(index.vocabulary_size() > 0);
    }

    #[test]
    fn test_memory_monitor() {
        let mut monitor = MemoryMonitor::new().with_warning_threshold(1000);

        assert_eq!(monitor.current_usage(), 0);
        assert_eq!(monitor.peak_usage(), 0);

        monitor.allocate(500);
        assert_eq!(monitor.current_usage(), 500);
        assert_eq!(monitor.peak_usage(), 500);

        monitor.allocate(300);
        assert_eq!(monitor.current_usage(), 800);
        assert_eq!(monitor.peak_usage(), 800);

        monitor.deallocate(200);
        assert_eq!(monitor.current_usage(), 600);
        assert_eq!(monitor.peak_usage(), 800); // Peak should remain

        monitor.reset();
        assert_eq!(monitor.current_usage(), 0);
        assert_eq!(monitor.peak_usage(), 0);
    }

    #[test]
    fn test_advanced_streaming_processor() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "hello world").unwrap();
        writeln!(file, "foo bar baz").unwrap();
        writeln!(file, "test document").unwrap();
        file.flush().unwrap();

        let corpus = MemoryMappedCorpus::from_file(file.path()).unwrap();
        let tokenizer = WordTokenizer::default();
        let mut processor = AdvancedStreamingProcessor::new(tokenizer);

        let results = processor
            .process_corpus_parallel(&corpus, |doc, idx| {
                let doc_len = doc.len();
                Ok(format!("Processed doc {idx}: {doc_len}"))
            })
            .unwrap();

        assert_eq!(results.len(), 3);
        assert!(results[0].contains("Processed doc 0"));
        assert!(results[1].contains("Processed doc 1"));
        assert!(results[2].contains("Processed doc 2"));

        // Test memory stats
        let (current, _peak) = processor.memory_stats();
        assert_eq!(current, 0); // Should be deallocated after processing
    }

    #[test]
    fn test_corpus_statistics() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "hello world test").unwrap();
        writeln!(file, "foo bar").unwrap();
        writeln!(file, "single").unwrap();
        file.flush().unwrap();

        let corpus = MemoryMappedCorpus::from_file(file.path()).unwrap();
        let tokenizer = WordTokenizer::default();
        let mut processor = AdvancedStreamingProcessor::new(tokenizer);

        let stats = processor.build_corpus_statistics(&corpus).unwrap();

        assert_eq!(stats.total_documents, 3);
        assert_eq!(stats.total_words, 6); // 3 + 2 + 1
        assert!(stats.avg_doc_length > 0.0);
        assert_eq!(stats.total_lines, 3);
    }

    #[test]
    fn test_document_stats() {
        let mut stats = CorpusStatistics::new();

        let doc_stats1 = DocumentStats {
            char_count: 100,
            word_count: 20,
            line_count: 5,
            unique_words: 15,
        };

        let doc_stats2 = DocumentStats {
            char_count: 50,
            word_count: 10,
            line_count: 2,
            unique_words: 8,
        };

        stats.add_document(doc_stats1);
        stats.add_document(doc_stats2);

        assert_eq!(stats.total_documents, 2);
        assert_eq!(stats.total_words, 30);
        assert_eq!(stats.total_chars, 150);
        assert_eq!(stats.total_lines, 7);
        assert_eq!(stats.avg_doc_length, 15.0); // 30 words / 2 docs
    }

    #[test]
    fn test_corpus_index_edge_cases() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file).unwrap(); // Empty document
        writeln!(file, "single").unwrap();
        file.flush().unwrap();

        let corpus = MemoryMappedCorpus::from_file(file.path()).unwrap();
        let tokenizer = WordTokenizer::default();
        let index = CorpusIndex::build(&corpus, &tokenizer).unwrap();

        // Search for non-existent word
        assert!(index.find_documents_with_word("nonexistent").is_none());

        // Search with empty word list
        let empty_result = index.find_documents_with_all_words(&[]);
        assert!(empty_result.is_empty());

        // Search for word that doesn't exist
        let no_match = index.find_documents_with_all_words(&["nonexistent"]);
        assert!(no_match.is_empty());
    }

    #[test]
    fn test_multi_file_iterator() {
        let mut file1 = NamedTempFile::new().unwrap();
        writeln!(file1, "doc1").unwrap();
        writeln!(file1, "doc2").unwrap();
        file1.flush().unwrap();

        let mut file2 = NamedTempFile::new().unwrap();
        writeln!(file2, "doc3").unwrap();
        file2.flush().unwrap();

        let paths = vec![file1.path(), file2.path()];
        let multi_corpus = MultiFileCorpus::from_files(&paths).unwrap();

        let docs: Result<Vec<_>> = multi_corpus.iter().collect();
        let docs = docs.unwrap();

        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0], "doc1");
        assert_eq!(docs[1], "doc2");
        assert_eq!(docs[2], "doc3");
    }
}
