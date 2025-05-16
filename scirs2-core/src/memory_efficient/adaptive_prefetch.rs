//! Adaptive prefetching strategy that dynamically learns from access patterns.
//!
//! This module provides an enhanced prefetching system that uses machine learning
//! techniques to dynamically adjust its prefetching strategy based on observed
//! access patterns and performance metrics.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use super::prefetch::{AccessPattern, AccessPatternTracker, PrefetchConfig, PrefetchStats};
use crate::error::{CoreError, CoreResult, ErrorContext};

/// Maximum number of strategies to try during exploration phase
const MAX_EXPLORATION_STRATEGIES: usize = 5;

/// Duration for testing each strategy during exploration
const STRATEGY_TEST_DURATION: Duration = Duration::from_secs(60);

/// Reinforcement learning parameters
const LEARNING_RATE: f64 = 0.1;
const DISCOUNT_FACTOR: f64 = 0.9;
const EXPLORATION_RATE_INITIAL: f64 = 0.3;
const EXPLORATION_RATE_DECAY: f64 = 0.995;

/// Types of prefetching strategies that can be dynamically selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrefetchStrategy {
    /// Prefetch next N consecutive blocks
    Sequential(usize),

    /// Prefetch blocks with a fixed stride
    Strided { stride: usize, count: usize },

    /// Prefetch blocks based on a custom pattern
    Pattern {
        window_size: usize,
        lookahead: usize,
    },

    /// Hybrid approach combining sequential and pattern-based
    Hybrid { sequential: usize, pattern: usize },

    /// Conservative prefetching (minimal prefetching)
    Conservative,

    /// Aggressive prefetching (prefetch many blocks)
    Aggressive,

    /// No prefetching (baseline for comparisons)
    None,
}

impl Default for PrefetchStrategy {
    fn default() -> Self {
        PrefetchStrategy::Sequential(2)
    }
}

/// Performance metrics for a particular prefetching strategy.
#[derive(Debug, Clone)]
struct StrategyPerformance {
    /// The strategy being evaluated
    strategy: PrefetchStrategy,

    /// Number of times this strategy has been used
    usage_count: usize,

    /// Cache hit rate when using this strategy
    hit_rate: f64,

    /// Average latency for block access with this strategy
    avg_latency_ns: f64,

    /// Time when this strategy was last used
    last_used: Instant,

    /// Q-value for reinforcement learning
    q_value: f64,
}

/// Advanced access pattern detector with dynamic learning.
#[derive(Debug)]
pub struct AdaptivePatternTracker {
    /// Base configuration
    config: PrefetchConfig,

    /// History of accessed blocks
    history: VecDeque<(usize, Instant, Duration)>, // (block_idx, timestamp, access_time)

    /// Current detected pattern
    current_pattern: AccessPattern,

    /// For strided patterns, the stride value
    stride: Option<usize>,

    /// Performance of different strategies
    strategy_performance: HashMap<PrefetchStrategy, StrategyPerformance>,

    /// Current active strategy
    current_strategy: PrefetchStrategy,

    /// Time when we should try another strategy
    next_strategy_change: Instant,

    /// Whether we're in exploration or exploitation phase
    exploring: bool,

    /// Current exploration rate (epsilon) for epsilon-greedy strategy
    exploration_rate: f64,

    /// Matrix dimension, if known
    dimensions: Option<Vec<usize>>,

    /// Patterns with dimension-aware context
    dimensional_patterns: HashMap<String, Vec<usize>>,
}

impl AdaptivePatternTracker {
    /// Create a new adaptive pattern tracker.
    pub fn new(config: PrefetchConfig) -> Self {
        let mut strategies = HashMap::new();

        // Initialize default strategies with neutral Q-values
        for strategy in [
            PrefetchStrategy::Sequential(2),
            PrefetchStrategy::Sequential(5),
            PrefetchStrategy::Strided {
                stride: 10,
                count: 3,
            },
            PrefetchStrategy::Conservative,
            PrefetchStrategy::Aggressive,
            PrefetchStrategy::None,
        ] {
            strategies.insert(
                strategy,
                StrategyPerformance {
                    strategy,
                    usage_count: 0,
                    hit_rate: 0.0,
                    avg_latency_ns: 0.0,
                    last_used: Instant::now(),
                    q_value: 0.0,
                },
            );
        }

        Self {
            config,
            history: VecDeque::with_capacity(config.history_size),
            current_pattern: AccessPattern::Random,
            stride: None,
            strategy_performance: strategies,
            current_strategy: PrefetchStrategy::default(),
            next_strategy_change: Instant::now() + STRATEGY_TEST_DURATION,
            exploring: true,
            exploration_rate: EXPLORATION_RATE_INITIAL,
            dimensions: None,
            dimensional_patterns: HashMap::new(),
        }
    }

    /// Set the array dimensions for better pattern detection.
    pub fn set_dimensions(&mut self, dimensions: Vec<usize>) {
        self.dimensions = Some(dimensions);
    }

    /// Update the performance metrics for the current strategy.
    pub fn update_performance(&mut self, stats: &PrefetchStats, avg_latency_ns: f64) {
        if let Some(perf) = self.strategy_performance.get_mut(&self.current_strategy) {
            // Update the performance metrics
            perf.usage_count += 1;
            perf.hit_rate = stats.hit_rate;
            perf.avg_latency_ns = avg_latency_ns;
            perf.last_used = Instant::now();

            // Calculate reward (higher hit rate and lower latency are better)
            let hit_rate_reward = stats.hit_rate;
            let latency_factor = if perf.avg_latency_ns > 0.0 {
                1.0 / (1.0 + perf.avg_latency_ns / 1_000_000.0) // Convert to ms and normalize
            } else {
                0.0
            };

            let reward = hit_rate_reward * 0.7 + latency_factor * 0.3;

            // Update Q-value with simple Q-learning
            perf.q_value = (1.0 - LEARNING_RATE) * perf.q_value + LEARNING_RATE * reward;
        }

        // Check if it's time to select a new strategy
        if Instant::now() >= self.next_strategy_change {
            self.select_next_strategy();
        }
    }

    /// Select the next strategy to use.
    fn select_next_strategy(&mut self) {
        // Decay exploration rate
        self.exploration_rate *= EXPLORATION_RATE_DECAY;

        // Decide whether to explore or exploit
        if self.exploring || rand::random::<f64>() < self.exploration_rate {
            // Exploration phase: try different strategies
            let available_strategies: Vec<PrefetchStrategy> =
                self.strategy_performance.keys().copied().collect();

            // Select a random strategy, but avoid the current one
            let candidates: Vec<PrefetchStrategy> = available_strategies
                .into_iter()
                .filter(|&s| s != self.current_strategy)
                .collect();

            if !candidates.is_empty() {
                let idx = rand::random::<usize>() % candidates.len();
                self.current_strategy = candidates[idx];
            }

            // Check if we should move to exploitation phase
            let total_usage: usize = self
                .strategy_performance
                .values()
                .map(|p| p.usage_count)
                .sum();

            if total_usage >= MAX_EXPLORATION_STRATEGIES * 2 {
                self.exploring = false;
            }
        } else {
            // Exploitation phase: choose the strategy with the highest Q-value
            let best_strategy = self
                .strategy_performance
                .values()
                .max_by(|a, b| a.q_value.partial_cmp(&b.q_value).unwrap())
                .map(|p| p.strategy)
                .unwrap_or_default();

            self.current_strategy = best_strategy;
        }

        // Set the next time to change strategies
        self.next_strategy_change = Instant::now() + STRATEGY_TEST_DURATION;

        // Update the strategy if it's based on detected pattern
        self.update_strategy_from_pattern();
    }

    /// Update strategy based on the current detected pattern.
    fn update_strategy_from_pattern(&mut self) {
        match self.current_pattern {
            AccessPattern::Sequential => {
                // If pattern is sequential but we're not using a sequential strategy,
                // consider switching to a sequential strategy
                match self.current_strategy {
                    PrefetchStrategy::Sequential(_) => {
                        // Already using sequential strategy, nothing to do
                    }
                    _ => {
                        // Consider switching to sequential, but respect the Q-values
                        let seq_strategy = PrefetchStrategy::Sequential(self.config.prefetch_count);

                        if let Some(seq_perf) = self.strategy_performance.get(&seq_strategy) {
                            let current_q = self
                                .strategy_performance
                                .get(&self.current_strategy)
                                .map(|p| p.q_value)
                                .unwrap_or(0.0);

                            if seq_perf.q_value > current_q * 1.2 {
                                // Sequential is significantly better, switch to it
                                self.current_strategy = seq_strategy;
                            }
                        } else {
                            // We don't have data on sequential yet, add it and possibly switch
                            self.strategy_performance.insert(
                                seq_strategy,
                                StrategyPerformance {
                                    strategy: seq_strategy,
                                    usage_count: 0,
                                    hit_rate: 0.0,
                                    avg_latency_ns: 0.0,
                                    last_used: Instant::now(),
                                    q_value: 0.2, // Slight bias towards sequential when detected
                                },
                            );

                            // Occasionally switch to it for exploration
                            if rand::random::<f64>() < 0.5 {
                                self.current_strategy = seq_strategy;
                            }
                        }
                    }
                }
            }
            AccessPattern::Strided(stride) => {
                // If pattern is strided but we're not using a strided strategy,
                // consider switching to a strided strategy
                let strided_strategy = PrefetchStrategy::Strided {
                    stride,
                    count: self.config.prefetch_count,
                };

                // Add or update this strategy in our performance map
                if !self.strategy_performance.contains_key(&strided_strategy) {
                    self.strategy_performance.insert(
                        strided_strategy,
                        StrategyPerformance {
                            strategy: strided_strategy,
                            usage_count: 0,
                            hit_rate: 0.0,
                            avg_latency_ns: 0.0,
                            last_used: Instant::now(),
                            q_value: 0.2, // Slight bias when detected
                        },
                    );
                }

                // Consider switching to this strided strategy
                match self.current_strategy {
                    PrefetchStrategy::Strided {
                        stride: current_stride,
                        ..
                    } => {
                        // Already using strided strategy, maybe update the stride
                        if current_stride != stride && rand::random::<f64>() < 0.7 {
                            self.current_strategy = strided_strategy;
                        }
                    }
                    _ => {
                        // Not using strided strategy, consider switching
                        let current_q = self
                            .strategy_performance
                            .get(&self.current_strategy)
                            .map(|p| p.q_value)
                            .unwrap_or(0.0);

                        if let Some(strided_perf) = self.strategy_performance.get(&strided_strategy)
                        {
                            if strided_perf.q_value > current_q * 1.1 || rand::random::<f64>() < 0.3
                            {
                                self.current_strategy = strided_strategy;
                            }
                        } else {
                            // Occasionally switch to it for exploration
                            if rand::random::<f64>() < 0.4 {
                                self.current_strategy = strided_strategy;
                            }
                        }
                    }
                }
            }
            AccessPattern::Custom => {
                // If we have dimensional information, try to detect specific patterns
                if let Some(dims) = &self.dimensions {
                    // Create pattern-specific strategies
                    let detected_patterns = self.detect_dimensional_patterns(dims);

                    for pattern_name in detected_patterns {
                        // For matrix traversal, use hybrid strategy
                        if pattern_name == "matrix_traversal_row_major" {
                            let strategy = PrefetchStrategy::Hybrid {
                                sequential: dims[1], // Row length
                                pattern: 2,
                            };

                            // Add this strategy if it doesn't exist
                            if !self.strategy_performance.contains_key(&strategy) {
                                self.strategy_performance.insert(
                                    strategy,
                                    StrategyPerformance {
                                        strategy,
                                        usage_count: 0,
                                        hit_rate: 0.0,
                                        avg_latency_ns: 0.0,
                                        last_used: Instant::now(),
                                        q_value: 0.3, // Higher bias for dimensional patterns
                                    },
                                );
                            }

                            // Consider switching to this strategy
                            if rand::random::<f64>() < 0.6 {
                                self.current_strategy = strategy;
                            }
                        } else if pattern_name == "matrix_traversal_col_major" {
                            let strategy = PrefetchStrategy::Strided {
                                stride: dims[0], // Column stride
                                count: 3,
                            };

                            // Add this strategy if it doesn't exist
                            if !self.strategy_performance.contains_key(&strategy) {
                                self.strategy_performance.insert(
                                    strategy,
                                    StrategyPerformance {
                                        strategy,
                                        usage_count: 0,
                                        hit_rate: 0.0,
                                        avg_latency_ns: 0.0,
                                        last_used: Instant::now(),
                                        q_value: 0.3,
                                    },
                                );
                            }

                            // Consider switching to this strategy
                            if rand::random::<f64>() < 0.6 {
                                self.current_strategy = strategy;
                            }
                        }
                    }
                } else {
                    // Without dimensional information, use pattern-based strategy
                    let strategy = PrefetchStrategy::Pattern {
                        window_size: self.config.min_pattern_length,
                        lookahead: self.config.prefetch_count,
                    };

                    // Add this strategy if it doesn't exist
                    if !self.strategy_performance.contains_key(&strategy) {
                        self.strategy_performance.insert(
                            strategy,
                            StrategyPerformance {
                                strategy,
                                usage_count: 0,
                                hit_rate: 0.0,
                                avg_latency_ns: 0.0,
                                last_used: Instant::now(),
                                q_value: 0.2,
                            },
                        );
                    }

                    // Occasionally switch to pattern-based strategy
                    if rand::random::<f64>() < 0.4 {
                        self.current_strategy = strategy;
                    }
                }
            }
            AccessPattern::Random => {
                // For random access, favor conservative or aggressive based on past performance
                let conservative_q = self
                    .strategy_performance
                    .get(&PrefetchStrategy::Conservative)
                    .map(|p| p.q_value)
                    .unwrap_or(0.1);

                let aggressive_q = self
                    .strategy_performance
                    .get(&PrefetchStrategy::Aggressive)
                    .map(|p| p.q_value)
                    .unwrap_or(0.1);

                if conservative_q > aggressive_q * 1.2 {
                    self.current_strategy = PrefetchStrategy::Conservative;
                } else if aggressive_q > conservative_q * 1.2 {
                    self.current_strategy = PrefetchStrategy::Aggressive;
                } else {
                    // They're similar, choose randomly
                    self.current_strategy = if rand::random::<f64>() < 0.5 {
                        PrefetchStrategy::Conservative
                    } else {
                        PrefetchStrategy::Aggressive
                    };
                }
            }
        }
    }

    /// Detect dimensional patterns in the access history.
    fn detect_dimensional_patterns(&mut self, dimensions: &[usize]) -> Vec<String> {
        if dimensions.len() < 2 || self.history.len() < 10 {
            return Vec::new();
        }

        let mut detected_patterns = Vec::new();

        // Get the flat indices from history
        let flat_indices: Vec<usize> = self.history.iter().map(|(idx, _, _)| *idx).collect();

        // Check for row-major traversal (adjacent elements in a row)
        let mut row_major_matches = 0;
        for i in 1..flat_indices.len() {
            if flat_indices[i] == flat_indices[i - 1] + 1 {
                row_major_matches += 1;
            }
        }

        // Check for column-major traversal (adjacent elements in a column)
        let mut col_major_matches = 0;
        let col_stride = dimensions[0]; // For 2D array, stride between columns
        for i in 1..flat_indices.len() {
            if flat_indices[i] == flat_indices[i - 1] + col_stride {
                col_major_matches += 1;
            }
        }

        // Calculate match percentages
        let total_pairs = flat_indices.len() - 1;
        let row_major_pct = row_major_matches as f64 / total_pairs as f64;
        let col_major_pct = col_major_matches as f64 / total_pairs as f64;

        // Detect patterns if they match a significant portion of the history
        if row_major_pct > 0.6 {
            detected_patterns.push("matrix_traversal_row_major".to_string());
        }

        if col_major_pct > 0.6 {
            detected_patterns.push("matrix_traversal_col_major".to_string());
        }

        // Try to detect zigzag pattern (alternating row directions)
        if self.detect_zigzag_pattern(&flat_indices, dimensions) {
            detected_patterns.push("zigzag_scan".to_string());
        }

        // Keep track of dimensional patterns
        for pattern in &detected_patterns {
            self.dimensional_patterns
                .entry(pattern.clone())
                .or_insert_with(Vec::new)
                .push(flat_indices.len());
        }

        detected_patterns
    }

    /// Detect zigzag pattern (alternating row directions).
    fn detect_zigzag_pattern(&self, indices: &[usize], dimensions: &[usize]) -> bool {
        if indices.len() < 10 || dimensions.len() < 2 {
            return false;
        }

        let row_size = dimensions[1];

        // Try to detect changes in direction at row boundaries
        let mut direction_changes = 0;
        let mut current_direction = if indices.len() >= 2 {
            if indices[1] > indices[0] {
                1
            } else {
                -1
            }
        } else {
            return false;
        };

        for i in 1..indices.len() - 1 {
            // Check if we're at a potential row boundary
            if (indices[i] % row_size == 0) || (indices[i] % row_size == row_size - 1) {
                let next_direction = if indices[i + 1] > indices[i] { 1 } else { -1 };

                if next_direction != current_direction {
                    direction_changes += 1;
                    current_direction = next_direction;
                }
            }
        }

        // Check if there are enough direction changes to indicate a zigzag pattern
        let expected_changes = indices.len() / row_size;
        direction_changes >= expected_changes / 2
    }

    /// Detect the access pattern based on the history.
    fn detect_pattern(&mut self) {
        if self.history.len() < self.config.min_pattern_length {
            // Not enough history to detect a pattern
            self.current_pattern = AccessPattern::Random;
            return;
        }

        // Extract just the block indices from history
        let indices: Vec<usize> = self.history.iter().map(|(idx, _, _)| *idx).collect();

        // Check for sequential access
        let mut is_sequential = true;
        for i in 1..indices.len() {
            if indices[i] != indices[i - 1] + 1 {
                is_sequential = false;
                break;
            }
        }

        if is_sequential {
            self.current_pattern = AccessPattern::Sequential;
            return;
        }

        // Check for strided access
        if indices.len() >= 3 {
            let mut possible_strides = Vec::new();

            // Calculate potential strides
            for window_size in 2..=std::cmp::min(indices.len() / 2, 10) {
                let mut stride_counts = HashMap::new();

                for i in window_size..indices.len() {
                    let stride = match indices[i].checked_sub(indices[i - window_size]) {
                        Some(s) => s / window_size,
                        None => continue,
                    };

                    *stride_counts.entry(stride).or_insert(0) += 1;
                }

                // Find the most common stride
                if let Some((stride, count)) =
                    stride_counts.into_iter().max_by_key(|(_, count)| *count)
                {
                    // Check if this stride appears enough times to be significant
                    let threshold = (indices.len() - window_size) / 2;
                    if count >= threshold {
                        possible_strides.push((stride, count, window_size));
                    }
                }
            }

            // Choose the stride with the highest count
            if let Some((stride, _, _)) = possible_strides
                .into_iter()
                .max_by_key(|(_, count, _)| *count)
            {
                if stride > 0 {
                    self.current_pattern = AccessPattern::Strided(stride);
                    self.stride = Some(stride);
                    return;
                }
            }
        }

        // Check for custom dimensional patterns
        if let Some(dims) = &self.dimensions {
            if !self.detect_dimensional_patterns(dims).is_empty() {
                self.current_pattern = AccessPattern::Custom;
                return;
            }
        }

        // No regular pattern detected
        self.current_pattern = AccessPattern::Random;
    }

    /// Get the blocks to prefetch based on the current strategy.
    pub fn get_blocks_to_prefetch(&self, count: usize) -> Vec<usize> {
        if self.history.is_empty() {
            return Vec::new();
        }

        let latest = self.history.back().unwrap().0;

        match self.current_strategy {
            PrefetchStrategy::Sequential(n) => {
                // Prefetch the next n blocks sequentially
                let prefetch_count = std::cmp::min(n, count);
                (1..=prefetch_count).map(|i| latest + i).collect()
            }
            PrefetchStrategy::Strided { stride, count: n } => {
                // Prefetch n blocks with the given stride
                let prefetch_count = std::cmp::min(n, count);
                (1..=prefetch_count).map(|i| latest + stride * i).collect()
            }
            PrefetchStrategy::Pattern {
                window_size,
                lookahead,
            } => {
                // Use pattern matching to predict future blocks
                self.predict_from_pattern(latest, std::cmp::min(lookahead, count))
            }
            PrefetchStrategy::Hybrid {
                sequential,
                pattern,
            } => {
                // Combine sequential and pattern-based prefetching
                let mut blocks = Vec::new();

                // First add sequential blocks
                for i in 1..=sequential {
                    blocks.push(latest + i);
                }

                // Then add pattern-based predictions
                blocks.extend(self.predict_from_pattern(
                    latest,
                    std::cmp::min(pattern, count.saturating_sub(sequential)),
                ));

                // Return unique blocks
                blocks
                    .into_iter()
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect()
            }
            PrefetchStrategy::Conservative => {
                // Prefetch conservatively (just 1-2 blocks)
                vec![latest + 1]
            }
            PrefetchStrategy::Aggressive => {
                // Prefetch aggressively
                let mut blocks = Vec::with_capacity(count);

                // First try sequential blocks
                for i in 1..=count / 2 {
                    blocks.push(latest + i);
                }

                // Then add some nearby blocks
                if let Some(stride) = self.stride {
                    blocks.push(latest + stride);
                    if stride > 1 && blocks.len() < count {
                        blocks.push(latest + stride * 2);
                    }
                }

                // For the remaining slots, add some pattern-based predictions
                let remaining = count.saturating_sub(blocks.len());
                if remaining > 0 {
                    blocks.extend(self.predict_from_pattern(latest, remaining));
                }

                // Return unique blocks
                blocks
                    .into_iter()
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect()
            }
            PrefetchStrategy::None => {
                // Don't prefetch anything
                Vec::new()
            }
        }
    }

    /// Predict blocks based on pattern matching in history.
    fn predict_from_pattern(&self, latest: usize, count: usize) -> Vec<usize> {
        // Get the last few block indices from history
        let history_window = std::cmp::min(8, self.history.len() - 1);
        let mut pattern = Vec::with_capacity(history_window);

        for i in 1..=history_window {
            if let Some((idx, _, _)) = self.history.get(self.history.len() - 1 - i) {
                pattern.push(*idx);
            }
        }

        if pattern.is_empty() {
            return vec![latest + 1]; // Default to next block if no pattern
        }

        // Look for this pattern elsewhere in history
        let mut predictions = Vec::new();
        let mut occurrences = Vec::new();

        for i in 0..self.history.len() - pattern.len() {
            let mut matches = true;
            for j in 0..pattern.len() {
                if let Some((idx, _, _)) = self.history.get(i + j) {
                    if *idx != pattern[j] {
                        matches = false;
                        break;
                    }
                } else {
                    matches = false;
                    break;
                }
            }

            if matches {
                occurrences.push(i);
            }
        }

        // For each occurrence, check what comes next
        for &i in &occurrences {
            if i + pattern.len() < self.history.len() {
                if let Some((next_idx, _, _)) = self.history.get(i + pattern.len()) {
                    let prediction = latest + (*next_idx - pattern[0]);
                    predictions.push(prediction);
                }
            }
        }

        // If no predictions from pattern matching, fall back to recent strides
        if predictions.is_empty() && pattern.len() >= 2 {
            let most_recent_stride = latest - pattern[0];
            predictions.push(latest + most_recent_stride);
        }

        // Return unique predictions, limited to count
        predictions
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .take(count)
            .collect()
    }
}

impl AccessPatternTracker for AdaptivePatternTracker {
    fn record_access(&mut self, block_idx: usize) {
        // Record the time since the last access (latency)
        let now = Instant::now();
        let access_time = if let Some((_, last_time, _)) = self.history.back() {
            now.duration_since(*last_time)
        } else {
            Duration::from_nanos(0)
        };

        // Add to history and remove oldest if needed
        self.history.push_back((block_idx, now, access_time));

        if self.history.len() > self.config.history_size {
            self.history.pop_front();
        }

        // Update pattern if we have enough history
        if self.history.len() >= self.config.min_pattern_length {
            self.detect_pattern();
        }
    }

    fn predict_next_blocks(&self, count: usize) -> Vec<usize> {
        self.get_blocks_to_prefetch(count)
    }

    fn current_pattern(&self) -> AccessPattern {
        self.current_pattern
    }

    fn clear_history(&mut self) {
        self.history.clear();
        self.current_pattern = AccessPattern::Random;
        self.stride = None;
    }
}

/// Factory for creating different types of access pattern trackers.
pub struct PatternTrackerFactory;

impl PatternTrackerFactory {
    /// Create a new access pattern tracker of the specified type.
    pub fn create_tracker(
        config: PrefetchConfig,
        tracker_type: &str,
    ) -> Box<dyn AccessPatternTracker + Send + Sync> {
        match tracker_type {
            "adaptive" => Box::new(AdaptivePatternTracker::new(config)),
            _ => Box::new(super::prefetch::BlockAccessTracker::new(config)),
        }
    }
}

/// Extended prefetching configuration with adaptive learning options.
#[derive(Debug, Clone)]
pub struct AdaptivePrefetchConfig {
    /// Base prefetching configuration
    pub base: PrefetchConfig,

    /// Whether to use the adaptive tracker
    pub use_adaptive_tracker: bool,

    /// Whether to enable reinforcement learning
    pub enable_learning: bool,

    /// Dimensions of the array (if known)
    pub dimensions: Option<Vec<usize>>,

    /// Learning rate for Q-value updates
    pub learning_rate: f64,

    /// How often to evaluate strategies (in seconds)
    pub evaluation_interval: Duration,
}

impl Default for AdaptivePrefetchConfig {
    fn default() -> Self {
        Self {
            base: PrefetchConfig::default(),
            use_adaptive_tracker: true,
            enable_learning: true,
            dimensions: None,
            learning_rate: LEARNING_RATE,
            evaluation_interval: STRATEGY_TEST_DURATION,
        }
    }
}

/// Builder for adaptive prefetch configuration.
#[derive(Debug, Clone)]
pub struct AdaptivePrefetchConfigBuilder {
    config: AdaptivePrefetchConfig,
}

impl AdaptivePrefetchConfigBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            config: AdaptivePrefetchConfig::default(),
        }
    }

    /// Enable or disable prefetching.
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.base.enabled = enabled;
        self
    }

    /// Set the number of blocks to prefetch ahead of the current access.
    pub fn prefetch_count(mut self, count: usize) -> Self {
        self.config.base.prefetch_count = count;
        self
    }

    /// Set the maximum number of blocks to keep in the prefetch history.
    pub fn history_size(mut self, size: usize) -> Self {
        self.config.base.history_size = size;
        self
    }

    /// Set the minimum number of accesses needed to detect a pattern.
    pub fn min_pattern_length(mut self, length: usize) -> Self {
        self.config.base.min_pattern_length = length;
        self
    }

    /// Enable or disable asynchronous prefetching.
    pub fn async_prefetch(mut self, async_prefetch: bool) -> Self {
        self.config.base.async_prefetch = async_prefetch;
        self
    }

    /// Set the timeout for prefetch operations.
    pub fn prefetch_timeout(mut self, timeout: Duration) -> Self {
        self.config.base.prefetch_timeout = timeout;
        self
    }

    /// Set whether to use the adaptive tracker.
    pub fn use_adaptive_tracker(mut self, use_adaptive: bool) -> Self {
        self.config.use_adaptive_tracker = use_adaptive;
        self
    }

    /// Enable or disable reinforcement learning.
    pub fn enable_learning(mut self, enable: bool) -> Self {
        self.config.enable_learning = enable;
        self
    }

    /// Set the dimensions of the array.
    pub fn dimensions(mut self, dimensions: Vec<usize>) -> Self {
        self.config.dimensions = Some(dimensions);
        self
    }

    /// Set the learning rate for Q-value updates.
    pub fn learning_rate(mut self, rate: f64) -> Self {
        self.config.learning_rate = rate;
        self
    }

    /// Set how often to evaluate strategies.
    pub fn evaluation_interval(mut self, interval: Duration) -> Self {
        self.config.evaluation_interval = interval;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> AdaptivePrefetchConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_pattern_detection_sequential() {
        let config = PrefetchConfig {
            min_pattern_length: 4,
            ..Default::default()
        };

        let mut tracker = AdaptivePatternTracker::new(config);

        // Record sequential access
        for i in 0..10 {
            tracker.record_access(i);
        }

        // Check that the pattern was detected correctly
        assert_eq!(tracker.current_pattern(), AccessPattern::Sequential);

        // Check predictions
        let predictions = tracker.predict_next_blocks(3);
        assert!(!predictions.is_empty());

        // Should include at least the next sequential block
        assert!(predictions.contains(&10));
    }

    #[test]
    fn test_adaptive_pattern_detection_strided() {
        let config = PrefetchConfig {
            min_pattern_length: 4,
            ..Default::default()
        };

        let mut tracker = AdaptivePatternTracker::new(config);

        // Record strided access with stride 3
        for i in (0..30).step_by(3) {
            tracker.record_access(i);
        }

        // Check that the pattern was detected correctly
        assert_eq!(tracker.current_pattern(), AccessPattern::Strided(3));

        // Check predictions
        let predictions = tracker.predict_next_blocks(3);
        assert!(!predictions.is_empty());

        // Should include at least the next strided block
        assert!(predictions.contains(&30));
    }

    #[test]
    fn test_adaptive_strategy_selection() {
        let config = PrefetchConfig {
            min_pattern_length: 4,
            ..Default::default()
        };

        let mut tracker = AdaptivePatternTracker::new(config);

        // Record a mix of access patterns
        for i in 0..5 {
            tracker.record_access(i);
        }

        for i in (10..30).step_by(5) {
            tracker.record_access(i);
        }

        // Update performance metrics
        let stats = PrefetchStats {
            prefetch_count: 10,
            prefetch_hits: 8,
            prefetch_misses: 2,
            hit_rate: 0.8,
        };

        tracker.update_performance(&stats, 100_000.0);

        // Check that strategy selection works
        let strategy = tracker.current_strategy;
        assert!(matches!(
            strategy,
            PrefetchStrategy::Sequential(_)
                | PrefetchStrategy::Strided { .. }
                | PrefetchStrategy::Conservative
                | PrefetchStrategy::Aggressive
        ));

        // Check predictions
        let predictions = tracker.predict_next_blocks(3);
        assert!(!predictions.is_empty());
    }

    #[test]
    fn test_dimensional_pattern_detection() {
        let config = PrefetchConfig {
            min_pattern_length: 4,
            history_size: 50,
            ..Default::default()
        };

        let mut tracker = AdaptivePatternTracker::new(config);

        // Set dimensions to a 5x5 matrix
        tracker.set_dimensions(vec![5, 5]);

        // Record row-major traversal
        for i in 0..5 {
            for j in 0..5 {
                tracker.record_access(i * 5 + j);
            }
        }

        // Check pattern detection
        let dimensions = vec![5, 5];
        let patterns = tracker.detect_dimensional_patterns(&dimensions);
        assert!(!patterns.is_empty());
        assert!(patterns.contains(&"matrix_traversal_row_major".to_string()));

        // Clear history
        tracker.clear_history();

        // Record column-major traversal
        for j in 0..5 {
            for i in 0..5 {
                tracker.record_access(i * 5 + j);
            }
        }

        // Check pattern detection
        let patterns = tracker.detect_dimensional_patterns(&dimensions);
        assert!(!patterns.is_empty());
        assert!(patterns.contains(&"matrix_traversal_col_major".to_string()));
    }

    #[test]
    fn test_zigzag_pattern_detection() {
        let config = PrefetchConfig {
            min_pattern_length: 4,
            history_size: 50,
            ..Default::default()
        };

        let mut tracker = AdaptivePatternTracker::new(config);

        // Set dimensions to a 5x5 matrix
        tracker.set_dimensions(vec![5, 5]);

        // Record zigzag traversal
        // Row 0: left to right
        for j in 0..5 {
            tracker.record_access(j);
        }
        // Row 1: right to left
        for j in (0..5).rev() {
            tracker.record_access(5 + j);
        }
        // Row 2: left to right
        for j in 0..5 {
            tracker.record_access(10 + j);
        }
        // Row 3: right to left
        for j in (0..5).rev() {
            tracker.record_access(15 + j);
        }

        // Get flat indices from history
        let indices: Vec<usize> = tracker.history.iter().map(|(idx, _, _)| *idx).collect();

        // Check zigzag detection
        let dimensions = vec![5, 5];
        assert!(tracker.detect_zigzag_pattern(&indices, &dimensions));
    }
}
