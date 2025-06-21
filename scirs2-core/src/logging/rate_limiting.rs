//! # Smart Rate Limiting for High-Frequency Log Events
//!
//! This module provides intelligent rate limiting for log events to prevent log spam
//! while ensuring important events are still captured and reported.

use crate::error::{CoreError, CoreResult, ErrorContext};
use once_cell::sync::Lazy;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::RwLock;
use std::time::{Duration, Instant};

/// Rate limiting strategy
#[derive(Debug, Clone)]
pub enum RateLimitStrategy {
    /// Fixed window: allow N events per window
    FixedWindow {
        max_events: u32,
        window_duration: Duration,
    },
    /// Sliding window: allow N events in any sliding window
    SlidingWindow {
        max_events: u32,
        window_duration: Duration,
    },
    /// Token bucket: allow bursts but maintain average rate
    TokenBucket {
        capacity: u32,
        refill_rate: f64, // tokens per second
    },
    /// Exponential backoff: increasing delays for repeated events
    ExponentialBackoff {
        initial_delay: Duration,
        max_delay: Duration,
        multiplier: f64,
    },
    /// Adaptive: automatically adjust based on event frequency and system load
    Adaptive {
        base_max_events: u32,
        base_window: Duration,
        load_threshold: f64,
    },
}

impl Default for RateLimitStrategy {
    fn default() -> Self {
        RateLimitStrategy::SlidingWindow {
            max_events: 10,
            window_duration: Duration::from_secs(60),
        }
    }
}

/// Log event classification for smart filtering
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EventClass {
    /// Critical errors that should never be rate limited
    Critical,
    /// Important errors that need careful rate limiting
    Error,
    /// Warnings that can be moderately rate limited
    Warning,
    /// Info messages that can be heavily rate limited
    Info,
    /// Debug messages that can be aggressively rate limited
    Debug,
    /// Trace messages that can be very aggressively rate limited
    Trace,
    /// Custom event class
    Custom(String),
}

impl EventClass {
    /// Get default priority for this event class (lower = higher priority)
    pub fn priority(&self) -> u8 {
        match self {
            EventClass::Critical => 0,
            EventClass::Error => 1,
            EventClass::Warning => 2,
            EventClass::Info => 3,
            EventClass::Debug => 4,
            EventClass::Trace => 5,
            EventClass::Custom(_) => 3, // Default to info level
        }
    }

    /// Check if this event class should bypass rate limiting
    pub fn bypass_rate_limiting(&self) -> bool {
        matches!(self, EventClass::Critical)
    }
}

/// Log event for rate limiting analysis
#[derive(Debug, Clone)]
pub struct LogEvent {
    /// Event message or template
    pub message: String,
    /// Event classification
    pub class: EventClass,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
    /// Timestamp when event occurred
    pub timestamp: Instant,
    /// Event source (file, module, etc.)
    pub source: Option<String>,
    /// Unique identifier for similar events
    pub fingerprint: u64,
}

impl LogEvent {
    /// Create a new log event
    pub fn new(message: String, class: EventClass) -> Self {
        let fingerprint = Self::calculate_fingerprint(&message, &class);
        Self {
            message,
            class,
            metadata: HashMap::new(),
            timestamp: Instant::now(),
            source: None,
            fingerprint,
        }
    }

    /// Create with source information
    pub fn with_source(mut self, source: String) -> Self {
        self.source = Some(source);
        // Recalculate fingerprint to include source
        self.fingerprint = Self::calculate_fingerprint(&self.message, &self.class);
        if let Some(ref source) = self.source {
            let mut hasher = DefaultHasher::new();
            self.fingerprint.hash(&mut hasher);
            source.hash(&mut hasher);
            self.fingerprint = hasher.finish();
        }
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Calculate fingerprint for event deduplication
    fn calculate_fingerprint(message: &str, class: &EventClass) -> u64 {
        let mut hasher = DefaultHasher::new();
        message.hash(&mut hasher);
        class.hash(&mut hasher);
        hasher.finish()
    }

    /// Check if this event is similar to another (same fingerprint)
    pub fn is_similar(&self, other: &LogEvent) -> bool {
        self.fingerprint == other.fingerprint
    }
}

/// Rate limiter state for a specific event type
#[derive(Debug, Clone)]
struct RateLimiterState {
    /// Strategy being used
    strategy: RateLimitStrategy,
    /// Event timestamps for sliding window
    event_times: Vec<Instant>,
    /// Token bucket state
    tokens: f64,
    /// Last token refill time
    last_refill: Instant,
    /// Exponential backoff state
    next_allowed_time: Instant,
    /// Current backoff delay
    current_delay: Duration,
    /// Suppressed event count
    suppressed_count: u32,
    /// Last time we logged a suppression summary
    last_summary_time: Instant,
}

impl RateLimiterState {
    /// Create new rate limiter state
    fn new(strategy: RateLimitStrategy) -> Self {
        let now = Instant::now();
        let tokens = match &strategy {
            RateLimitStrategy::TokenBucket { capacity, .. } => *capacity as f64,
            _ => 0.0,
        };

        Self {
            strategy,
            event_times: Vec::new(),
            tokens,
            last_refill: now,
            next_allowed_time: now,
            current_delay: Duration::from_secs(0),
            suppressed_count: 0,
            last_summary_time: now,
        }
    }

    /// Check if an event should be allowed
    fn should_allow(&mut self, event: &LogEvent) -> RateLimitDecision {
        let now = event.timestamp;

        // Critical events always bypass rate limiting
        if event.class.bypass_rate_limiting() {
            return RateLimitDecision::Allow;
        }

        match &self.strategy {
            RateLimitStrategy::FixedWindow {
                max_events,
                window_duration,
            } => self.should_allow_fixed_window(*max_events, *window_duration, now),
            RateLimitStrategy::SlidingWindow {
                max_events,
                window_duration,
            } => self.should_allow_sliding_window(*max_events, *window_duration, now),
            RateLimitStrategy::TokenBucket {
                capacity,
                refill_rate,
            } => self.should_allow_token_bucket(*capacity, *refill_rate, now),
            RateLimitStrategy::ExponentialBackoff {
                initial_delay,
                max_delay,
                multiplier,
            } => {
                self.should_allow_exponential_backoff(*initial_delay, *max_delay, *multiplier, now)
            }
            RateLimitStrategy::Adaptive {
                base_max_events,
                base_window,
                load_threshold,
            } => self.should_allow_adaptive(*base_max_events, *base_window, *load_threshold, now),
        }
    }

    fn should_allow_fixed_window(
        &mut self,
        max_events: u32,
        window_duration: Duration,
        now: Instant,
    ) -> RateLimitDecision {
        // Remove events outside the current window
        let window_start = now.checked_sub(window_duration).unwrap_or(Instant::now());
        self.event_times.retain(|&time| time >= window_start);

        if self.event_times.len() < max_events as usize {
            self.event_times.push(now);
            RateLimitDecision::Allow
        } else {
            self.suppressed_count += 1;
            RateLimitDecision::Suppress {
                reason: format!(
                    "Fixed window limit exceeded ({} events in {:?})",
                    max_events, window_duration
                ),
                retry_after: Some(window_start + window_duration),
            }
        }
    }

    fn should_allow_sliding_window(
        &mut self,
        max_events: u32,
        window_duration: Duration,
        now: Instant,
    ) -> RateLimitDecision {
        // Remove events outside the sliding window
        let window_start = now.checked_sub(window_duration).unwrap_or(Instant::now());
        self.event_times.retain(|&time| time >= window_start);

        if self.event_times.len() < max_events as usize {
            self.event_times.push(now);
            RateLimitDecision::Allow
        } else {
            self.suppressed_count += 1;
            // Calculate when the oldest event will expire
            let retry_after = self
                .event_times
                .first()
                .map(|&oldest| oldest + window_duration);
            RateLimitDecision::Suppress {
                reason: format!(
                    "Sliding window limit exceeded ({} events in {:?})",
                    max_events, window_duration
                ),
                retry_after,
            }
        }
    }

    fn should_allow_token_bucket(
        &mut self,
        capacity: u32,
        refill_rate: f64,
        now: Instant,
    ) -> RateLimitDecision {
        // Refill tokens based on elapsed time
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * refill_rate).min(capacity as f64);
        self.last_refill = now;

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            RateLimitDecision::Allow
        } else {
            self.suppressed_count += 1;
            // Calculate when next token will be available
            let time_to_token = Duration::from_secs_f64((1.0 - self.tokens) / refill_rate);
            RateLimitDecision::Suppress {
                reason: format!("Token bucket empty (refill rate: {:.2}/sec)", refill_rate),
                retry_after: Some(now + time_to_token),
            }
        }
    }

    fn should_allow_exponential_backoff(
        &mut self,
        initial_delay: Duration,
        max_delay: Duration,
        multiplier: f64,
        now: Instant,
    ) -> RateLimitDecision {
        if now >= self.next_allowed_time {
            // Reset delay after successful allowance
            self.current_delay = initial_delay;
            self.next_allowed_time = now + self.current_delay;
            RateLimitDecision::Allow
        } else {
            self.suppressed_count += 1;
            // Increase delay for next time
            self.current_delay = Duration::from_secs_f64(
                (self.current_delay.as_secs_f64() * multiplier).min(max_delay.as_secs_f64()),
            );
            self.next_allowed_time = now + self.current_delay;

            RateLimitDecision::Suppress {
                reason: format!(
                    "Exponential backoff (current delay: {:?})",
                    self.current_delay
                ),
                retry_after: Some(self.next_allowed_time),
            }
        }
    }

    fn should_allow_adaptive(
        &mut self,
        base_max_events: u32,
        base_window: Duration,
        load_threshold: f64,
        now: Instant,
    ) -> RateLimitDecision {
        // Simple adaptive strategy: adjust limits based on system load
        // In a real implementation, this would check actual system metrics
        let current_load = self.estimate_system_load();

        let adjusted_max_events = if current_load > load_threshold {
            // Reduce limits under high load
            (base_max_events as f64 * (2.0 - current_load / load_threshold)).max(1.0) as u32
        } else {
            base_max_events
        };

        // Use sliding window with adjusted limits
        let window_start = now.checked_sub(base_window).unwrap_or(Instant::now());
        self.event_times.retain(|&time| time >= window_start);

        if self.event_times.len() < adjusted_max_events as usize {
            self.event_times.push(now);
            RateLimitDecision::Allow
        } else {
            self.suppressed_count += 1;
            RateLimitDecision::Suppress {
                reason: format!(
                    "Adaptive limit exceeded (load: {:.2}, limit: {})",
                    current_load, adjusted_max_events
                ),
                retry_after: self.event_times.first().map(|&oldest| oldest + base_window),
            }
        }
    }

    /// Simple system load estimation (placeholder)
    fn estimate_system_load(&self) -> f64 {
        // In a real implementation, this would check CPU usage, memory pressure, etc.
        // For now, estimate based on recent event frequency
        let recent_events = self.event_times.len();
        (recent_events as f64 / 100.0).min(1.0)
    }

    /// Check if we should log a suppression summary
    fn should_log_summary(&mut self, summary_interval: Duration) -> Option<SuppressionSummary> {
        let now = Instant::now();
        if self.suppressed_count > 0
            && now.duration_since(self.last_summary_time) >= summary_interval
        {
            let summary = SuppressionSummary {
                suppressed_count: self.suppressed_count,
                time_period: now.duration_since(self.last_summary_time),
                strategy: format!("{:?}", self.strategy),
            };

            self.suppressed_count = 0;
            self.last_summary_time = now;
            Some(summary)
        } else {
            None
        }
    }
}

/// Decision from rate limiter
#[derive(Debug, Clone)]
pub enum RateLimitDecision {
    /// Allow the event to be logged
    Allow,
    /// Suppress the event
    Suppress {
        /// Reason for suppression
        reason: String,
        /// When the event might be allowed again
        retry_after: Option<Instant>,
    },
}

/// Summary of suppressed events
#[derive(Debug, Clone)]
pub struct SuppressionSummary {
    /// Number of events suppressed
    pub suppressed_count: u32,
    /// Time period over which events were suppressed
    pub time_period: Duration,
    /// Strategy that caused suppression
    pub strategy: String,
}

impl std::fmt::Display for SuppressionSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Suppressed {} events over {:?} using {}",
            self.suppressed_count, self.time_period, self.strategy
        )
    }
}

/// Smart rate limiter that manages multiple event types
pub struct SmartRateLimiter {
    /// Rate limiter states per event fingerprint
    limiters: RwLock<HashMap<u64, RateLimiterState>>,
    /// Global configuration
    config: RateLimiterConfig,
    /// Statistics
    stats: RwLock<RateLimiterStats>,
}

/// Configuration for the rate limiter
#[derive(Debug, Clone)]
pub struct RateLimiterConfig {
    /// Default strategy for new event types
    pub default_strategy: RateLimitStrategy,
    /// Strategies per event class
    pub class_strategies: HashMap<EventClass, RateLimitStrategy>,
    /// How often to log suppression summaries
    pub summary_interval: Duration,
    /// Maximum number of unique event types to track
    pub max_tracked_events: usize,
    /// Whether to enable adaptive features
    pub enable_adaptive: bool,
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        let mut class_strategies = HashMap::new();

        // Critical events are never rate limited
        class_strategies.insert(
            EventClass::Critical,
            RateLimitStrategy::SlidingWindow {
                max_events: u32::MAX,
                window_duration: Duration::from_secs(1),
            },
        );

        // Error events get generous limits
        class_strategies.insert(
            EventClass::Error,
            RateLimitStrategy::SlidingWindow {
                max_events: 50,
                window_duration: Duration::from_secs(60),
            },
        );

        // Warning events get moderate limits
        class_strategies.insert(
            EventClass::Warning,
            RateLimitStrategy::SlidingWindow {
                max_events: 20,
                window_duration: Duration::from_secs(60),
            },
        );

        // Info events get tighter limits
        class_strategies.insert(
            EventClass::Info,
            RateLimitStrategy::TokenBucket {
                capacity: 10,
                refill_rate: 0.5, // 1 event per 2 seconds
            },
        );

        // Debug events get very tight limits
        class_strategies.insert(
            EventClass::Debug,
            RateLimitStrategy::TokenBucket {
                capacity: 5,
                refill_rate: 0.1, // 1 event per 10 seconds
            },
        );

        // Trace events get extremely tight limits
        class_strategies.insert(
            EventClass::Trace,
            RateLimitStrategy::ExponentialBackoff {
                initial_delay: Duration::from_secs(1),
                max_delay: Duration::from_secs(300), // 5 minutes max
                multiplier: 2.0,
            },
        );

        Self {
            default_strategy: RateLimitStrategy::default(),
            class_strategies,
            summary_interval: Duration::from_secs(300), // 5 minutes
            max_tracked_events: 10000,
            enable_adaptive: true,
        }
    }
}

/// Statistics about rate limiter performance
#[derive(Debug, Clone, Default)]
pub struct RateLimiterStats {
    /// Total events processed
    pub total_events: u64,
    /// Total events allowed
    pub allowed_events: u64,
    /// Total events suppressed
    pub suppressed_events: u64,
    /// Events by class
    pub events_by_class: HashMap<EventClass, u64>,
    /// Unique event types tracked
    pub tracked_event_types: usize,
}

impl SmartRateLimiter {
    /// Create a new smart rate limiter
    pub fn new(config: RateLimiterConfig) -> Self {
        Self {
            limiters: RwLock::new(HashMap::new()),
            config,
            stats: RwLock::new(RateLimiterStats::default()),
        }
    }

    /// Check if an event should be allowed
    pub fn should_allow(&self, event: &LogEvent) -> CoreResult<RateLimitDecision> {
        // Update statistics
        {
            let mut stats = self.stats.write().map_err(|_| {
                CoreError::ComputationError(ErrorContext::new("Failed to acquire stats write lock"))
            })?;
            stats.total_events += 1;
            *stats
                .events_by_class
                .entry(event.class.clone())
                .or_insert(0) += 1;
        }

        // Get or create rate limiter for this event type
        let decision = {
            let mut limiters = self.limiters.write().map_err(|_| {
                CoreError::ComputationError(ErrorContext::new(
                    "Failed to acquire limiters write lock",
                ))
            })?;

            // Check if we've hit the maximum tracked events limit
            if limiters.len() >= self.config.max_tracked_events
                && !limiters.contains_key(&event.fingerprint)
            {
                // Remove oldest limiter (simple LRU approximation)
                if let Some((&oldest_key, _)) = limiters.iter().next() {
                    limiters.remove(&oldest_key);
                }
            }

            let limiter = limiters.entry(event.fingerprint).or_insert_with(|| {
                let strategy = self
                    .config
                    .class_strategies
                    .get(&event.class)
                    .cloned()
                    .unwrap_or_else(|| self.config.default_strategy.clone());
                RateLimiterState::new(strategy)
            });

            limiter.should_allow(event)
        };

        // Update statistics based on decision
        {
            let mut stats = self.stats.write().map_err(|_| {
                CoreError::ComputationError(ErrorContext::new("Failed to acquire stats write lock"))
            })?;

            match &decision {
                RateLimitDecision::Allow => stats.allowed_events += 1,
                RateLimitDecision::Suppress { .. } => stats.suppressed_events += 1,
            }

            stats.tracked_event_types = {
                let limiters = self.limiters.read().map_err(|_| {
                    CoreError::ComputationError(ErrorContext::new(
                        "Failed to acquire limiters read lock",
                    ))
                })?;
                limiters.len()
            };
        }

        Ok(decision)
    }

    /// Get suppression summaries for events that have been suppressed
    pub fn get_suppression_summaries(&self) -> CoreResult<Vec<(u64, SuppressionSummary)>> {
        let mut summaries = Vec::new();

        let mut limiters = self.limiters.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire limiters write lock"))
        })?;

        for (&fingerprint, limiter) in limiters.iter_mut() {
            if let Some(summary) = limiter.should_log_summary(self.config.summary_interval) {
                summaries.push((fingerprint, summary));
            }
        }

        Ok(summaries)
    }

    /// Get current statistics
    pub fn get_stats(&self) -> CoreResult<RateLimiterStats> {
        let stats = self.stats.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire stats read lock"))
        })?;
        Ok(stats.clone())
    }

    /// Clear all rate limiter state (useful for testing)
    pub fn clear(&self) -> CoreResult<()> {
        let mut limiters = self.limiters.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire limiters write lock"))
        })?;
        limiters.clear();

        let mut stats = self.stats.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire stats write lock"))
        })?;
        *stats = RateLimiterStats::default();

        Ok(())
    }

    /// Update configuration
    pub fn update_config(&mut self, config: RateLimiterConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub const fn get_config(&self) -> &RateLimiterConfig {
        &self.config
    }
}

impl Default for SmartRateLimiter {
    fn default() -> Self {
        Self::new(RateLimiterConfig::default())
    }
}

/// Global rate limiter instance
static GLOBAL_RATE_LIMITER: Lazy<SmartRateLimiter> = Lazy::new(SmartRateLimiter::default);

/// Get the global rate limiter
pub fn global_rate_limiter() -> &'static SmartRateLimiter {
    &GLOBAL_RATE_LIMITER
}

/// Convenience functions for common use cases
pub mod utils {
    use super::*;

    /// Create a log event for an error message
    pub fn error_event(message: String) -> LogEvent {
        LogEvent::new(message, EventClass::Error)
    }

    /// Create a log event for a warning message
    pub fn warning_event(message: String) -> LogEvent {
        LogEvent::new(message, EventClass::Warning)
    }

    /// Create a log event for an info message
    pub fn info_event(message: String) -> LogEvent {
        LogEvent::new(message, EventClass::Info)
    }

    /// Create a log event for a debug message
    pub fn debug_event(message: String) -> LogEvent {
        LogEvent::new(message, EventClass::Debug)
    }

    /// Check if an event should be logged using the global rate limiter
    pub fn should_log(event: &LogEvent) -> bool {
        match global_rate_limiter().should_allow(event) {
            Ok(RateLimitDecision::Allow) => true,
            Ok(RateLimitDecision::Suppress { .. }) => false,
            Err(_) => true, // Log on error for safety
        }
    }

    /// Create a rate limiting strategy for high-frequency events
    pub fn high_frequency_strategy() -> RateLimitStrategy {
        RateLimitStrategy::TokenBucket {
            capacity: 5,
            refill_rate: 0.1, // Very slow refill
        }
    }

    /// Create a rate limiting strategy for burst events
    pub fn burst_strategy() -> RateLimitStrategy {
        RateLimitStrategy::TokenBucket {
            capacity: 20,
            refill_rate: 2.0, // Allow bursts but maintain reasonable average
        }
    }

    /// Create a rate limiting strategy for periodic events
    pub fn periodic_strategy(period: Duration) -> RateLimitStrategy {
        RateLimitStrategy::FixedWindow {
            max_events: 1,
            window_duration: period,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_event_classification() {
        let event = LogEvent::new("Test message".to_string(), EventClass::Error);
        assert_eq!(event.class, EventClass::Error);
        assert_eq!(event.class.priority(), 1);
        assert!(!event.class.bypass_rate_limiting());

        let critical_event = LogEvent::new("Critical".to_string(), EventClass::Critical);
        assert!(critical_event.class.bypass_rate_limiting());
    }

    #[test]
    fn test_sliding_window_rate_limiting() {
        let mut state = RateLimiterState::new(RateLimitStrategy::SlidingWindow {
            max_events: 3,
            window_duration: Duration::from_millis(100),
        });

        let event = LogEvent::new("Test".to_string(), EventClass::Info);

        // First 3 events should be allowed
        for _ in 0..3 {
            match state.should_allow(&event) {
                RateLimitDecision::Allow => {}
                RateLimitDecision::Suppress { .. } => panic!("Should not suppress yet"),
            }
        }

        // 4th event should be suppressed
        match state.should_allow(&event) {
            RateLimitDecision::Allow => panic!("Should suppress"),
            RateLimitDecision::Suppress { .. } => {}
        }

        // After window expires, should allow again
        thread::sleep(Duration::from_millis(110));
        let new_event = LogEvent::new("Test".to_string(), EventClass::Info);
        match state.should_allow(&new_event) {
            RateLimitDecision::Allow => {}
            RateLimitDecision::Suppress { .. } => panic!("Should allow after window"),
        }
    }

    #[test]
    fn test_token_bucket() {
        let mut state = RateLimiterState::new(RateLimitStrategy::TokenBucket {
            capacity: 2,
            refill_rate: 10.0, // 10 tokens per second
        });

        let event = LogEvent::new("Test".to_string(), EventClass::Info);

        // Should allow 2 events immediately (full bucket)
        for _ in 0..2 {
            match state.should_allow(&event) {
                RateLimitDecision::Allow => {}
                RateLimitDecision::Suppress { .. } => panic!("Should not suppress yet"),
            }
        }

        // 3rd event should be suppressed (bucket empty)
        match state.should_allow(&event) {
            RateLimitDecision::Allow => panic!("Should suppress when bucket empty"),
            RateLimitDecision::Suppress { .. } => {}
        }
    }

    #[test]
    fn test_smart_rate_limiter() {
        let limiter = SmartRateLimiter::default();

        let error_event = LogEvent::new("Error message".to_string(), EventClass::Error);
        let debug_event = LogEvent::new("Debug message".to_string(), EventClass::Debug);

        // Error events should have more generous limits
        let error_decision = limiter.should_allow(&error_event).unwrap();
        assert!(matches!(error_decision, RateLimitDecision::Allow));

        // Debug events should have tighter limits
        let debug_decision = limiter.should_allow(&debug_event).unwrap();
        assert!(matches!(debug_decision, RateLimitDecision::Allow)); // First one should be allowed

        // Stats should be updated
        let stats = limiter.get_stats().unwrap();
        assert_eq!(stats.total_events, 2);
        assert_eq!(stats.allowed_events, 2);
    }

    #[test]
    fn test_event_fingerprinting() {
        let event1 = LogEvent::new("Same message".to_string(), EventClass::Info);
        let event2 = LogEvent::new("Same message".to_string(), EventClass::Info);
        let event3 = LogEvent::new("Different message".to_string(), EventClass::Info);

        assert!(event1.is_similar(&event2));
        assert!(!event1.is_similar(&event3));
    }

    #[test]
    fn test_suppression_summary() {
        let summary = SuppressionSummary {
            suppressed_count: 10,
            time_period: Duration::from_secs(60),
            strategy: "TokenBucket".to_string(),
        };

        let display_str = format!("{}", summary);
        assert!(display_str.contains("10 events"));
        assert!(display_str.contains("TokenBucket"));
    }

    #[test]
    fn test_critical_events_bypass() {
        let limiter = SmartRateLimiter::default();

        // Create a critical event
        let critical_event = LogEvent::new("Critical error".to_string(), EventClass::Critical);

        // Critical events should always be allowed, regardless of rate limiting
        for _ in 0..1000 {
            let decision = limiter.should_allow(&critical_event).unwrap();
            assert!(matches!(decision, RateLimitDecision::Allow));
        }
    }

    #[test]
    fn test_utils_functions() {
        let error_event = utils::error_event("Test error".to_string());
        assert_eq!(error_event.class, EventClass::Error);

        let warning_event = utils::warning_event("Test warning".to_string());
        assert_eq!(warning_event.class, EventClass::Warning);

        // Test the should_log utility
        let info_event = utils::info_event("Test info".to_string());
        assert!(utils::should_log(&info_event)); // First call should be allowed
    }
}
