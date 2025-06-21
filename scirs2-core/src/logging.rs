//! # Logging and Diagnostics
//!
//! This module provides structured logging and diagnostics utilities for scientific computing.
//!
//! ## Features
//!
//! * Structured logging for scientific computing
//! * Enhanced progress tracking with multiple visualization styles
//! * Performance metrics collection
//! * Log filtering and formatting
//! * Multi-progress tracking for parallel operations
//! * Adaptive update rates and predictive ETA calculations
//!
//! ## Usage
//!
//! ```rust,no_run
//! use scirs2_core::logging::{Logger, LogLevel, ProgressTracker};
//! use scirs2_core::logging::progress::{ProgressBuilder, ProgressStyle};
//!
//! // Create a logger
//! let logger = Logger::new("matrix_operations");
//!
//! // Log messages at different levels
//! logger.info("Starting matrix multiplication");
//! logger.debug("Using algorithm: Standard");
//!
//! // Create an enhanced progress tracker
//! let mut progress = ProgressBuilder::new("Matrix multiplication", 1000)
//!     .style(ProgressStyle::DetailedBar)
//!     .show_statistics(true)
//!     .build();
//!
//! progress.start();
//!
//! for i in 0..1000 {
//!     // Perform computation
//!
//!     // Update progress
//!     progress.update(i + 1);
//!
//!     // Log intermediate results at low frequency to avoid flooding logs
//!     if i % 100 == 0 {
//!         logger.debug(&format!("Completed {}/{} iterations", i + 1, 1000));
//!     }
//! }
//!
//! // Complete the progress tracking
//! progress.finish();
//!
//! logger.info("Matrix multiplication completed");
//! ```

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::fmt::Display;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// Enhanced progress tracking module
pub mod progress;

/// Smart rate limiting for high-frequency log events
pub mod rate_limiting;

/// Log level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    /// Trace level for detailed debugging
    Trace = 0,
    /// Debug level for debugging information
    Debug = 1,
    /// Info level for general information
    Info = 2,
    /// Warning level for potential issues
    Warn = 3,
    /// Error level for error conditions
    Error = 4,
    /// Critical level for critical errors
    Critical = 5,
}

impl LogLevel {
    /// Convert a log level to a string
    pub const fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
            LogLevel::Critical => "CRITICAL",
        }
    }
}

/// Structured log entry
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// Timestamp of the log entry
    pub timestamp: std::time::SystemTime,
    /// Log level
    pub level: LogLevel,
    /// Module or component name
    pub module: String,
    /// Log message
    pub message: String,
    /// Additional context fields
    pub fields: HashMap<String, String>,
}

/// Logger configuration
#[derive(Debug, Clone)]
pub struct LoggerConfig {
    /// Minimum log level to record
    pub min_level: LogLevel,
    /// Enable/disable timestamps
    pub show_timestamps: bool,
    /// Enable/disable module names
    pub show_modules: bool,
    /// Module-specific log levels
    pub module_levels: HashMap<String, LogLevel>,
}

impl Default for LoggerConfig {
    fn default() -> Self {
        Self {
            min_level: LogLevel::Info,
            show_timestamps: true,
            show_modules: true,
            module_levels: HashMap::new(),
        }
    }
}

/// Global logger configuration
static LOGGER_CONFIG: Lazy<Mutex<LoggerConfig>> = Lazy::new(|| Mutex::new(LoggerConfig::default()));

/// Configure the global logger
pub fn configure_logger(config: LoggerConfig) {
    let mut global_config = LOGGER_CONFIG.lock().unwrap();
    *global_config = config;
}

/// Set the global minimum log level
pub fn set_min_log_level(level: LogLevel) {
    let mut config = LOGGER_CONFIG.lock().unwrap();
    config.min_level = level;
}

/// Set a module-specific log level
pub fn set_module_log_level(module: &str, level: LogLevel) {
    let mut config = LOGGER_CONFIG.lock().unwrap();
    config.module_levels.insert(module.to_string(), level);
}

/// Handler trait for processing log entries
pub trait LogHandler: Send + Sync {
    /// Handle a log entry
    fn handle(&self, entry: &LogEntry);
}

/// Console log handler
pub struct ConsoleLogHandler {
    /// Format string for log entries
    pub format: String,
}

impl Default for ConsoleLogHandler {
    fn default() -> Self {
        Self {
            format: "[{level}] {module}: {message}".to_string(),
        }
    }
}

impl LogHandler for ConsoleLogHandler {
    fn handle(&self, entry: &LogEntry) {
        let mut output = self.format.clone();

        // Replace placeholders in the format string
        output = output.replace("{level}", entry.level.as_str());
        output = output.replace("{module}", &entry.module);
        output = output.replace("{message}", &entry.message);

        if self.format.contains("{timestamp}") {
            let datetime = chrono::DateTime::<chrono::Utc>::from(entry.timestamp);
            output = output.replace(
                "{timestamp}",
                &datetime.format("%Y-%m-%d %H:%M:%S%.3f").to_string(),
            );
        }

        if self.format.contains("{fields}") {
            let fields_str = entry
                .fields
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(", ");
            output = output.replace("{fields}", &fields_str);
        }

        // Print to the appropriate output stream based on level
        match entry.level {
            LogLevel::Error | LogLevel::Critical => eprintln!("{}", output),
            _ => println!("{}", output),
        }
    }
}

/// File log handler
pub struct FileLogHandler {
    /// Path to the log file
    pub file_path: String,
    /// Format string for log entries
    pub format: String,
}

impl LogHandler for FileLogHandler {
    fn handle(&self, entry: &LogEntry) {
        // This is a simplified implementation
        // A real implementation would handle file I/O more efficiently

        let mut output = self.format.clone();

        // Replace placeholders in the format string
        output = output.replace("{level}", entry.level.as_str());
        output = output.replace("{module}", &entry.module);
        output = output.replace("{message}", &entry.message);

        if self.format.contains("{timestamp}") {
            let datetime = chrono::DateTime::<chrono::Utc>::from(entry.timestamp);
            output = output.replace(
                "{timestamp}",
                &datetime.format("%Y-%m-%d %H:%M:%S%.3f").to_string(),
            );
        }

        if self.format.contains("{fields}") {
            let fields_str = entry
                .fields
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(", ");
            output = output.replace("{fields}", &fields_str);
        }

        // Append to the log file
        // This would use proper error handling and buffering in a real implementation
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)
        {
            use std::io::Write;
            let _ = writeln!(file, "{}", output);
        }
    }
}

/// Global log handlers
static LOG_HANDLERS: Lazy<Mutex<Vec<Arc<dyn LogHandler>>>> = Lazy::new(|| {
    let console_handler = Arc::new(ConsoleLogHandler::default());
    Mutex::new(vec![console_handler])
});

/// Register a log handler
pub fn register_log_handler(handler: Arc<dyn LogHandler>) {
    let mut handlers = LOG_HANDLERS.lock().unwrap();
    handlers.push(handler);
}

/// Clear all log handlers
pub fn clear_log_handlers() {
    let mut handlers = LOG_HANDLERS.lock().unwrap();
    handlers.clear();
}

/// Reset log handlers to the default configuration
pub fn reset_log_handlers() {
    let mut handlers = LOG_HANDLERS.lock().unwrap();
    handlers.clear();
    handlers.push(Arc::new(ConsoleLogHandler::default()));
}

/// Logger for a specific module
#[derive(Clone)]
pub struct Logger {
    /// Module name
    module: String,
    /// Additional context fields
    fields: HashMap<String, String>,
}

impl Logger {
    /// Create a new logger for the specified module
    pub fn new(module: &str) -> Self {
        Self {
            module: module.to_string(),
            fields: HashMap::new(),
        }
    }

    /// Add a context field to the logger
    pub fn with_field<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Display,
    {
        self.fields.insert(key.into(), format!("{}", value));
        self
    }

    /// Add multiple context fields to the logger
    pub fn with_fields<K, V, I>(mut self, fields: I) -> Self
    where
        K: Into<String>,
        V: Display,
        I: IntoIterator<Item = (K, V)>,
    {
        for (key, value) in fields {
            self.fields.insert(key.into(), format!("{}", value));
        }
        self
    }

    /// Log a message at a specific level
    pub fn log(&self, level: LogLevel, message: &str) {
        // Check if this log should be processed based on configuration
        let config = LOGGER_CONFIG.lock().unwrap();
        let module_level = config
            .module_levels
            .get(&self.module)
            .copied()
            .unwrap_or(config.min_level);

        if level < module_level {
            return;
        }

        // Create the log entry
        let entry = LogEntry {
            timestamp: std::time::SystemTime::now(),
            level,
            module: self.module.clone(),
            message: message.to_string(),
            fields: self.fields.clone(),
        };

        // Process the log entry with all registered handlers
        let handlers = LOG_HANDLERS.lock().unwrap();
        for handler in handlers.iter() {
            handler.handle(&entry);
        }
    }

    /// Log a message at trace level
    pub fn trace(&self, message: &str) {
        self.log(LogLevel::Trace, message);
    }

    /// Log a message at debug level
    pub fn debug(&self, message: &str) {
        self.log(LogLevel::Debug, message);
    }

    /// Log a message at info level
    pub fn info(&self, message: &str) {
        self.log(LogLevel::Info, message);
    }

    /// Log a message at warning level
    pub fn warn(&self, message: &str) {
        self.log(LogLevel::Warn, message);
    }

    /// Log a message at error level
    pub fn error(&self, message: &str) {
        self.log(LogLevel::Error, message);
    }

    /// Log a message at critical level
    pub fn critical(&self, message: &str) {
        self.log(LogLevel::Critical, message);
    }

    /// Create an enhanced progress tracker using the logger's context
    pub fn track_progress(
        &self,
        description: &str,
        total: u64,
    ) -> progress::EnhancedProgressTracker {
        use progress::{ProgressBuilder, ProgressStyle};

        let builder = ProgressBuilder::new(description, total)
            .style(ProgressStyle::DetailedBar)
            .show_statistics(true);

        let mut tracker = builder.build();

        // Log the start of progress tracking
        self.info(&format!("Starting progress tracking: {}", description));

        tracker.start();
        tracker
    }

    /// Log a message with progress update
    pub fn info_with_progress(
        &self,
        message: &str,
        progress: &mut progress::EnhancedProgressTracker,
        update: u64,
    ) {
        self.info(message);
        progress.update(update);
    }

    /// Execute an operation with progress tracking
    pub fn with_progress<F, R>(&self, description: &str, total: u64, operation: F) -> R
    where
        F: FnOnce(&mut progress::EnhancedProgressTracker) -> R,
    {
        let mut progress = self.track_progress(description, total);
        let result = operation(&mut progress);
        progress.finish();

        // Log completion
        let stats = progress.stats();
        self.info(&format!(
            "Completed progress tracking: {} - {:.1}s elapsed",
            description,
            stats.elapsed.as_secs_f64()
        ));

        result
    }
}

/// Progress tracker for long-running operations
pub struct ProgressTracker {
    /// Operation name
    name: String,
    /// Total number of steps
    total: usize,
    /// Current progress
    current: usize,
    /// Start time
    start_time: Instant,
    /// Last update time
    last_update: Instant,
    /// Minimum time between progress updates
    update_interval: Duration,
    /// Associated logger
    logger: Logger,
}

impl ProgressTracker {
    /// Create a new progress tracker
    pub fn new(name: &str, total: usize) -> Self {
        let now = Instant::now();
        let logger = Logger::new("progress").with_field("operation", name);

        logger.info(&format!("Starting operation: {}", name));

        Self {
            name: name.to_string(),
            total,
            current: 0,
            start_time: now,
            last_update: now,
            update_interval: Duration::from_millis(500), // Update at most every 500ms
            logger,
        }
    }

    /// Set the minimum interval between progress updates
    pub fn set_update_interval(&mut self, interval: Duration) {
        self.update_interval = interval;
    }

    /// Update the current progress
    pub fn update(&mut self, current: usize) {
        self.current = current;

        let now = Instant::now();

        // Only log an update if enough time has passed since the last update
        if now.duration_since(self.last_update) >= self.update_interval {
            self.last_update = now;

            let elapsed = now.duration_since(self.start_time);
            let percent = (self.current as f64 / self.total as f64) * 100.0;

            let eta = if self.current > 0 {
                let time_per_item = elapsed.as_secs_f64() / self.current as f64;
                let remaining = time_per_item * (self.total - self.current) as f64;
                format!("ETA: {:.1}s", remaining)
            } else {
                "ETA: calculating...".to_string()
            };

            self.logger.debug(&format!(
                "{}: {}/{} ({:.1}%) - Elapsed: {:.1}s - {}",
                self.name,
                self.current,
                self.total,
                percent,
                elapsed.as_secs_f64(),
                eta
            ));
        }
    }

    /// Mark the operation as complete
    pub fn complete(&mut self) {
        let elapsed = self.start_time.elapsed();
        self.current = self.total;

        self.logger.info(&format!(
            "{} completed: {}/{} (100%) - Total time: {:.1}s",
            self.name,
            self.total,
            self.total,
            elapsed.as_secs_f64()
        ));
    }

    /// Get the current progress as a percentage
    pub fn progress_percent(&self) -> f64 {
        (self.current as f64 / self.total as f64) * 100.0
    }

    /// Get the elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Estimate the remaining time
    pub fn eta(&self) -> Option<Duration> {
        if self.current == 0 {
            return None;
        }

        let elapsed = self.start_time.elapsed();
        let time_per_item = elapsed.as_secs_f64() / self.current as f64;
        let remaining_secs = time_per_item * (self.total - self.current) as f64;

        Some(Duration::from_secs_f64(remaining_secs))
    }
}

/// Initialize the default logging system
pub fn init() {
    // Register the default console handler if not already done
    let handlers = LOG_HANDLERS.lock().unwrap();
    if handlers.is_empty() {
        drop(handlers);
        reset_log_handlers();
    }
}

/// Get a logger for the current module
#[macro_export]
macro_rules! get_logger {
    () => {
        $crate::logging::Logger::new(module_path!())
    };
    ($name:expr) => {
        $crate::logging::Logger::new($name)
    };
}

// # Distributed Logging and Adaptive Rate Limiting (Alpha 6)
//
// This section provides advanced distributed logging capabilities with
// aggregation, adaptive rate limiting, and multi-node coordination.

/// Distributed logging capabilities for multi-node computations
pub mod distributed {
    use super::*;
    use std::collections::{HashMap, VecDeque};
    use std::fmt;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, RwLock};
    use std::thread;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    /// Node identifier for distributed logging
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct NodeId {
        name: String,
        instance_id: String,
    }

    impl NodeId {
        /// Create a new node identifier
        pub fn new(name: String, instance_id: String) -> Self {
            Self { name, instance_id }
        }

        /// Create from hostname and process ID
        pub fn from_hostname() -> Self {
            let hostname = std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string());
            let pid = std::process::id();
            Self::new(hostname, pid.to_string())
        }

        /// Get node name
        pub fn name(&self) -> &str {
            &self.name
        }

        /// Get instance ID
        pub fn instance_id(&self) -> &str {
            &self.instance_id
        }
    }

    impl fmt::Display for NodeId {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}:{}", self.name, self.instance_id)
        }
    }

    /// Distributed log entry with metadata
    #[derive(Debug, Clone)]
    pub struct DistributedLogEntry {
        /// Unique entry ID
        pub id: u64,
        /// Source node
        #[allow(dead_code)]
        pub node_id: NodeId,
        /// Timestamp (Unix epoch milliseconds)
        pub timestamp: u64,
        /// Log level
        pub level: LogLevel,
        /// Logger name
        pub logger: String,
        /// Message content
        pub message: String,
        /// Additional context fields
        pub context: HashMap<String, String>,
        /// Sequence number for ordering
        pub sequence: u64,
    }

    impl DistributedLogEntry {
        /// Create a new distributed log entry
        pub fn new(
            node_id: NodeId,
            level: LogLevel,
            logger: String,
            message: String,
            context: HashMap<String, String>,
        ) -> Self {
            static ID_COUNTER: AtomicU64 = AtomicU64::new(1);
            static SEQ_COUNTER: AtomicU64 = AtomicU64::new(1);

            Self {
                id: ID_COUNTER.fetch_add(1, Ordering::Relaxed),
                node_id,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
                level,
                logger,
                message,
                context,
                sequence: SEQ_COUNTER.fetch_add(1, Ordering::Relaxed),
            }
        }

        /// Get age of this log entry
        pub fn age(&self) -> Duration {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            Duration::from_millis(now.saturating_sub(self.timestamp))
        }
    }

    /// Log aggregator that collects and processes distributed log entries
    #[allow(dead_code)]
    pub struct LogAggregator {
        #[allow(dead_code)]
        node_id: NodeId,
        entries: Arc<RwLock<VecDeque<DistributedLogEntry>>>,
        max_entries: usize,
        aggregation_window: Duration,
        stats: Arc<RwLock<AggregationStats>>,
    }

    /// Statistics for log aggregation
    #[derive(Debug, Clone, Default)]
    pub struct AggregationStats {
        pub total_entries: u64,
        pub entries_by_level: HashMap<LogLevel, u64>,
        pub entries_by_node: HashMap<NodeId, u64>,
        pub dropped_entries: u64,
        pub aggregation_windows: u64,
    }

    impl LogAggregator {
        /// Create a new log aggregator
        pub fn new(node_id: NodeId, max_entries: usize, aggregation_window: Duration) -> Self {
            Self {
                node_id,
                entries: Arc::new(RwLock::new(VecDeque::new())),
                max_entries,
                aggregation_window,
                stats: Arc::new(RwLock::new(AggregationStats::default())),
            }
        }

        /// Add a log entry to the aggregator
        pub fn add_entry(&self, entry: DistributedLogEntry) {
            let mut entries = self.entries.write().unwrap();
            let mut stats = self.stats.write().unwrap();

            // Remove old entries beyond the window
            let cutoff = entry
                .timestamp
                .saturating_sub(self.aggregation_window.as_millis() as u64);
            while let Some(front) = entries.front() {
                if front.timestamp >= cutoff {
                    break;
                }
                let removed = entries.pop_front().unwrap();
                // Update stats for removed entry
                if let Some(count) = stats.entries_by_level.get_mut(&removed.level) {
                    *count = count.saturating_sub(1);
                }
                if let Some(count) = stats.entries_by_node.get_mut(&removed.node_id) {
                    *count = count.saturating_sub(1);
                }
            }

            // Add new entry
            if entries.len() >= self.max_entries {
                if let Some(removed) = entries.pop_front() {
                    stats.dropped_entries += 1;
                    // Update stats for dropped entry
                    if let Some(count) = stats.entries_by_level.get_mut(&removed.level) {
                        *count = count.saturating_sub(1);
                    }
                    if let Some(count) = stats.entries_by_node.get_mut(&removed.node_id) {
                        *count = count.saturating_sub(1);
                    }
                }
            }

            // Update stats for new entry
            stats.total_entries += 1;
            *stats.entries_by_level.entry(entry.level).or_insert(0) += 1;
            *stats
                .entries_by_node
                .entry(entry.node_id.clone())
                .or_insert(0) += 1;

            entries.push_back(entry);
        }

        /// Get all entries within the aggregation window
        pub fn get_entries(&self) -> Vec<DistributedLogEntry> {
            self.entries.read().unwrap().iter().cloned().collect()
        }

        /// Get entries filtered by level
        pub fn get_entries_by_level(&self, level: LogLevel) -> Vec<DistributedLogEntry> {
            self.entries
                .read()
                .unwrap()
                .iter()
                .filter(|entry| entry.level == level)
                .cloned()
                .collect()
        }

        /// Get entries from specific node
        pub fn get_entries_by_node(&self, node_id: &NodeId) -> Vec<DistributedLogEntry> {
            self.entries
                .read()
                .unwrap()
                .iter()
                .filter(|entry| &entry.node_id == node_id)
                .cloned()
                .collect()
        }

        /// Get aggregation statistics
        pub fn stats(&self) -> AggregationStats {
            self.stats.read().unwrap().clone()
        }

        /// Clear all entries
        pub fn clear(&self) {
            self.entries.write().unwrap().clear();
            *self.stats.write().unwrap() = AggregationStats::default();
        }
    }

    /// Adaptive rate limiter for high-frequency logging
    pub struct AdaptiveRateLimiter {
        max_rate: Arc<Mutex<f64>>, // Maximum messages per second
        current_rate: Arc<Mutex<f64>>,
        last_reset: Arc<Mutex<Instant>>,
        message_count: Arc<AtomicUsize>,
        window_duration: Duration,
        adaptation_factor: f64,
        min_rate: f64,
        max_rate_absolute: f64,
    }

    impl AdaptiveRateLimiter {
        /// Create a new adaptive rate limiter
        pub fn new(
            initial_max_rate: f64,
            window_duration: Duration,
            adaptation_factor: f64,
        ) -> Self {
            Self {
                max_rate: Arc::new(Mutex::new(initial_max_rate)),
                current_rate: Arc::new(Mutex::new(0.0)),
                last_reset: Arc::new(Mutex::new(Instant::now())),
                message_count: Arc::new(AtomicUsize::new(0)),
                window_duration,
                adaptation_factor,
                min_rate: initial_max_rate * 0.1, // 10% of initial rate
                max_rate_absolute: initial_max_rate * 10.0, // 10x initial rate
            }
        }

        /// Check if a message should be allowed through
        pub fn try_acquire(&self) -> bool {
            let now = Instant::now();
            let count = self.message_count.fetch_add(1, Ordering::Relaxed);

            let mut last_reset = self.last_reset.lock().unwrap();
            let elapsed = now.duration_since(*last_reset);

            if elapsed >= self.window_duration {
                // Reset window and update current rate
                let actual_rate = count as f64 / elapsed.as_secs_f64();
                {
                    let mut current_rate = self.current_rate.lock().unwrap();
                    *current_rate = actual_rate;
                }

                self.message_count.store(0, Ordering::Relaxed);
                *last_reset = now;

                // Adapt max rate based on actual usage
                self.adapt_rate(actual_rate);

                true // Allow message at window boundary
            } else {
                // Check if current rate exceeds limit
                let elapsed_secs = elapsed.as_secs_f64();
                if elapsed_secs < 0.001 {
                    // For very short durations, allow the message
                    true
                } else {
                    let current_rate = count as f64 / elapsed_secs;
                    let max_rate = *self.max_rate.lock().unwrap();
                    current_rate <= max_rate
                }
            }
        }

        /// Adapt the maximum rate based on observed patterns
        fn adapt_rate(&self, actual_rate: f64) {
            let mut max_rate = self.max_rate.lock().unwrap();

            // If actual rate is consistently lower, reduce max rate
            // If actual rate hits the limit, increase max rate
            if actual_rate < *max_rate * 0.5 {
                // Reduce max rate
                *max_rate = (*max_rate * (1.0 - self.adaptation_factor)).max(self.min_rate);
            } else if actual_rate >= *max_rate * 0.9 {
                // Increase max rate
                *max_rate =
                    (*max_rate * (1.0 + self.adaptation_factor)).min(self.max_rate_absolute);
            }
        }

        /// Get current rate statistics
        pub fn get_stats(&self) -> RateLimitStats {
            let current_rate = *self.current_rate.lock().unwrap();
            let max_rate = *self.max_rate.lock().unwrap();
            RateLimitStats {
                current_rate,
                max_rate,
                message_count: self.message_count.load(Ordering::Relaxed),
                window_duration: self.window_duration,
            }
        }

        /// Reset the rate limiter
        pub fn reset(&self) {
            *self.current_rate.lock().unwrap() = 0.0;
            *self.last_reset.lock().unwrap() = Instant::now();
            self.message_count.store(0, Ordering::Relaxed);
        }
    }

    /// Rate limiting statistics
    #[derive(Debug, Clone)]
    pub struct RateLimitStats {
        pub current_rate: f64,
        pub max_rate: f64,
        pub message_count: usize,
        pub window_duration: Duration,
    }

    /// Distributed logger that coordinates with multiple nodes
    pub struct DistributedLogger {
        #[allow(dead_code)]
        node_id: NodeId,
        local_logger: Logger,
        aggregator: Arc<LogAggregator>,
        rate_limiters: Arc<RwLock<HashMap<String, AdaptiveRateLimiter>>>,
        default_rate_limit: f64,
    }

    impl DistributedLogger {
        /// Create a new distributed logger
        pub fn new(
            node_id: NodeId,
            logger_name: &str,
            max_entries: usize,
            aggregation_window: Duration,
            default_rate_limit: f64,
        ) -> Self {
            let local_logger = Logger::new(logger_name);
            let aggregator = Arc::new(LogAggregator::new(
                node_id.clone(),
                max_entries,
                aggregation_window,
            ));

            Self {
                node_id,
                local_logger,
                aggregator,
                rate_limiters: Arc::new(RwLock::new(HashMap::new())),
                default_rate_limit,
            }
        }

        /// Log a message with adaptive rate limiting
        pub fn log_adaptive(
            &self,
            level: LogLevel,
            message: &str,
            context: Option<HashMap<String, String>>,
        ) {
            let logger_key = self.local_logger.module.clone();

            // Get or create rate limiter for this logger
            let should_log = {
                let rate_limiters = self.rate_limiters.read().unwrap();
                if let Some(limiter) = rate_limiters.get(&logger_key) {
                    limiter.try_acquire()
                } else {
                    drop(rate_limiters);

                    // Create new rate limiter
                    let mut rate_limiters = self.rate_limiters.write().unwrap();
                    let limiter = AdaptiveRateLimiter::new(
                        self.default_rate_limit,
                        Duration::from_secs(1),
                        0.1, // 10% adaptation factor
                    );
                    let should_log = limiter.try_acquire();
                    rate_limiters.insert(logger_key, limiter);
                    should_log
                }
            };

            if should_log {
                // Log locally
                self.local_logger.log(level, message);

                // Create distributed log entry
                let entry = DistributedLogEntry::new(
                    self.node_id.clone(),
                    level,
                    self.local_logger.module.clone(),
                    message.to_string(),
                    context.unwrap_or_default(),
                );

                // Add to aggregator
                self.aggregator.add_entry(entry);
            }
        }

        /// Convenience methods for different log levels
        pub fn error_adaptive(&self, message: &str) {
            self.log_adaptive(LogLevel::Error, message, None);
        }

        pub fn warn_adaptive(&self, message: &str) {
            self.log_adaptive(LogLevel::Warn, message, None);
        }

        pub fn info_adaptive(&self, message: &str) {
            self.log_adaptive(LogLevel::Info, message, None);
        }

        pub fn debug_adaptive(&self, message: &str) {
            self.log_adaptive(LogLevel::Debug, message, None);
        }

        /// Get aggregated log entries
        pub fn get_aggregated_logs(&self) -> Vec<DistributedLogEntry> {
            self.aggregator.get_entries()
        }

        /// Get rate limiting statistics for all loggers
        pub fn get_rate_stats(&self) -> HashMap<String, RateLimitStats> {
            self.rate_limiters
                .read()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.get_stats()))
                .collect()
        }

        /// Get aggregation statistics
        pub fn get_aggregation_stats(&self) -> AggregationStats {
            self.aggregator.stats()
        }

        /// Export logs to JSON format
        pub fn export_logs_json(&self) -> Result<String, Box<dyn std::error::Error>> {
            let entries = self.get_aggregated_logs();
            let stats = self.get_aggregation_stats();

            let export_data = serde_json::json!({
                "node_id": self.node_id.to_string(),
                "timestamp": SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis(),
                "stats": {
                    "total_entries": stats.total_entries,
                    "dropped_entries": stats.dropped_entries,
                    "aggregation_windows": stats.aggregation_windows
                },
                "entries": entries.iter().map(|entry| serde_json::json!({
                    "id": entry.id,
                    "node_id": entry.node_id.to_string(),
                    "timestamp": entry.timestamp,
                    "level": format!("{:?}", entry.level),
                    "logger": entry.logger,
                    "message": entry.message,
                    "context": entry.context,
                    "sequence": entry.sequence
                })).collect::<Vec<_>>()
            });

            Ok(serde_json::to_string_pretty(&export_data)?)
        }

        /// Clear all aggregated data
        pub fn clear_aggregated_data(&self) {
            self.aggregator.clear();

            // Reset rate limiters
            let rate_limiters = self.rate_limiters.write().unwrap();
            for limiter in rate_limiters.values() {
                limiter.reset();
            }
        }
    }

    /// Multi-node log coordinator for distributed systems
    pub struct MultiNodeCoordinator {
        nodes: Arc<RwLock<HashMap<NodeId, Arc<DistributedLogger>>>>,
        global_aggregator: Arc<LogAggregator>,
        coordination_interval: Duration,
        running: Arc<AtomicUsize>, // 0 = stopped, 1 = running
    }

    impl MultiNodeCoordinator {
        /// Create a new multi-node coordinator
        pub fn new(coordination_interval: Duration) -> Self {
            let global_node = NodeId::new("global".to_string(), "coordinator".to_string());
            let global_aggregator = Arc::new(LogAggregator::new(
                global_node,
                100000,                    // Large capacity for global aggregation
                Duration::from_secs(3600), // 1 hour window
            ));

            Self {
                nodes: Arc::new(RwLock::new(HashMap::new())),
                global_aggregator,
                coordination_interval,
                running: Arc::new(AtomicUsize::new(0)),
            }
        }

        /// Register a distributed logger
        pub fn register_node(&self, node_id: NodeId, logger: Arc<DistributedLogger>) {
            let mut nodes = self.nodes.write().unwrap();
            nodes.insert(node_id, logger);
        }

        /// Unregister a node
        pub fn unregister_node(&self, node_id: &NodeId) {
            let mut nodes = self.nodes.write().unwrap();
            nodes.remove(node_id);
        }

        /// Start coordination process
        pub fn start(&self) {
            if self
                .running
                .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                let nodes = self.nodes.clone();
                let global_aggregator = self.global_aggregator.clone();
                let interval = self.coordination_interval;
                let running = self.running.clone();

                thread::spawn(move || {
                    while running.load(Ordering::Relaxed) == 1 {
                        // Collect logs from all nodes
                        let nodes_guard = nodes.read().unwrap();
                        for logger in nodes_guard.values() {
                            let entries = logger.get_aggregated_logs();
                            for entry in entries {
                                global_aggregator.add_entry(entry);
                            }
                        }
                        drop(nodes_guard);

                        thread::sleep(interval);
                    }
                });
            }
        }

        /// Stop coordination process
        pub fn stop(&self) {
            self.running.store(0, Ordering::Relaxed);
        }

        /// Get global aggregated statistics
        pub fn get_global_stats(&self) -> AggregationStats {
            self.global_aggregator.stats()
        }

        /// Get all global log entries
        pub fn get_global_entries(&self) -> Vec<DistributedLogEntry> {
            self.global_aggregator.get_entries()
        }

        /// Export global logs to JSON
        pub fn export_global_logs_json(&self) -> Result<String, Box<dyn std::error::Error>> {
            let entries = self.get_global_entries();
            let stats = self.get_global_stats();

            let export_data = serde_json::json!({
                "coordinator": "global",
                "timestamp": SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis(),
                "stats": {
                    "total_entries": stats.total_entries,
                    "dropped_entries": stats.dropped_entries,
                    "nodes_count": self.nodes.read().unwrap().len(),
                    "entries_by_level": stats.entries_by_level.iter().map(|(k, v)| (format!("{:?}", k), v)).collect::<HashMap<_, _>>()
                },
                "entries": entries.iter().map(|entry| serde_json::json!({
                    "id": entry.id,
                    "node_id": entry.node_id.to_string(),
                    "timestamp": entry.timestamp,
                    "level": format!("{:?}", entry.level),
                    "logger": entry.logger,
                    "message": entry.message,
                    "context": entry.context,
                    "sequence": entry.sequence
                })).collect::<Vec<_>>()
            });

            Ok(serde_json::to_string_pretty(&export_data)?)
        }
    }

    impl Drop for MultiNodeCoordinator {
        fn drop(&mut self) {
            self.stop();
        }
    }
}

#[cfg(test)]
mod distributed_tests {
    use super::distributed::*;
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_node_id_creation() {
        let node = NodeId::new("worker1".to_string(), "pid123".to_string());
        assert_eq!(node.name(), "worker1");
        assert_eq!(node.instance_id(), "pid123");
        assert_eq!(node.to_string(), "worker1:pid123");
    }

    #[test]
    fn test_log_aggregator() {
        let node_id = NodeId::new("test_node".to_string(), "1".to_string());
        let aggregator = LogAggregator::new(node_id.clone(), 100, Duration::from_secs(60));

        let entry = DistributedLogEntry::new(
            node_id,
            LogLevel::Info,
            "test_logger".to_string(),
            "Test message".to_string(),
            HashMap::new(),
        );

        aggregator.add_entry(entry);

        let entries = aggregator.get_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].message, "Test message");

        let stats = aggregator.stats();
        assert_eq!(stats.total_entries, 1);
    }

    #[test]
    fn test_adaptive_rate_limiter() {
        let limiter = AdaptiveRateLimiter::new(10.0, Duration::from_millis(100), 0.1);

        // Should allow initial messages
        assert!(limiter.try_acquire());
        assert!(limiter.try_acquire());

        let stats = limiter.get_stats();
        assert!(stats.current_rate >= 0.0);
        assert_eq!(stats.max_rate, 10.0);
    }

    #[test]
    fn test_distributed_logger() {
        let node_id = NodeId::new("test_node".to_string(), "1".to_string());
        let logger =
            DistributedLogger::new(node_id, "test_logger", 1000, Duration::from_secs(60), 100.0);

        logger.info_adaptive("Test message 1");
        logger.warn_adaptive("Test message 2");

        let entries = logger.get_aggregated_logs();
        assert!(!entries.is_empty()); // At least one message should go through

        let stats = logger.get_aggregation_stats();
        assert!(stats.total_entries >= 1);
    }

    #[test]
    fn test_multi_node_coordinator() {
        let coordinator = MultiNodeCoordinator::new(Duration::from_millis(10));

        let node1_id = NodeId::new("node1".to_string(), "1".to_string());
        let node1_logger = Arc::new(DistributedLogger::new(
            node1_id.clone(),
            "node1_logger",
            100,
            Duration::from_secs(10),
            50.0,
        ));

        coordinator.register_node(node1_id, node1_logger);

        // Start coordination
        coordinator.start();

        // Let it run briefly
        std::thread::sleep(Duration::from_millis(50));

        coordinator.stop();

        let stats = coordinator.get_global_stats();
        // Should have basic structure even if no messages
        // Note: total_entries is u64 so always >= 0, just check it exists
        let _ = stats.total_entries;
    }

    #[test]
    fn test_log_export() {
        let node_id = NodeId::new("export_test".to_string(), "1".to_string());
        let logger = DistributedLogger::new(
            node_id,
            "export_logger",
            100,
            Duration::from_secs(60),
            100.0,
        );

        logger.info_adaptive("Export test message");

        let json_export = logger.export_logs_json().unwrap();
        assert!(json_export.contains("export_test"));
        assert!(json_export.contains("Export test message"));
    }
}
