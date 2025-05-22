//! # Logging and Diagnostics
//!
//! This module provides structured logging and diagnostics utilities for scientific computing.
//!
//! ## Features
//!
//! * Structured logging for scientific computing
//! * Progress tracking for long computations
//! * Performance metrics collection
//! * Log filtering and formatting
//!
//! ## Usage
//!
//! ```rust,no_run
//! use scirs2_core::logging::{Logger, LogLevel, ProgressTracker};
//!
//! // Create a logger
//! let logger = Logger::new("matrix_operations");
//!
//! // Log messages at different levels
//! logger.info("Starting matrix multiplication");
//! logger.debug("Using algorithm: Standard");
//!
//! // Create a progress tracker for a long computation
//! let mut progress = ProgressTracker::new("Matrix multiplication", 1000);
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
//! progress.complete();
//!
//! logger.info("Matrix multiplication completed");
//! ```

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::fmt::Display;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Log level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
    pub fn as_str(&self) -> &'static str {
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
