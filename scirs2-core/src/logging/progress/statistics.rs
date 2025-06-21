//! Progress tracking statistics
//!
//! This module provides statistical tracking for progress operations including
//! throughput analysis, ETA calculation, and performance metrics.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Progress tracking statistics
#[derive(Debug, Clone)]
pub struct ProgressStats {
    /// Number of items processed
    pub processed: u64,
    /// Total number of items to process
    pub total: u64,
    /// Percentage complete (0-100)
    pub percentage: f64,
    /// Items per second
    pub items_per_second: f64,
    /// Estimated time remaining
    pub eta: Duration,
    /// Time elapsed
    pub elapsed: Duration,
    /// A record of recent processing speeds
    pub recent_speeds: VecDeque<f64>,
    /// Highest observed items per second
    pub max_speed: f64,
    /// Time of last update
    pub last_update: Instant,
    /// Number of updates
    pub update_count: u64,
    /// Start time of the tracking
    pub start_time: Instant,
}

impl ProgressStats {
    /// Create new progress statistics
    pub fn new(total: u64) -> Self {
        let now = Instant::now();
        Self {
            processed: 0,
            total,
            percentage: 0.0,
            items_per_second: 0.0,
            eta: Duration::from_secs(0),
            elapsed: Duration::from_secs(0),
            recent_speeds: VecDeque::with_capacity(20),
            max_speed: 0.0,
            last_update: now,
            update_count: 0,
            start_time: now,
        }
    }

    /// Update statistics based on current processed count
    pub fn update(&mut self, processed: u64, now: Instant) {
        let old_processed = self.processed;
        self.processed = processed.min(self.total);

        // Calculate percentage
        if self.total > 0 {
            self.percentage = (self.processed as f64 / self.total as f64) * 100.0;
        } else {
            self.percentage = 0.0;
        }

        // Calculate elapsed time from start
        self.elapsed = now.duration_since(self.start_time);

        // Calculate processing speed
        let time_diff = now.duration_since(self.last_update);
        let items_diff = self.processed.saturating_sub(old_processed);

        if items_diff > 0 && !time_diff.is_zero() {
            let speed = items_diff as f64 / time_diff.as_secs_f64();
            self.recent_speeds.push_back(speed);

            // Keep only the last 20 speed measurements for smoothing
            if self.recent_speeds.len() > 20 {
                self.recent_speeds.pop_front();
            }

            // Calculate average speed from recent measurements
            let avg_speed: f64 =
                self.recent_speeds.iter().sum::<f64>() / self.recent_speeds.len() as f64;
            self.items_per_second = avg_speed;
            self.max_speed = self.max_speed.max(avg_speed);
        } else if self.elapsed.as_secs_f64() > 0.0 && self.processed > 0 {
            // Fallback to overall average if no recent measurements
            self.items_per_second = self.processed as f64 / self.elapsed.as_secs_f64();
        }

        // Calculate ETA
        if self.items_per_second > 0.0 && self.processed < self.total {
            let remaining_items = self.total - self.processed;
            let remaining_seconds = remaining_items as f64 / self.items_per_second;
            self.eta = Duration::from_secs_f64(remaining_seconds.max(0.0));
        } else {
            self.eta = Duration::from_secs(0);
        }

        self.last_update = now;
        self.update_count += 1;
    }

    /// Get the processing rate in items per second
    pub fn rate(&self) -> f64 {
        self.items_per_second
    }

    /// Get the average processing rate since start
    pub fn average_rate(&self) -> f64 {
        if self.elapsed.as_secs_f64() > 0.0 && self.processed > 0 {
            self.processed as f64 / self.elapsed.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get the maximum observed processing rate
    pub fn peak_rate(&self) -> f64 {
        self.max_speed
    }

    /// Check if the processing is complete
    pub fn is_complete(&self) -> bool {
        self.processed >= self.total && self.total > 0
    }

    /// Get remaining items to process
    pub fn remaining(&self) -> u64 {
        self.total.saturating_sub(self.processed)
    }

    /// Get a smoothed ETA based on recent performance
    pub fn smoothed_eta(&self) -> Duration {
        if self.recent_speeds.is_empty() || self.processed >= self.total {
            return Duration::from_secs(0);
        }

        // Use recent speed measurements for more accurate ETA
        let recent_avg: f64 =
            self.recent_speeds.iter().sum::<f64>() / self.recent_speeds.len() as f64;

        if recent_avg > 0.0 {
            let remaining_items = self.total - self.processed;
            let remaining_seconds = remaining_items as f64 / recent_avg;
            Duration::from_secs_f64(remaining_seconds.max(0.0))
        } else {
            self.eta
        }
    }

    /// Get progress efficiency (actual vs expected based on average)
    pub fn efficiency(&self) -> f64 {
        let avg_rate = self.average_rate();
        if avg_rate > 0.0 && self.items_per_second > 0.0 {
            self.items_per_second / avg_rate
        } else {
            1.0
        }
    }
}

/// Format duration in human-readable format
pub fn format_duration(duration: &Duration) -> String {
    let total_secs = duration.as_secs();

    if total_secs < 60 {
        return format!("{}s", total_secs);
    }

    let mins = total_secs / 60;
    let secs = total_secs % 60;

    if mins < 60 {
        return format!("{}m {}s", mins, secs);
    }

    let hours = mins / 60;
    let mins = mins % 60;

    if hours < 24 {
        return format!("{}h {}m {}s", hours, mins, secs);
    }

    let days = hours / 24;
    let hours = hours % 24;

    format!("{}d {}h {}m {}s", days, hours, mins, secs)
}

/// Format processing rate in human-readable format
pub fn format_rate(rate: f64) -> String {
    if rate >= 1000000.0 {
        format!("{:.1}M it/s", rate / 1000000.0)
    } else if rate >= 1000.0 {
        format!("{:.1}k it/s", rate / 1000.0)
    } else {
        format!("{:.1} it/s", rate)
    }
}

/// Format byte count in human-readable format
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_progress_stats_creation() {
        let stats = ProgressStats::new(100);
        assert_eq!(stats.total, 100);
        assert_eq!(stats.processed, 0);
        assert_eq!(stats.percentage, 0.0);
    }

    #[test]
    fn test_progress_stats_update() {
        let mut stats = ProgressStats::new(100);
        let now = Instant::now();

        // Simulate some processing
        thread::sleep(Duration::from_millis(10));
        stats.update(25, now + Duration::from_millis(10));

        assert_eq!(stats.processed, 25);
        assert_eq!(stats.percentage, 25.0);
        assert!(stats.items_per_second > 0.0);
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(&Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(&Duration::from_secs(90)), "1m 30s");
        assert_eq!(format_duration(&Duration::from_secs(3665)), "1h 1m 5s");
    }

    #[test]
    fn test_format_rate() {
        assert_eq!(format_rate(10.5), "10.5 it/s");
        assert_eq!(format_rate(1500.0), "1.5k it/s");
        assert_eq!(format_rate(2500000.0), "2.5M it/s");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(2097152), "2.0 MB");
    }
}
