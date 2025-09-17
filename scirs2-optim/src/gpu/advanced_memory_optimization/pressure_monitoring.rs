//! Memory pressure monitoring for advanced GPU memory optimization
//!
//! This module provides real-time memory pressure monitoring and
//! threshold-based alerting for memory optimization decisions.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Memory pressure monitor
#[derive(Debug, Clone)]
pub struct MemoryPressureMonitor {
    /// Current memory pressure level (0.0-1.0)
    pub current_pressure: f32,

    /// Historical pressure readings
    pub pressure_history: VecDeque<PressureReading>,

    /// Pressure thresholds for different actions
    pub thresholds: PressureThresholds,

    /// Monitoring interval
    pub monitor_interval: Duration,

    /// Last monitoring timestamp
    pub last_monitor: Instant,

    /// Maximum history size
    max_history_size: usize,

    /// Trend analysis
    pressure_trend: PressureTrend,

    /// Statistics
    stats: PressureStats,
}

impl MemoryPressureMonitor {
    /// Create a new memory pressure monitor
    pub fn new(pressure_threshold: f32) -> Self {
        Self {
            current_pressure: 0.0,
            pressure_history: VecDeque::new(),
            thresholds: PressureThresholds::new(pressure_threshold),
            monitor_interval: Duration::from_millis(100),
            last_monitor: Instant::now(),
            max_history_size: 1000,
            pressure_trend: PressureTrend::Stable,
            stats: PressureStats::default(),
        }
    }

    /// Update pressure level
    pub fn update_pressure(&mut self, new_pressure: f32) {
        let now = Instant::now();

        // Rate limit updates
        if now.duration_since(self.last_monitor) < self.monitor_interval {
            return;
        }

        // Update current pressure
        let old_pressure = self.current_pressure;
        self.current_pressure = new_pressure.clamp(0.0, 1.0);

        // Record pressure reading
        let reading = PressureReading {
            timestamp: now,
            pressure: self.current_pressure,
            change: self.current_pressure - old_pressure,
        };

        self.pressure_history.push_back(reading);

        // Trim history
        if self.pressure_history.len() > self.max_history_size {
            self.pressure_history.pop_front();
        }

        // Update trend
        self.update_trend();

        // Update statistics
        self.update_stats();

        self.last_monitor = now;
    }

    /// Get current pressure level
    pub fn get_pressure(&self) -> f32 {
        self.current_pressure
    }

    /// Check if pressure exceeds threshold
    pub fn is_high_pressure(&self) -> bool {
        self.current_pressure > self.thresholds.high_pressure
    }

    /// Check if pressure is critical
    pub fn is_critical_pressure(&self) -> bool {
        self.current_pressure > self.thresholds.critical_pressure
    }

    /// Get pressure trend
    pub fn get_trend(&self) -> PressureTrend {
        self.pressure_trend
    }

    /// Get pressure statistics
    pub fn get_statistics(&self) -> &PressureStats {
        &self.stats
    }

    /// Get recommended action based on current pressure
    pub fn get_recommended_action(&self) -> PressureAction {
        match self.current_pressure {
            p if p > self.thresholds.critical_pressure => PressureAction::Emergency,
            p if p > self.thresholds.high_pressure => PressureAction::Aggressive,
            p if p > self.thresholds.moderate_pressure => PressureAction::Moderate,
            p if p > self.thresholds.low_pressure => PressureAction::Conservative,
            _ => PressureAction::None,
        }
    }

    /// Calculate moving average pressure
    pub fn get_moving_average(&self, window: Duration) -> f32 {
        let cutoff = Instant::now() - window;
        let recent_readings: Vec<_> = self.pressure_history
            .iter()
            .filter(|r| r.timestamp >= cutoff)
            .collect();

        if recent_readings.is_empty() {
            self.current_pressure
        } else {
            recent_readings.iter().map(|r| r.pressure).sum::<f32>() / recent_readings.len() as f32
        }
    }

    /// Predict future pressure based on trend
    pub fn predict_future_pressure(&self, duration: Duration) -> f32 {
        match self.pressure_trend {
            PressureTrend::Increasing => {
                let rate = self.calculate_pressure_rate();
                (self.current_pressure + rate * duration.as_secs_f32()).min(1.0)
            }
            PressureTrend::Decreasing => {
                let rate = self.calculate_pressure_rate();
                (self.current_pressure + rate * duration.as_secs_f32()).max(0.0)
            }
            PressureTrend::Stable => self.current_pressure,
        }
    }

    // Private helper methods

    fn update_trend(&mut self) {
        if self.pressure_history.len() < 10 {
            return;
        }

        let recent_window = 5;
        let recent_readings: Vec<_> = self.pressure_history
            .iter()
            .rev()
            .take(recent_window)
            .collect();

        let avg_recent = recent_readings.iter().map(|r| r.pressure).sum::<f32>() / recent_readings.len() as f32;
        let avg_older = self.pressure_history
            .iter()
            .rev()
            .skip(recent_window)
            .take(recent_window)
            .map(|r| r.pressure)
            .sum::<f32>() / recent_window as f32;

        let change = avg_recent - avg_older;
        const THRESHOLD: f32 = 0.05;

        self.pressure_trend = if change > THRESHOLD {
            PressureTrend::Increasing
        } else if change < -THRESHOLD {
            PressureTrend::Decreasing
        } else {
            PressureTrend::Stable
        };
    }

    fn update_stats(&mut self) {
        self.stats.total_readings += 1;

        if self.current_pressure > self.stats.peak_pressure {
            self.stats.peak_pressure = self.current_pressure;
        }

        if self.stats.total_readings == 1 || self.current_pressure < self.stats.min_pressure {
            self.stats.min_pressure = self.current_pressure;
        }

        // Update running average
        self.stats.avg_pressure = (self.stats.avg_pressure * (self.stats.total_readings - 1) as f32 + self.current_pressure) / self.stats.total_readings as f32;

        // Update threshold crossings
        if self.current_pressure > self.thresholds.high_pressure {
            self.stats.high_pressure_events += 1;
        }
        if self.current_pressure > self.thresholds.critical_pressure {
            self.stats.critical_pressure_events += 1;
        }
    }

    fn calculate_pressure_rate(&self) -> f32 {
        if self.pressure_history.len() < 2 {
            return 0.0;
        }

        let recent_readings: Vec<_> = self.pressure_history
            .iter()
            .rev()
            .take(10)
            .collect();

        if recent_readings.len() < 2 {
            return 0.0;
        }

        let total_change = recent_readings.first().unwrap().pressure - recent_readings.last().unwrap().pressure;
        let total_time = recent_readings.first().unwrap().timestamp
            .duration_since(recent_readings.last().unwrap().timestamp)
            .as_secs_f32();

        if total_time > 0.0 {
            total_change / total_time
        } else {
            0.0
        }
    }
}

/// Pressure reading with timestamp
#[derive(Debug, Clone)]
pub struct PressureReading {
    /// When the reading was taken
    pub timestamp: Instant,

    /// Pressure level (0.0-1.0)
    pub pressure: f32,

    /// Change from previous reading
    pub change: f32,
}

/// Pressure thresholds for different severity levels
#[derive(Debug, Clone)]
pub struct PressureThresholds {
    /// Low pressure threshold
    pub low_pressure: f32,

    /// Moderate pressure threshold
    pub moderate_pressure: f32,

    /// High pressure threshold
    pub high_pressure: f32,

    /// Critical pressure threshold
    pub critical_pressure: f32,
}

impl PressureThresholds {
    /// Create new pressure thresholds
    pub fn new(base_threshold: f32) -> Self {
        Self {
            low_pressure: base_threshold * 0.6,
            moderate_pressure: base_threshold * 0.8,
            high_pressure: base_threshold,
            critical_pressure: base_threshold * 1.2,
        }
    }
}

/// Pressure trend analysis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PressureTrend {
    /// Pressure is increasing
    Increasing,

    /// Pressure is decreasing
    Decreasing,

    /// Pressure is stable
    Stable,
}

/// Recommended actions based on pressure level
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PressureAction {
    /// No action needed
    None,

    /// Conservative memory management
    Conservative,

    /// Moderate memory optimization
    Moderate,

    /// Aggressive memory optimization
    Aggressive,

    /// Emergency memory cleanup
    Emergency,
}

/// Pressure monitoring statistics
#[derive(Debug, Clone, Default)]
pub struct PressureStats {
    /// Total number of pressure readings
    pub total_readings: usize,

    /// Average pressure level
    pub avg_pressure: f32,

    /// Peak pressure observed
    pub peak_pressure: f32,

    /// Minimum pressure observed
    pub min_pressure: f32,

    /// Number of high pressure events
    pub high_pressure_events: usize,

    /// Number of critical pressure events
    pub critical_pressure_events: usize,

    /// Time spent in high pressure state
    pub high_pressure_duration: Duration,

    /// Time spent in critical pressure state
    pub critical_pressure_duration: Duration,
}

impl PressureStats {
    /// Calculate pressure stability (lower variance = more stable)
    pub fn stability(&self) -> f32 {
        if self.peak_pressure == self.min_pressure {
            1.0
        } else {
            1.0 - (self.peak_pressure - self.min_pressure) / self.avg_pressure.max(0.01)
        }
    }

    /// Calculate pressure volatility
    pub fn volatility(&self) -> f32 {
        1.0 - self.stability()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pressure_monitor_creation() {
        let monitor = MemoryPressureMonitor::new(0.8);
        assert_eq!(monitor.current_pressure, 0.0);
        assert_eq!(monitor.thresholds.high_pressure, 0.8);
    }

    #[test]
    fn test_pressure_update() {
        let mut monitor = MemoryPressureMonitor::new(0.8);

        monitor.update_pressure(0.5);
        assert_eq!(monitor.current_pressure, 0.5);
        assert!(!monitor.is_high_pressure());

        monitor.update_pressure(0.9);
        assert_eq!(monitor.current_pressure, 0.9);
        assert!(monitor.is_high_pressure());
    }

    #[test]
    fn test_pressure_clamping() {
        let mut monitor = MemoryPressureMonitor::new(0.8);

        monitor.update_pressure(-0.1);
        assert_eq!(monitor.current_pressure, 0.0);

        monitor.update_pressure(1.5);
        assert_eq!(monitor.current_pressure, 1.0);
    }

    #[test]
    fn test_recommended_actions() {
        let monitor = MemoryPressureMonitor::new(0.8);
        let mut monitor = monitor;

        monitor.update_pressure(0.3);
        assert_eq!(monitor.get_recommended_action(), PressureAction::None);

        monitor.update_pressure(0.5);
        assert_eq!(monitor.get_recommended_action(), PressureAction::Conservative);

        monitor.update_pressure(0.7);
        assert_eq!(monitor.get_recommended_action(), PressureAction::Moderate);

        monitor.update_pressure(0.9);
        assert_eq!(monitor.get_recommended_action(), PressureAction::Aggressive);

        monitor.update_pressure(1.0);
        assert_eq!(monitor.get_recommended_action(), PressureAction::Emergency);
    }

    #[test]
    fn test_moving_average() {
        let mut monitor = MemoryPressureMonitor::new(0.8);

        for pressure in [0.2, 0.4, 0.6, 0.8, 1.0] {
            monitor.update_pressure(pressure);
            std::thread::sleep(Duration::from_millis(1));
        }

        let avg = monitor.get_moving_average(Duration::from_secs(1));
        assert!(avg > 0.2 && avg < 1.0);
    }

    #[test]
    fn test_trend_detection() {
        let mut monitor = MemoryPressureMonitor::new(0.8);

        // Simulate increasing trend
        for i in 0..15 {
            monitor.update_pressure(i as f32 * 0.05);
            std::thread::sleep(Duration::from_millis(1));
        }

        // Should detect increasing trend after enough samples
        assert_eq!(monitor.get_trend(), PressureTrend::Increasing);
    }

    #[test]
    fn test_pressure_prediction() {
        let mut monitor = MemoryPressureMonitor::new(0.8);

        monitor.update_pressure(0.5);
        let prediction = monitor.predict_future_pressure(Duration::from_secs(10));

        // With stable trend, prediction should be close to current
        assert!((prediction - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_pressure_thresholds() {
        let thresholds = PressureThresholds::new(0.8);

        assert_eq!(thresholds.low_pressure, 0.48);
        assert_eq!(thresholds.moderate_pressure, 0.64);
        assert_eq!(thresholds.high_pressure, 0.8);
        assert_eq!(thresholds.critical_pressure, 0.96);
    }

    #[test]
    fn test_pressure_stats() {
        let mut monitor = MemoryPressureMonitor::new(0.8);

        monitor.update_pressure(0.2);
        monitor.update_pressure(0.8);
        monitor.update_pressure(0.5);

        let stats = monitor.get_statistics();
        assert_eq!(stats.total_readings, 3);
        assert_eq!(stats.peak_pressure, 0.8);
        assert_eq!(stats.min_pressure, 0.2);
        assert!((stats.avg_pressure - 0.5).abs() < 0.01);
    }
}