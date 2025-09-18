//! Timer functionality for profiling code execution time

use crate::profiling::profiler::Profiler;
use std::time::{Duration, Instant};

/// Timer for measuring code execution time
pub struct Timer {
    /// Name of the operation being timed
    name: String,
    /// Start time
    start_time: Instant,
    /// Whether the timer is currently running
    running: bool,
    /// Whether to automatically report the timing when dropped
    auto_report: bool,
    /// Parent timer name for hierarchical profiling
    parent: Option<String>,
}

impl Timer {
    /// Start a new timer with the given name
    pub fn start(name: &str) -> Self {
        let timer = Self {
            name: name.to_string(),
            start_time: Instant::now(),
            running: true,
            auto_report: true,
            parent: None,
        };
        if let Ok(mut profiler) = Profiler::global().lock() {
            profiler.register_timer_start(&timer);
        }
        timer
    }

    /// Start a new hierarchical timer with a parent
    pub fn start_with_parent(name: &str, parent: &str) -> Self {
        let timer = Self {
            name: name.to_string(),
            start_time: Instant::now(),
            running: true,
            auto_report: true,
            parent: Some(parent.to_string()),
        };
        if let Ok(mut profiler) = Profiler::global().lock() {
            profiler.register_timer_start(&timer);
        }
        timer
    }

    /// Time a function call and return its result
    pub fn time_function<F, R>(name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let timer = Self::start(name);
        let result = f();
        timer.stop();
        result
    }

    /// Time a function call with a parent timer and return its result
    pub fn time_function_with_parent<F, R>(name: &str, parent: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let timer = Self::start_with_parent(name, parent);
        let result = f();
        timer.stop();
        result
    }

    /// Stop the timer and record the elapsed time
    pub fn stop(&self) {
        if !self.running {
            return;
        }

        let elapsed = self.start_time.elapsed();
        if let Ok(mut profiler) = Profiler::global().lock() {
            profiler.register_timer_stop(&self.name, elapsed, self.parent.as_deref());
        }
    }

    /// Get the elapsed time without stopping the timer
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Disable auto-reporting when dropped
    pub fn without_auto_report(mut self) -> Self {
        self.auto_report = false;
        self
    }

    /// Get the name of the timer
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the start time
    pub fn start_time(&self) -> Instant {
        self.start_time
    }

    /// Get the parent timer name
    pub fn parent(&self) -> Option<&str> {
        self.parent.as_deref()
    }

    /// Check if the timer is running
    pub fn is_running(&self) -> bool {
        self.running
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        if self.running && self.auto_report {
            let elapsed = self.start_time.elapsed();
            if let Ok(mut profiler) = Profiler::global().lock() {
                profiler.register_timer_stop(&self.name, elapsed, self.parent.as_deref());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_timer_basic() {
        let timer = Timer::start("test_operation");
        thread::sleep(Duration::from_millis(10));
        timer.stop();

        let elapsed = timer.elapsed();
        assert!(elapsed >= Duration::from_millis(10));
    }
}
