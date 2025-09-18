//! Main profiler implementation for collecting performance metrics

use crate::profiling::entries::{MemoryEntry, TimingEntry};
use crate::profiling::memory::MemoryTracker;
use crate::profiling::timer::Timer;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Profiler for collecting performance metrics
#[derive(Debug)]
pub struct Profiler {
    /// Timing measurements
    timings: HashMap<String, TimingEntry>,
    /// Memory measurements
    memory: HashMap<String, MemoryEntry>,
    /// Currently active timers
    active_timers: HashMap<String, Instant>,
    /// Whether the profiler is currently running
    running: bool,
}

impl Profiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            timings: HashMap::new(),
            memory: HashMap::new(),
            active_timers: HashMap::new(),
            running: false,
        }
    }

    /// Get the global profiler instance
    pub fn global() -> &'static Mutex<Profiler> {
        static GLOBAL_PROFILER: once_cell::sync::Lazy<Mutex<Profiler>> =
            once_cell::sync::Lazy::new(|| Mutex::new(Profiler::new()));
        &GLOBAL_PROFILER
    }

    /// Start the profiler
    pub fn start(&mut self) {
        self.running = true;
        self.timings.clear();
        self.memory.clear();
        self.active_timers.clear();
    }

    /// Stop the profiler
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Reset the profiler
    pub fn reset(&mut self) {
        self.timings.clear();
        self.memory.clear();
        self.active_timers.clear();
    }

    /// Register the start of a timer
    pub fn register_timer_start(&mut self, timer: &Timer) {
        if !self.running {
            return;
        }

        self.active_timers
            .insert(timer.name().to_string(), timer.start_time());

        // Register the parent-child relationship
        if let Some(parent) = timer.parent() {
            if let Some(entry) = self.timings.get_mut(parent) {
                entry.add_child(timer.name());
            }
        }
    }

    /// Register the stop of a timer
    pub fn register_timer_stop(&mut self, name: &str, duration: Duration, parent: Option<&str>) {
        if !self.running {
            return;
        }

        // Remove from active timers
        self.active_timers.remove(name);

        // Update the timing entry
        match self.timings.get_mut(name) {
            Some(entry) => {
                entry.add_measurement(duration);
            }
            None => {
                let entry = TimingEntry::new(duration, parent);
                self.timings.insert(name.to_string(), entry);
            }
        }

        // Register the parent-child relationship
        if let Some(parent) = parent {
            if let Some(entry) = self.timings.get_mut(parent) {
                entry.add_child(name);
            }
        }
    }

    /// Register the start of a memory tracker
    pub fn register_memory_tracker_start(&mut self, _tracker: &MemoryTracker) {
        if !self.running {
            // Nothing to do at start, just ensure the method exists for symmetry
        }
    }

    /// Register the stop of a memory tracker
    pub fn register_memory_tracker_stop(&mut self, name: &str, delta: usize) {
        if !self.running {
            return;
        }

        // Update the memory entry
        match self.memory.get_mut(name) {
            Some(entry) => {
                entry.add_measurement(delta);
            }
            None => {
                let entry = MemoryEntry::new(delta);
                self.memory.insert(name.to_string(), entry);
            }
        }
    }

    /// Print a report of the profiling results
    pub fn print_report(&self) {
        if self.timings.is_empty() && self.memory.is_empty() {
            println!("No profiling data collected.");
            return;
        }

        if !self.timings.is_empty() {
            println!("\n=== Timing Report ===");
            println!(
                "{:<30} {:<10} {:<15} {:<15} {:<15}",
                "Operation", "Calls", "Total (ms)", "Average (ms)", "Max (ms)"
            );
            println!("{}", "-".repeat(90));

            // Sort by total duration
            let mut entries: Vec<(&String, &TimingEntry)> = self.timings.iter().collect();
            entries.sort_by(|a, b| b.1.total_duration().cmp(&a.1.total_duration()));

            for (name, entry) in entries {
                println!(
                    "{:<30} {:<10} {:<15.2} {:<15.2} {:<15.2}",
                    name,
                    entry.calls(),
                    entry.total_duration().as_secs_f64() * 1000.0,
                    entry.average_duration().as_secs_f64() * 1000.0,
                    entry.max_duration().as_secs_f64() * 1000.0
                );
            }
        }

        if !self.memory.is_empty() {
            println!("\n=== Memory Report ===");
            println!(
                "{:<30} {:<10} {:<15} {:<15}",
                "Operation", "Counts", "Total (KB)", "Max (KB)"
            );
            println!("{}", "-".repeat(75));

            // Sort by total memory delta
            let mut entries: Vec<(&String, &MemoryEntry)> = self.memory.iter().collect();
            entries.sort_by(|a, b| b.1.total_delta().abs().cmp(&a.1.total_delta().abs()));

            for (name, entry) in entries {
                println!(
                    "{:<30} {:<10} {:<15.2} {:<15.2}",
                    name,
                    entry.allocations(),
                    entry.total_delta() as f64 / 1024.0,
                    entry.max_delta() as f64 / 1024.0
                );
            }
        }
    }

    /// Get a report of the profiling results as a string
    pub fn get_report(&self) -> String {
        use std::fmt::Write;
        let mut report = String::new();

        if self.timings.is_empty() && self.memory.is_empty() {
            writeln!(report, "No profiling data collected.").unwrap();
            return report;
        }

        if !self.timings.is_empty() {
            writeln!(report, "\n=== Timing Report ===").unwrap();
            writeln!(
                report,
                "{:<30} {:<10} {:<15} {:<15} {:<15}",
                "Operation", "Calls", "Total (ms)", "Average (ms)", "Max (ms)"
            )
            .unwrap();
            writeln!(report, "{}", "-".repeat(90)).unwrap();

            // Sort by total duration
            let mut entries: Vec<(&String, &TimingEntry)> = self.timings.iter().collect();
            entries.sort_by(|a, b| b.1.total_duration().cmp(&a.1.total_duration()));

            for (name, entry) in entries {
                writeln!(
                    report,
                    "{:<30} {:<10} {:<15.2} {:<15.2} {:<15.2}",
                    name,
                    entry.calls(),
                    entry.total_duration().as_secs_f64() * 1000.0,
                    entry.average_duration().as_secs_f64() * 1000.0,
                    entry.max_duration().as_secs_f64() * 1000.0
                )
                .unwrap();
            }
        }

        if !self.memory.is_empty() {
            writeln!(report, "\n=== Memory Report ===").unwrap();
            writeln!(
                report,
                "{:<30} {:<10} {:<15} {:<15}",
                "Operation", "Counts", "Total (KB)", "Max (KB)"
            )
            .unwrap();
            writeln!(report, "{}", "-".repeat(75)).unwrap();

            // Sort by total memory delta
            let mut entries: Vec<(&String, &MemoryEntry)> = self.memory.iter().collect();
            entries.sort_by(|a, b| b.1.total_delta().abs().cmp(&a.1.total_delta().abs()));

            for (name, entry) in entries {
                writeln!(
                    report,
                    "{:<30} {:<10} {:<15.2} {:<15.2}",
                    name,
                    entry.allocations(),
                    entry.total_delta() as f64 / 1024.0,
                    entry.max_delta() as f64 / 1024.0
                )
                .unwrap();
            }
        }

        report
    }

    /// Get timing statistics for a specific operation
    pub fn get_timing_stats(&self, name: &str) -> Option<(usize, Duration, Duration, Duration)> {
        self.timings.get(name).map(|entry| {
            (
                entry.calls(),
                entry.total_duration(),
                entry.average_duration(),
                entry.max_duration(),
            )
        })
    }

    /// Get memory statistics for a specific operation
    pub fn get_memory_stats(&self, name: &str) -> Option<(usize, isize, usize)> {
        self.memory
            .get(name)
            .map(|entry| (entry.allocations(), entry.total_delta(), entry.max_delta()))
    }

    /// Check if the profiler is running
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Get access to timing data
    pub fn timings(&self) -> &HashMap<String, TimingEntry> {
        &self.timings
    }

    /// Get access to memory data
    pub fn memory(&self) -> &HashMap<String, MemoryEntry> {
        &self.memory
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}
