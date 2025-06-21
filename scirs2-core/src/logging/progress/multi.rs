//! Multi-progress tracking
//!
//! This module provides support for tracking multiple progress operations
//! simultaneously with coordinated display and management.

use super::tracker::EnhancedProgressTracker;
use std::collections::HashMap;

/// Manager for multiple progress bars
pub struct MultiProgress {
    /// Active progress trackers
    trackers: HashMap<usize, EnhancedProgressTracker>,
    /// Next available ID
    next_id: usize,
    /// Whether to show all trackers or just the active one
    show_all: bool,
}

impl MultiProgress {
    /// Create a new multi-progress manager
    pub fn new() -> Self {
        Self {
            trackers: HashMap::new(),
            next_id: 0,
            show_all: true,
        }
    }

    /// Create a new multi-progress manager that shows only the active tracker
    pub fn new_single_view() -> Self {
        Self {
            trackers: HashMap::new(),
            next_id: 0,
            show_all: false,
        }
    }

    /// Add a progress tracker
    pub fn add(&mut self, mut tracker: EnhancedProgressTracker) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        // Hide individual trackers if we're managing display
        if !self.show_all {
            tracker.hide();
        }

        self.trackers.insert(id, tracker);
        id
    }

    /// Get a progress tracker by ID
    pub fn get(&mut self, id: usize) -> Option<&mut EnhancedProgressTracker> {
        self.trackers.get_mut(&id)
    }

    /// Remove a progress tracker
    pub fn remove(&mut self, id: usize) -> Option<EnhancedProgressTracker> {
        self.trackers.remove(&id)
    }

    /// Start a specific tracker
    pub fn start(&mut self, id: usize) {
        if let Some(tracker) = self.trackers.get_mut(&id) {
            tracker.start();
        }
    }

    /// Start all progress trackers
    pub fn start_all(&mut self) {
        for tracker in self.trackers.values_mut() {
            tracker.start();
        }
    }

    /// Update a specific tracker
    pub fn update(&mut self, id: usize, processed: u64) {
        if let Some(tracker) = self.trackers.get_mut(&id) {
            tracker.update(processed);
        }

        if !self.show_all {
            self.render_overview();
        }
    }

    /// Increment a specific tracker
    pub fn increment(&mut self, id: usize, amount: u64) {
        if let Some(tracker) = self.trackers.get_mut(&id) {
            tracker.increment(amount);
        }

        if !self.show_all {
            self.render_overview();
        }
    }

    /// Finish a specific tracker
    pub fn finish(&mut self, id: usize) {
        if let Some(tracker) = self.trackers.get_mut(&id) {
            tracker.finish();
        }

        if !self.show_all {
            self.render_overview();
        }
    }

    /// Finish all trackers
    pub fn finish_all(&mut self) {
        for tracker in self.trackers.values_mut() {
            tracker.finish();
        }
    }

    /// Get the number of active trackers
    pub fn active_count(&self) -> usize {
        self.trackers.len()
    }

    /// Get the overall progress (average of all trackers)
    pub fn overall_progress(&self) -> f64 {
        if self.trackers.is_empty() {
            return 0.0;
        }

        let total_progress: f64 = self
            .trackers
            .values()
            .map(|tracker| tracker.stats().percentage)
            .sum();

        total_progress / self.trackers.len() as f64
    }

    /// Check if all trackers are complete
    pub fn all_complete(&self) -> bool {
        self.trackers.values().all(|tracker| {
            let stats = tracker.stats();
            stats.is_complete()
        })
    }

    /// Get IDs of all trackers
    pub fn tracker_ids(&self) -> Vec<usize> {
        self.trackers.keys().copied().collect()
    }

    /// Clear all completed trackers
    pub fn clear_completed(&mut self) {
        self.trackers.retain(|_, tracker| {
            let stats = tracker.stats();
            !stats.is_complete()
        });
    }

    /// Render an overview of all progress bars (for single view mode)
    fn render_overview(&self) {
        if self.show_all || self.trackers.is_empty() {
            return;
        }

        // Clear previous output
        print!("\r\x1b[K");

        // Show overall progress
        let overall = self.overall_progress();
        print!("Overall: {:.1}% | ", overall);

        // Show active trackers
        let active_trackers: Vec<_> = self
            .trackers
            .iter()
            .filter(|(_, tracker)| {
                let stats = tracker.stats();
                !stats.is_complete()
            })
            .collect();

        match active_trackers.len() {
            1 => {
                let (_, tracker) = active_trackers[0];
                let stats = tracker.stats();
                print!(
                    "{}: {:.1}% ({}/{})",
                    tracker.description, stats.percentage, stats.processed, stats.total
                );
            }
            n if n > 1 => {
                print!("{} active tasks", active_trackers.len());
            }
            _ => {
                print!("All tasks complete");
            }
        }

        use std::io::{self, Write};
        let _ = io::stdout().flush();
    }
}

impl Default for MultiProgress {
    fn default() -> Self {
        Self::new()
    }
}

/// A progress group that manages related progress trackers
pub struct ProgressGroup {
    /// Group name
    name: String,
    /// Multi-progress manager
    multi: MultiProgress,
    /// Group-level statistics
    started_at: Option<std::time::Instant>,
}

impl ProgressGroup {
    /// Create a new progress group
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            multi: MultiProgress::new(),
            started_at: None,
        }
    }

    /// Add a tracker to this group
    pub fn add_tracker(&mut self, tracker: EnhancedProgressTracker) -> usize {
        if self.started_at.is_none() {
            self.started_at = Some(std::time::Instant::now());
        }
        self.multi.add(tracker)
    }

    /// Start all trackers in the group
    pub fn start_all(&mut self) {
        self.started_at = Some(std::time::Instant::now());
        self.multi.start_all();

        println!("Starting group: {}", self.name);
    }

    /// Update a tracker in the group
    pub fn update(&mut self, id: usize, processed: u64) {
        self.multi.update(id, processed);
    }

    /// Finish all trackers in the group
    pub fn finish_all(&mut self) {
        self.multi.finish_all();

        if let Some(started) = self.started_at {
            let duration = started.elapsed();
            println!(
                "Group '{}' completed in {:.2}s",
                self.name,
                duration.as_secs_f64()
            );
        }
    }

    /// Get the group's overall progress
    pub fn overall_progress(&self) -> f64 {
        self.multi.overall_progress()
    }

    /// Check if the entire group is complete
    pub fn is_complete(&self) -> bool {
        self.multi.all_complete()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::progress::tracker::{ProgressBuilder, ProgressStyle};

    #[test]
    fn test_multi_progress_creation() {
        let multi = MultiProgress::new();
        assert_eq!(multi.active_count(), 0);
        assert_eq!(multi.overall_progress(), 0.0);
    }

    #[test]
    fn test_multi_progress_add_tracker() {
        let mut multi = MultiProgress::new();

        let tracker = ProgressBuilder::new("Test", 100)
            .style(ProgressStyle::Bar)
            .build();

        let id = multi.add(tracker);
        assert_eq!(id, 0);
        assert_eq!(multi.active_count(), 1);
    }

    #[test]
    fn test_multi_progress_overall_progress() {
        let mut multi = MultiProgress::new();

        let tracker1 = ProgressBuilder::new("Test 1", 100).build();
        let tracker2 = ProgressBuilder::new("Test 2", 100).build();

        let id1 = multi.add(tracker1);
        let id2 = multi.add(tracker2);

        multi.start_all();
        multi.update(id1, 50); // 50% complete
        multi.update(id2, 25); // 25% complete

        // Overall should be (50 + 25) / 2 = 37.5%
        let overall = multi.overall_progress();
        assert!((overall - 37.5).abs() < 0.1);
    }

    #[test]
    fn test_progress_group() {
        let mut group = ProgressGroup::new("Test Group");

        let tracker = ProgressBuilder::new("Task", 100).build();
        let _id = group.add_tracker(tracker);

        assert_eq!(group.overall_progress(), 0.0);
        assert!(!group.is_complete());
    }
}
