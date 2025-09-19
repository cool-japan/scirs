//! Work queue implementation for the work-stealing scheduler
//!
//! This module contains the WorkQueue struct and its implementation,
//! which manages work items for individual worker threads.

use std::collections::VecDeque;
use std::time::Duration;

use super::core::WorkItem;

/// Work queue for a single worker thread
#[derive(Debug)]
pub struct WorkQueue<T: Clone> {
    /// Double-ended queue for work items
    pub items: VecDeque<WorkItem<T>>,
    /// Number of items processed by this worker
    pub processed_count: usize,
    /// Total execution time for this worker
    pub total_time: Duration,
    /// Average execution time per item
    pub avg_time: Duration,
}

impl<T: Clone> Default for WorkQueue<T> {
    fn default() -> Self {
        Self {
            items: VecDeque::new(),
            processed_count: 0,
            total_time: Duration::ZERO,
            avg_time: Duration::ZERO,
        }
    }
}

impl<T: Clone> WorkQueue<T> {
    /// Add work item to the front of the queue (for local work)
    pub fn push_front(&mut self, item: WorkItem<T>) {
        self.items.push_front(item);
    }

    /// Add work item to the back of the queue (for stolen work)
    #[allow(dead_code)]
    pub fn push_back(&mut self, item: WorkItem<T>) {
        self.items.push_back(item);
    }

    /// Take work from the front (local work)
    pub fn pop_front(&mut self) -> Option<WorkItem<T>> {
        self.items.pop_front()
    }

    /// Steal work from the back (work stealing)
    pub fn steal_back(&mut self) -> Option<WorkItem<T>> {
        if self.items.len() > 1 {
            self.items.pop_back()
        } else {
            None
        }
    }

    /// Update timing statistics
    pub fn update_timing(&mut self, executiontime: Duration) {
        self.processed_count += 1;
        self.total_time += executiontime;
        self.avg_time = self.total_time / self.processed_count as u32;
    }

    /// Get the current load (estimated remaining work time)
    pub fn estimated_load(&self) -> Duration {
        let base_time = if self.avg_time.is_zero() {
            Duration::from_millis(1) // Default estimate
        } else {
            self.avg_time
        };

        self.items
            .iter()
            .map(|item| item.estimated_time.unwrap_or(base_time))
            .sum()
    }
}
