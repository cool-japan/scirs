//! Cluster event logging and tracking
//!
//! This module provides event logging capabilities for tracking
//! cluster activities, node changes, and system events.

use std::collections::VecDeque;

use super::types::ClusterEvent;

/// Cluster event logging
#[derive(Debug)]
pub struct ClusterEventLog {
    events: VecDeque<ClusterEvent>,
    max_events: usize,
}

impl Default for ClusterEventLog {
    fn default() -> Self {
        Self::new()
    }
}

impl ClusterEventLog {
    /// Create a new cluster event log
    pub fn new() -> Self {
        Self {
            events: VecDeque::with_capacity(10000usize),
            max_events: 10000,
        }
    }

    /// Create a new event log with a specific capacity
    pub fn with_capacity(max_events: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(max_events),
            max_events,
        }
    }

    /// Log a new event
    pub fn log_event(&mut self, event: ClusterEvent) {
        self.events.push_back(event);

        // Maintain max size
        while self.events.len() > self.max_events {
            self.events.pop_front();
        }
    }

    /// Get the most recent events
    pub fn get_recent_events(&self, count: usize) -> Vec<ClusterEvent> {
        self.events.iter().rev().take(count).cloned().collect()
    }

    /// Get all events
    pub fn get_all_events(&self) -> Vec<ClusterEvent> {
        self.events.iter().cloned().collect()
    }

    /// Get events in chronological order (oldest first)
    pub fn get_events_chronological(&self) -> Vec<ClusterEvent> {
        self.events.iter().cloned().collect()
    }

    /// Get the number of events stored
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Clear all events
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Set the maximum number of events to store
    pub fn set_max_events(&mut self, max_events: usize) {
        self.max_events = max_events;

        // Trim events if necessary
        while self.events.len() > self.max_events {
            self.events.pop_front();
        }
    }

    /// Get the maximum number of events that can be stored
    pub fn get_max_events(&self) -> usize {
        self.max_events
    }

    /// Check if the event log is empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Check if the event log is full
    pub fn is_full(&self) -> bool {
        self.events.len() >= self.max_events
    }

    /// Get events matching a specific filter
    pub fn get_filtered_events<F>(&self, filter: F) -> Vec<ClusterEvent>
    where
        F: Fn(&ClusterEvent) -> bool,
    {
        self.events
            .iter()
            .filter(|event| filter(event))
            .cloned()
            .collect()
    }

    /// Get the oldest event
    pub fn get_oldest_event(&self) -> Option<&ClusterEvent> {
        self.events.front()
    }

    /// Get the newest event
    pub fn get_newest_event(&self) -> Option<&ClusterEvent> {
        self.events.back()
    }
}
