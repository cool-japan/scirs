//! Asynchronous execution and event-based synchronization for GPU operations
//!
//! This module provides comprehensive support for asynchronous GPU operations with
//! event-based synchronization, enabling efficient overlapping of computation and
//! memory transfers.

use crate::gpu::{GpuBuffer, GpuError, GpuKernelHandle};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Type alias for a callback function
type CallbackFn = Box<dyn FnOnce() + Send + 'static>;

/// Type alias for a list of callbacks
type CallbackList = Arc<Mutex<Vec<CallbackFn>>>;

/// Unique identifier for GPU events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventId(u64);

impl EventId {
    /// Create a new unique event ID
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for EventId {
    fn default() -> Self {
        Self::new()
    }
}

/// Unique identifier for GPU streams
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamId(u64);

impl StreamId {
    /// Create a new unique stream ID
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for StreamId {
    fn default() -> Self {
        Self::new()
    }
}

/// Event state for synchronization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventState {
    /// Event has been recorded but not yet completed
    Pending,
    /// Event has completed successfully
    Completed,
    /// Event has failed
    Failed,
    /// Event was cancelled
    Cancelled,
}

/// GPU event for synchronization
pub struct GpuEvent {
    id: EventId,
    state: Arc<Mutex<EventState>>,
    timestamp: Option<Instant>,
    duration: Arc<Mutex<Option<Duration>>>,
    dependencies: Vec<EventId>,
    callbacks: CallbackList,
}

impl GpuEvent {
    /// Create a new GPU event
    pub fn new() -> Self {
        Self {
            id: EventId::new(),
            state: Arc::new(Mutex::new(EventState::Pending)),
            timestamp: Some(Instant::now()),
            duration: Arc::new(Mutex::new(None)),
            dependencies: Vec::new(),
            callbacks: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create a new event with dependencies
    pub fn with_dependencies(dependencies: Vec<EventId>) -> Self {
        Self {
            id: EventId::new(),
            state: Arc::new(Mutex::new(EventState::Pending)),
            timestamp: Some(Instant::now()),
            duration: Arc::new(Mutex::new(None)),
            dependencies,
            callbacks: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Get the event ID
    pub fn id(&self) -> EventId {
        self.id
    }

    /// Get the current state of the event
    pub fn state(&self) -> EventState {
        *self.state.lock().unwrap()
    }

    /// Check if the event has completed
    pub fn is_completed(&self) -> bool {
        self.state() == EventState::Completed
    }

    /// Check if the event has failed
    pub fn is_failed(&self) -> bool {
        self.state() == EventState::Failed
    }

    /// Wait for the event to complete
    pub fn wait(&self) -> Result<(), GpuError> {
        self.wait_timeout(Duration::from_secs(30))
    }

    /// Wait for the event to complete with a timeout
    pub fn wait_timeout(&self, timeout: Duration) -> Result<(), GpuError> {
        let start = Instant::now();
        while start.elapsed() < timeout {
            match self.state() {
                EventState::Completed => return Ok(()),
                EventState::Failed => {
                    return Err(GpuError::KernelExecutionError(
                        "Event execution failed".to_string(),
                    ))
                }
                EventState::Cancelled => {
                    return Err(GpuError::Other("Event was cancelled".to_string()))
                }
                EventState::Pending => {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
        Err(GpuError::Other("Event wait timeout".to_string()))
    }

    /// Get the execution duration if completed
    pub fn duration(&self) -> Option<Duration> {
        *self.duration.lock().unwrap()
    }

    /// Add a callback to be executed when the event completes
    pub fn add_callback<F>(&self, callback: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.callbacks.lock().unwrap().push(Box::new(callback));
    }

    /// Get dependencies
    pub fn dependencies(&self) -> &[EventId] {
        &self.dependencies
    }

    /// Mark the event as completed
    #[allow(dead_code)]
    pub(crate) fn complete(&self) {
        let start_time = self.timestamp.unwrap_or_else(Instant::now);
        let duration = start_time.elapsed();

        *self.duration.lock().unwrap() = Some(duration);
        *self.state.lock().unwrap() = EventState::Completed;

        // Execute callbacks
        let callbacks = std::mem::take(&mut *self.callbacks.lock().unwrap());
        for callback in callbacks {
            callback();
        }
    }

    /// Mark the event as failed
    #[allow(dead_code)]
    pub(crate) fn fail(&self) {
        *self.state.lock().unwrap() = EventState::Failed;
    }

    /// Cancel the event
    #[allow(dead_code)]
    pub(crate) fn cancel(&self) {
        *self.state.lock().unwrap() = EventState::Cancelled;
    }
}

impl Default for GpuEvent {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for GpuEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuEvent")
            .field("id", &self.id)
            .field("state", &self.state)
            .field("timestamp", &self.timestamp)
            .field("duration", &self.duration)
            .field("dependencies", &self.dependencies)
            .field(
                "callbacks",
                &format!("{} callbacks", self.callbacks.lock().unwrap().len()),
            )
            .finish()
    }
}

/// Priority levels for stream operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StreamPriority {
    /// Low priority for background operations
    Low = 0,
    /// Normal priority for regular operations
    Normal = 1,
    /// High priority for critical operations
    High = 2,
}

impl Default for StreamPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// GPU stream for organizing operations
#[derive(Debug)]
pub struct GpuStream {
    id: StreamId,
    priority: StreamPriority,
    events: Arc<Mutex<Vec<Weak<GpuEvent>>>>,
    operations_count: Arc<Mutex<usize>>,
}

impl GpuStream {
    /// Create a new GPU stream
    pub fn new() -> Self {
        Self {
            id: StreamId::new(),
            priority: StreamPriority::Normal,
            events: Arc::new(Mutex::new(Vec::new())),
            operations_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Create a new GPU stream with priority
    pub fn with_priority(priority: StreamPriority) -> Self {
        Self {
            id: StreamId::new(),
            priority,
            events: Arc::new(Mutex::new(Vec::new())),
            operations_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Get the stream ID
    pub fn id(&self) -> StreamId {
        self.id
    }

    /// Get the stream priority
    pub fn priority(&self) -> StreamPriority {
        self.priority
    }

    /// Add an event to this stream
    pub fn add_event(&self, event: &Arc<GpuEvent>) {
        self.events.lock().unwrap().push(Arc::downgrade(event));
        *self.operations_count.lock().unwrap() += 1;
    }

    /// Wait for all operations in this stream to complete
    pub fn synchronize(&self) -> Result<(), GpuError> {
        let events = self.events.lock().unwrap().clone();
        for weak_event in events {
            if let Some(event) = weak_event.upgrade() {
                event.wait()?;
            }
        }
        Ok(())
    }

    /// Get the number of operations in this stream
    pub fn operations_count(&self) -> usize {
        *self.operations_count.lock().unwrap()
    }

    /// Check if the stream is idle (all operations completed)
    pub fn is_idle(&self) -> bool {
        let events = self.events.lock().unwrap();
        events.iter().all(|weak_event| {
            weak_event
                .upgrade()
                .map(|event| event.is_completed())
                .unwrap_or(true)
        })
    }

    /// Clean up completed events
    pub fn cleanup(&self) {
        let mut events = self.events.lock().unwrap();
        events.retain(|weak_event| {
            weak_event
                .upgrade()
                .is_some_and(|event| !event.is_completed())
        });
    }
}

impl Default for GpuStream {
    fn default() -> Self {
        Self::new()
    }
}

/// Error types for asynchronous GPU operations
#[derive(Error, Debug)]
pub enum AsyncGpuError {
    /// Stream not found
    #[error("Stream not found: {0:?}")]
    StreamNotFound(StreamId),

    /// Event not found
    #[error("Event not found: {0:?}")]
    EventNotFound(EventId),

    /// Operation timeout
    #[error("Operation timeout after {0:?}")]
    Timeout(Duration),

    /// Dependency cycle detected
    #[error("Dependency cycle detected in events")]
    DependencyCycle,

    /// Underlying GPU error
    #[error("GPU error: {0}")]
    GpuError(#[from] GpuError),
}

/// Asynchronous GPU operation result
pub type AsyncResult<T> = Result<T, AsyncGpuError>;

/// Manager for asynchronous GPU operations
#[derive(Debug)]
pub struct AsyncGpuManager {
    streams: Arc<Mutex<HashMap<StreamId, Arc<GpuStream>>>>,
    events: Arc<Mutex<HashMap<EventId, Arc<GpuEvent>>>>,
    default_stream: Arc<GpuStream>,
}

impl AsyncGpuManager {
    /// Create a new async GPU manager
    pub fn new() -> Self {
        let default_stream = Arc::new(GpuStream::new());
        let mut streams = HashMap::new();
        streams.insert(default_stream.id(), default_stream.clone());

        Self {
            streams: Arc::new(Mutex::new(streams)),
            events: Arc::new(Mutex::new(HashMap::new())),
            default_stream,
        }
    }

    /// Create a new stream
    pub fn create_stream(&self) -> Arc<GpuStream> {
        self.create_stream_with_priority(StreamPriority::Normal)
    }

    /// Create a new stream with priority
    pub fn create_stream_with_priority(&self, priority: StreamPriority) -> Arc<GpuStream> {
        let stream = Arc::new(GpuStream::with_priority(priority));
        self.streams
            .lock()
            .unwrap()
            .insert(stream.id(), stream.clone());
        stream
    }

    /// Get the default stream
    pub fn default_stream(&self) -> Arc<GpuStream> {
        self.default_stream.clone()
    }

    /// Get a stream by ID
    pub fn get_stream(&self, id: StreamId) -> Option<Arc<GpuStream>> {
        self.streams.lock().unwrap().get(&id).cloned()
    }

    /// Record an event in a stream
    pub fn record_event(&self, stream: &Arc<GpuStream>) -> Arc<GpuEvent> {
        let event = Arc::new(GpuEvent::new());
        stream.add_event(&event);
        self.events
            .lock()
            .unwrap()
            .insert(event.id(), event.clone());
        event
    }

    /// Record an event with dependencies
    pub fn record_event_with_dependencies(
        &self,
        stream: &Arc<GpuStream>,
        dependencies: Vec<EventId>,
    ) -> AsyncResult<Arc<GpuEvent>> {
        // Check for dependency cycles
        self.check_dependency_cycles(&dependencies)?;

        let event = Arc::new(GpuEvent::with_dependencies(dependencies));
        stream.add_event(&event);
        self.events
            .lock()
            .unwrap()
            .insert(event.id(), event.clone());
        Ok(event)
    }

    /// Wait for multiple events
    pub fn wait_for_events(&self, event_ids: &[EventId]) -> AsyncResult<()> {
        for &event_id in event_ids {
            if let Some(event) = self.events.lock().unwrap().get(&event_id).cloned() {
                event.wait()?;
            } else {
                return Err(AsyncGpuError::EventNotFound(event_id));
            }
        }
        Ok(())
    }

    /// Synchronize all streams
    pub fn synchronize_all(&self) -> AsyncResult<()> {
        let streams = self
            .streams
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for stream in streams {
            stream.synchronize()?;
        }
        Ok(())
    }

    /// Clean up completed events and empty streams
    pub fn cleanup(&self) {
        // Clean up streams
        let stream_ids: Vec<_> = self.streams.lock().unwrap().keys().cloned().collect();
        for stream_id in stream_ids {
            if let Some(stream) = self.streams.lock().unwrap().get(&stream_id).cloned() {
                stream.cleanup();
            }
        }

        // Clean up completed events
        let mut events = self.events.lock().unwrap();
        events.retain(|_, event| !event.is_completed() && !event.is_failed());
    }

    /// Get statistics about async operations
    pub fn get_statistics(&self) -> AsyncGpuStatistics {
        let streams = self.streams.lock().unwrap();
        let events = self.events.lock().unwrap();

        let total_streams = streams.len();
        let total_events = events.len();
        let completed_events = events.values().filter(|e| e.is_completed()).count();
        let failed_events = events.values().filter(|e| e.is_failed()).count();
        let pending_events = events
            .values()
            .filter(|e| e.state() == EventState::Pending)
            .count();

        AsyncGpuStatistics {
            total_streams,
            total_events,
            completed_events,
            failed_events,
            pending_events,
        }
    }

    /// Check for dependency cycles in events
    fn check_dependency_cycles(&self, dependencies: &[EventId]) -> AsyncResult<()> {
        let events = self.events.lock().unwrap();

        // Simple cycle detection using DFS
        fn has_cycle(
            event_id: EventId,
            events: &HashMap<EventId, Arc<GpuEvent>>,
            visited: &mut std::collections::HashSet<EventId>,
            rec_stack: &mut std::collections::HashSet<EventId>,
        ) -> bool {
            visited.insert(event_id);
            rec_stack.insert(event_id);

            if let Some(event) = events.get(&event_id) {
                for &dep_id in event.dependencies() {
                    if !visited.contains(&dep_id) {
                        if has_cycle(dep_id, events, visited, rec_stack) {
                            return true;
                        }
                    } else if rec_stack.contains(&dep_id) {
                        return true;
                    }
                }
            }

            rec_stack.remove(&event_id);
            false
        }

        let mut visited = std::collections::HashSet::new();
        let mut rec_stack = std::collections::HashSet::new();

        for &dep_id in dependencies {
            if !visited.contains(&dep_id)
                && has_cycle(dep_id, &events, &mut visited, &mut rec_stack)
            {
                return Err(AsyncGpuError::DependencyCycle);
            }
        }

        Ok(())
    }
}

impl Default for AsyncGpuManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for asynchronous GPU operations
#[derive(Debug, Clone)]
pub struct AsyncGpuStatistics {
    /// Total number of streams
    pub total_streams: usize,
    /// Total number of events
    pub total_events: usize,
    /// Number of completed events
    pub completed_events: usize,
    /// Number of failed events
    pub failed_events: usize,
    /// Number of pending events
    pub pending_events: usize,
}

/// Extension trait for adding async capabilities to GPU operations
pub trait AsyncGpuOps {
    /// Launch a kernel asynchronously
    fn launch_async(
        &self,
        kernel: &GpuKernelHandle,
        work_groups: [u32; 3],
        stream: &Arc<GpuStream>,
    ) -> Arc<GpuEvent>;

    /// Copy data asynchronously
    fn copy_async<T: crate::gpu::GpuDataType>(
        &self,
        src: &GpuBuffer<T>,
        dst: &GpuBuffer<T>,
        stream: &Arc<GpuStream>,
    ) -> Arc<GpuEvent>;

    /// Copy from host asynchronously
    fn copy_from_host_async<T: crate::gpu::GpuDataType>(
        &self,
        src: &[T],
        dst: &GpuBuffer<T>,
        stream: &Arc<GpuStream>,
    ) -> Arc<GpuEvent>;

    /// Copy to host asynchronously
    fn copy_to_host_async<T: crate::gpu::GpuDataType>(
        &self,
        src: &GpuBuffer<T>,
        dst: &mut [T],
        stream: &Arc<GpuStream>,
    ) -> Arc<GpuEvent>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_creation() {
        let event = GpuEvent::new();
        assert_eq!(event.state(), EventState::Pending);
        assert!(!event.is_completed());
        assert!(!event.is_failed());
    }

    #[test]
    fn test_event_completion() {
        let event = GpuEvent::new();
        event.complete();
        assert_eq!(event.state(), EventState::Completed);
        assert!(event.is_completed());
        assert!(!event.is_failed());
        assert!(event.duration().is_some());
    }

    #[test]
    fn test_stream_creation() {
        let stream = GpuStream::new();
        assert_eq!(stream.priority(), StreamPriority::Normal);
        assert_eq!(stream.operations_count(), 0);
        assert!(stream.is_idle());
    }

    #[test]
    fn test_async_manager() {
        let manager = AsyncGpuManager::new();
        let stream = manager.create_stream();
        let event = manager.record_event(&stream);

        assert_eq!(stream.operations_count(), 1);
        assert!(!stream.is_idle());

        event.complete();
        assert!(event.is_completed());
    }

    #[test]
    fn test_event_dependencies() {
        let event1 = GpuEvent::new();
        let event2 = GpuEvent::with_dependencies(vec![event1.id()]);

        assert_eq!(event2.dependencies().len(), 1);
        assert_eq!(event2.dependencies()[0], event1.id());
    }

    #[test]
    fn test_stream_priority() {
        let low_stream = GpuStream::with_priority(StreamPriority::Low);
        let high_stream = GpuStream::with_priority(StreamPriority::High);

        assert_eq!(low_stream.priority(), StreamPriority::Low);
        assert_eq!(high_stream.priority(), StreamPriority::High);
        assert!(high_stream.priority() > low_stream.priority());
    }
}
