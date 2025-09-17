//! Memory management and pooling for streaming
//!
//! This module provides advanced memory management capabilities including frame pooling,
//! zero-copy processing, and memory profiling for high-performance streaming applications.

use super::core::{Frame, FrameMetadata, PipelineMetrics, ProcessingStage};
use crate::error::Result;
use crossbeam_channel::{bounded, Receiver};
use ndarray::Array2;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

/// Advanced streaming pipeline with memory optimization
pub struct AdvancedStreamPipeline {
    pub(crate) stages: Vec<Box<dyn ProcessingStage>>,
    pub(crate) buffer_size: usize,
    pub(crate) num_threads: usize,
    pub(crate) metrics: Arc<Mutex<PipelineMetrics>>,
    pub(crate) frame_pool: Arc<Mutex<FramePool>>,
    pub(crate) memory_profiler: Arc<Mutex<MemoryProfiler>>,
}

/// Frame pool for memory reuse and zero-copy operations
pub struct FramePool {
    pub(crate) available_frames: Vec<Frame>,
    pub(crate) max_pool_size: usize,
    pub(crate) frame_dimensions: Option<(usize, usize)>,
}

impl FramePool {
    /// Create a new frame pool
    ///
    /// # Returns
    ///
    /// * New frame pool instance
    pub fn new() -> Self {
        Self {
            available_frames: Vec::new(),
            max_pool_size: 50,
            frame_dimensions: None,
        }
    }

    /// Get a frame from the pool or create a new one
    ///
    /// # Arguments
    ///
    /// * `width` - Frame width
    /// * `height` - Frame height
    ///
    /// # Returns
    ///
    /// * Frame ready for use
    pub fn get_frame(&mut self, width: usize, height: usize) -> Frame {
        // Try to reuse an existing frame with matching dimensions
        if let Some(frame_dims) = self.frame_dimensions {
            if frame_dims == (height, width) && !self.available_frames.is_empty() {
                let mut frame = self.available_frames.pop().unwrap();
                // Reset the frame data
                frame.data.fill(0.0);
                frame.timestamp = Instant::now();
                return frame;
            }
        }

        // Create new frame if none available
        Frame {
            data: Array2::zeros((height, width)),
            timestamp: Instant::now(),
            index: 0,
            metadata: Some(FrameMetadata {
                width: width as u32,
                height: height as u32,
                fps: 30.0,
                channels: 1,
            }),
        }
    }

    /// Return a frame to the pool
    ///
    /// # Arguments
    ///
    /// * `frame` - Frame to return to the pool
    pub fn return_frame(&mut self, frame: Frame) {
        if self.available_frames.len() < self.max_pool_size {
            let (height, width) = frame.data.dim();
            self.frame_dimensions = Some((height, width));
            self.available_frames.push(frame);
        }
    }
}

impl Default for FramePool {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage profiler for streaming operations
pub struct MemoryProfiler {
    peak_memory: usize,
    current_memory: usize,
    allocation_count: usize,
    memory_timeline: Vec<(Instant, usize)>,
}

impl MemoryProfiler {
    /// Create a new memory profiler
    ///
    /// # Returns
    ///
    /// * New memory profiler instance
    pub fn new() -> Self {
        Self {
            peak_memory: 0,
            current_memory: 0,
            allocation_count: 0,
            memory_timeline: Vec::new(),
        }
    }

    /// Record a memory allocation
    ///
    /// # Arguments
    ///
    /// * `size` - Size of allocation in bytes
    pub fn record_allocation(&mut self, size: usize) {
        self.current_memory += size;
        self.allocation_count += 1;
        if self.current_memory > self.peak_memory {
            self.peak_memory = self.current_memory;
        }
        self.memory_timeline
            .push((Instant::now(), self.current_memory));
    }

    /// Record a memory deallocation
    ///
    /// # Arguments
    ///
    /// * `size` - Size of deallocation in bytes
    pub fn record_deallocation(&mut self, size: usize) {
        self.current_memory = self.current_memory.saturating_sub(size);
        self.memory_timeline
            .push((Instant::now(), self.current_memory));
    }

    /// Get current memory statistics
    ///
    /// # Returns
    ///
    /// * Memory usage statistics
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            peak_memory: self.peak_memory,
            current_memory: self.current_memory,
            allocation_count: self.allocation_count,
            average_memory: if !self.memory_timeline.is_empty() {
                self.memory_timeline
                    .iter()
                    .map(|(_, mem)| *mem)
                    .sum::<usize>()
                    / self.memory_timeline.len()
            } else {
                0
            },
        }
    }

    /// Reset profiler statistics
    pub fn reset(&mut self) {
        self.peak_memory = 0;
        self.current_memory = 0;
        self.allocation_count = 0;
        self.memory_timeline.clear();
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Peak memory usage observed
    pub peak_memory: usize,
    /// Current memory usage
    pub current_memory: usize,
    /// Total number of allocations
    pub allocation_count: usize,
    /// Average memory usage across all operations
    pub average_memory: usize,
}

impl Default for AdvancedStreamPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl AdvancedStreamPipeline {
    /// Create a new advanced-performance streaming pipeline
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            buffer_size: 10,
            num_threads: num_cpus::get(),
            metrics: Arc::new(Mutex::new(PipelineMetrics::default())),
            frame_pool: Arc::new(Mutex::new(FramePool::new())),
            memory_profiler: Arc::new(Mutex::new(MemoryProfiler::new())),
        }
    }

    /// Set buffer size for inter-stage communication
    ///
    /// # Arguments
    ///
    /// * `size` - Buffer size for channels
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set number of worker threads
    ///
    /// # Arguments
    ///
    /// * `threads` - Number of worker threads
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_num_threads(mut self, threads: usize) -> Self {
        self.num_threads = threads;
        self
    }

    /// Enable zero-copy processing with memory pooling
    pub fn with_zero_copy(self) -> Self {
        // Pre-allocate frame pool for common video sizes
        {
            let mut pool = self.frame_pool.lock().unwrap();

            // Common video resolutions
            let common_sizes = [(480, 640), (720, 1280), (1080, 1920), (240, 320)];

            for &(height, width) in &common_sizes {
                for _ in 0..5 {
                    let frame = Frame {
                        data: Array2::zeros((height, width)),
                        timestamp: Instant::now(),
                        index: 0,
                        metadata: Some(FrameMetadata {
                            width: width as u32,
                            height: height as u32,
                            fps: 30.0,
                            channels: 1,
                        }),
                    };
                    pool.available_frames.push(frame);
                }
            }
        } // Drop the lock here

        self
    }

    /// Add a SIMD-optimized processing stage
    pub fn add_simd_stage<S: ProcessingStage>(mut self, stage: S) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Process stream with advanced-performance optimizations
    pub fn process_advanced_stream<I>(&mut self, input: I) -> AdvancedStreamProcessor
    where
        I: Iterator<Item = Frame> + Send + 'static,
    {
        let (tx, rx) = bounded::<Frame>(self.buffer_size);
        let metrics = Arc::clone(&self.metrics);
        let frame_pool = Arc::clone(&self.frame_pool);
        let memory_profiler = Arc::clone(&self.memory_profiler);

        // Create optimized pipeline with pre-allocated channels
        let mut channels = vec![rx];
        let mut worker_handles = Vec::new();

        for stage in self.stages.drain(..) {
            let (stage_tx, stage_rx) = bounded(self.buffer_size);
            channels.push(stage_rx);

            let stage_metrics = Arc::clone(&metrics);
            let _stage_frame_pool = Arc::clone(&frame_pool);
            let stage_memory_profiler = Arc::clone(&memory_profiler);
            let stagename = stage.name().to_string();
            let prev_rx = channels[channels.len() - 2].clone();

            // Spawn optimized worker thread
            let handle = thread::spawn(move || {
                let mut stage = stage;
                let _local_frame_buffer: Vec<Frame> = Vec::with_capacity(10);

                while let Ok(frame) = prev_rx.recv() {
                    let start = Instant::now();
                    let frame_size = frame.data.len() * std::mem::size_of::<f32>();

                    // Record memory usage
                    if let Ok(mut profiler) = stage_memory_profiler.lock() {
                        profiler.record_allocation(frame_size);
                    }

                    match stage.process(frame) {
                        Ok(processed) => {
                            let duration = start.elapsed();

                            // Update metrics with lock optimization
                            if let Ok(mut m) = stage_metrics.try_lock() {
                                m.frames_processed += 1;
                                m.avg_processing_time = std::time::Duration::from_secs_f64(
                                    (m.avg_processing_time.as_secs_f64()
                                        * (m.frames_processed - 1) as f64
                                        + duration.as_secs_f64())
                                        / m.frames_processed as f64,
                                );
                                if duration > m.peak_processing_time {
                                    m.peak_processing_time = duration;
                                }

                                // Calculate FPS
                                let fps = (1.0 / duration.as_secs_f64()) as f32;
                                m.fps = m.fps * 0.9 + fps * 0.1; // Smooth FPS calculation
                            }

                            if stage_tx.send(processed).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            eprintln!("Stage {stagename} error: {e}");
                            if let Ok(mut m) = stage_metrics.try_lock() {
                                m.dropped_frames += 1;
                            }
                        }
                    }

                    // Record memory deallocation
                    if let Ok(mut profiler) = stage_memory_profiler.lock() {
                        profiler.record_deallocation(frame_size);
                    }
                }
            });

            worker_handles.push(handle);
        }

        let output_rx = channels.pop().unwrap();

        // Optimized input thread with batching
        thread::spawn(move || {
            let mut frame_batch = Vec::with_capacity(4);

            for frame in input {
                frame_batch.push(frame);

                // Process in small batches for better cache locality
                if frame_batch.len() >= 4 {
                    for frame in frame_batch.drain(..) {
                        if tx.send(frame).is_err() {
                            return;
                        }
                    }
                }
            }

            // Send remaining frames
            for frame in frame_batch {
                if tx.send(frame).is_err() {
                    break;
                }
            }
        });

        AdvancedStreamProcessor {
            output: output_rx,
            metrics,
            frame_pool,
            memory_profiler,
            worker_handles,
        }
    }

    /// Get current memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        self.memory_profiler.lock().unwrap().get_stats()
    }

    /// Reset memory profiler statistics
    pub fn reset_memory_stats(&self) {
        self.memory_profiler.lock().unwrap().reset();
    }

    /// Get current pipeline metrics
    pub fn metrics(&self) -> PipelineMetrics {
        self.metrics.lock().unwrap().clone()
    }
}

/// Advanced-high performance stream processor
pub struct AdvancedStreamProcessor {
    pub(crate) output: Receiver<Frame>,
    pub(crate) metrics: Arc<Mutex<PipelineMetrics>>,
    pub(crate) frame_pool: Arc<Mutex<FramePool>>,
    pub(crate) memory_profiler: Arc<Mutex<MemoryProfiler>>,
    #[allow(dead_code)]
    pub(crate) worker_handles: Vec<thread::JoinHandle<()>>,
}

impl AdvancedStreamProcessor {
    /// Get next frame with zero-copy optimization
    pub fn next_zero_copy(&self) -> Option<Frame> {
        self.output.recv().ok()
    }

    /// Try to get the next frame without blocking
    pub fn try_next(&self) -> Option<Frame> {
        self.output.try_recv().ok()
    }

    /// Get batch of frames for efficient processing
    ///
    /// # Arguments
    ///
    /// * `batchsize` - Number of frames to retrieve
    ///
    /// # Returns
    ///
    /// * Vector of frames
    pub fn next_batch(&self, batchsize: usize) -> Vec<Frame> {
        let mut batch = Vec::with_capacity(batchsize);

        for _ in 0..batchsize {
            if let Some(frame) = self.try_next() {
                batch.push(frame);
            } else {
                break;
            }
        }

        batch
    }

    /// Return frame to memory pool
    ///
    /// # Arguments
    ///
    /// * `frame` - Frame to return to the pool
    pub fn return_frame(&self, frame: Frame) {
        if let Ok(mut pool) = self.frame_pool.lock() {
            pool.return_frame(frame);
        }
    }

    /// Get current metrics
    pub fn metrics(&self) -> PipelineMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Get current memory statistics
    pub fn memory_stats(&self) -> MemoryStats {
        self.memory_profiler.lock().unwrap().get_stats()
    }

    /// Reset memory profiler statistics
    pub fn reset_memory_stats(&self) {
        self.memory_profiler.lock().unwrap().reset();
    }
}

impl Iterator for AdvancedStreamProcessor {
    type Item = Frame;

    fn next(&mut self) -> Option<Self::Item> {
        self.output.recv().ok()
    }
}
