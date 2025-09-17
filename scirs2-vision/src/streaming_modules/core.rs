//! Core streaming infrastructure
//!
//! This module provides the fundamental building blocks for streaming video processing,
//! including frame structures, processing pipeline architecture, and performance metrics.

use crate::error::Result;
use crossbeam_channel::{bounded, Receiver};
use ndarray::Array2;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Frame type for streaming processing
#[derive(Clone)]
pub struct Frame {
    /// Frame data as 2D array
    pub data: Array2<f32>,
    /// Frame timestamp
    pub timestamp: Instant,
    /// Frame index
    pub index: usize,
    /// Optional metadata
    pub metadata: Option<FrameMetadata>,
}

/// Frame metadata
#[derive(Clone, Debug)]
pub struct FrameMetadata {
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Frames per second
    pub fps: f32,
    /// Color channels
    pub channels: u8,
}

/// Processing stage trait
pub trait ProcessingStage: Send + 'static {
    /// Process a single frame
    fn process(&mut self, frame: Frame) -> Result<Frame>;

    /// Get stage name for monitoring
    fn name(&self) -> &str;
}

/// Stream processing pipeline
pub struct StreamPipeline {
    pub(crate) stages: Vec<Box<dyn ProcessingStage>>,
    pub(crate) buffer_size: usize,
    pub(crate) num_threads: usize,
    pub(crate) metrics: Arc<Mutex<PipelineMetrics>>,
}

/// Pipeline performance metrics
#[derive(Default, Clone)]
pub struct PipelineMetrics {
    /// Total frames processed
    pub frames_processed: usize,
    /// Average processing time per frame
    pub avg_processing_time: Duration,
    /// Peak processing time
    pub peak_processing_time: Duration,
    /// Frames per second
    pub fps: f32,
    /// Dropped frames
    pub dropped_frames: usize,
}

impl Default for StreamPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamPipeline {
    /// Create a new streaming pipeline
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            buffer_size: 10,
            num_threads: num_cpus::get(),
            metrics: Arc::new(Mutex::new(PipelineMetrics::default())),
        }
    }

    /// Set buffer size for inter-stage communication
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set number of worker threads
    pub fn with_num_threads(mut self, threads: usize) -> Self {
        self.num_threads = threads;
        self
    }

    /// Add a processing stage to the pipeline
    pub fn add_stage<S: ProcessingStage>(mut self, stage: S) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Process a stream of frames
    pub fn process_stream<I>(&mut self, input: I) -> StreamProcessor
    where
        I: Iterator<Item = Frame> + Send + 'static,
    {
        let (tx, rx) = bounded(self.buffer_size);
        let metrics = Arc::clone(&self.metrics);

        // Create pipeline stages with channels
        let mut channels = vec![rx];

        for stage in self.stages.drain(..) {
            let (stage_tx, stage_rx) = bounded(self.buffer_size);
            channels.push(stage_rx);

            let stage_metrics = Arc::clone(&metrics);
            let stagename = stage.name().to_string();
            let prev_rx = channels[channels.len() - 2].clone();

            // Spawn worker thread for this stage
            thread::spawn(move || {
                let mut stage = stage;
                while let Ok(frame) = prev_rx.recv() {
                    let start = Instant::now();

                    match stage.process(frame) {
                        Ok(processed) => {
                            let duration = start.elapsed();

                            // Update metrics
                            if let Ok(mut m) = stage_metrics.lock() {
                                m.frames_processed += 1;
                                m.avg_processing_time = Duration::from_secs_f64(
                                    (m.avg_processing_time.as_secs_f64()
                                        * (m.frames_processed - 1) as f64
                                        + duration.as_secs_f64())
                                        / m.frames_processed as f64,
                                );
                                if duration > m.peak_processing_time {
                                    m.peak_processing_time = duration;
                                }
                            }

                            if stage_tx.send(processed).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            eprintln!("Stage {stagename} error: {e}");
                            if let Ok(mut m) = stage_metrics.lock() {
                                m.dropped_frames += 1;
                            }
                        }
                    }
                }
            });
        }

        let output_rx = channels.pop().unwrap();

        // Input thread
        thread::spawn(move || {
            for frame in input {
                if tx.send(frame).is_err() {
                    break;
                }
            }
        });

        // Return processor with output channel
        StreamProcessor {
            output: output_rx,
            metrics,
        }
    }

    /// Get current pipeline metrics
    pub fn metrics(&self) -> PipelineMetrics {
        self.metrics.lock().unwrap().clone()
    }
}

/// Stream processor handle
pub struct StreamProcessor {
    pub(crate) output: Receiver<Frame>,
    pub(crate) metrics: Arc<Mutex<PipelineMetrics>>,
}

impl StreamProcessor {
    /// Get the next processed frame
    pub fn next(&self) -> Option<Frame> {
        self.output.recv().ok()
    }

    /// Try to get the next frame without blocking
    pub fn try_next(&self) -> Option<Frame> {
        self.output.try_recv().ok()
    }

    /// Get current metrics
    pub fn metrics(&self) -> PipelineMetrics {
        self.metrics.lock().unwrap().clone()
    }
}

impl Iterator for StreamProcessor {
    type Item = Frame;

    fn next(&mut self) -> Option<Self::Item> {
        self.output.recv().ok()
    }
}
