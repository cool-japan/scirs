//! Streaming processing modules
//!
//! This module has been refactored into focused sub-modules for better organization:
//! - `core`: Core streaming infrastructure and pipeline management
//! - `stages`: Processing stages for image transformation and analysis
//! - `video_io`: Video input/output operations and source management
//! - `performance`: Advanced performance monitoring and optimization
//! - `memory`: Memory management and pooling for high-performance streaming

pub mod core;
pub mod stages;
pub mod video_io;
pub mod performance;
pub mod memory;

// Re-export main public API from core module
pub use core::{
    Frame, FrameMetadata, PipelineMetrics, ProcessingStage, StreamPipeline, StreamProcessor,
};

// Re-export processing stages
pub use stages::{
    BlurStage, BufferOperation, EdgeDetectionStage, FeatureDetectionStage, FeatureDetectorType,
    FrameBufferStage, GrayscaleStage, PerspectiveTransformStage, SimdHistogramEqualizationStage,
    SimdNormalizationStage,
};

// Re-export video I/O functionality
pub use video_io::{SimplePerformanceMonitor, VideoSource, VideoStreamReader};

// Re-export performance monitoring
pub use performance::{
    AdaptiveConfig, AdaptivePerformanceMonitor, AutoScalingThreadPoolManager,
    PerformanceSnapshot, StagePerformanceMetrics, SystemResourceMonitor, ThreadPoolConfig,
};

// Re-export memory management
pub use memory::{AdvancedStreamPipeline, AdvancedStreamProcessor, FramePool, MemoryProfiler, MemoryStats};