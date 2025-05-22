//! Memory-efficient operations and views for large arrays.
//!
//! This module provides utilities for working with large arrays efficiently by:
//! - Using chunk-wise processing to reduce memory requirements
//! - Creating memory-efficient views that avoid full data copies
//! - Implementing lazy evaluation and operation fusion for improved performance
//! - Supporting out-of-core processing for data that doesn't fit in RAM
//! - Smart prefetching for improved performance with predictable access patterns

mod adaptive_chunking;
mod adaptive_prefetch;
mod chunked;
mod compressed_memmap;
mod cross_device;
mod cross_file_prefetch;
mod fusion;
mod lazy_array;
mod memmap;
mod memmap_chunks;
mod memmap_slice;
mod out_of_core;
mod pattern_recognition;
mod prefetch;
mod resource_aware;
mod streaming;
mod validation;
mod views;
mod zero_serialization;
mod zerocopy;

pub use adaptive_chunking::{
    AdaptiveChunking, AdaptiveChunkingBuilder, AdaptiveChunkingParams, AdaptiveChunkingResult,
};
pub use adaptive_prefetch::{
    AdaptivePatternTracker, AdaptivePrefetchConfig, AdaptivePrefetchConfigBuilder,
    PatternTrackerFactory, PrefetchStrategy,
};
pub use chunked::{
    chunk_wise_binary_op, chunk_wise_op, chunk_wise_reduce, ChunkedArray, ChunkingStrategy,
    OPTIMAL_CHUNK_SIZE,
};
pub use compressed_memmap::{
    CompressedFileMetadata, CompressedMemMapBuilder, CompressedMemMappedArray, CompressionAlgorithm,
};
pub use cross_device::{
    create_cpu_array, create_cross_device_manager, create_gpu_array, to_best_device,
    CrossDeviceManager, DeviceArray, DeviceBuffer, DeviceMemoryManager, DeviceMemoryPool,
    DeviceStream, DeviceType, KernelParam, ToDevice, ToHost, TransferDirection, TransferEvent,
    TransferMode, TransferOptions, TransferOptionsBuilder,
};
pub use cross_file_prefetch::{
    AccessType, CrossFilePrefetchConfig, CrossFilePrefetchConfigBuilder, CrossFilePrefetchManager,
    CrossFilePrefetchRegistry, DataAccess, DatasetId, DatasetPrefetcher,
};
pub use fusion::{register_fusion, FusedOp, OpFusion};
pub use lazy_array::{evaluate, LazyArray, LazyOp, LazyOpKind};
pub use memmap::{create_mmap, create_temp_mmap, open_mmap, AccessMode, MemoryMappedArray};
#[cfg(feature = "parallel")]
pub use memmap_chunks::MemoryMappedChunksParallel;
pub use memmap_chunks::{ChunkIter, MemoryMappedChunkIter, MemoryMappedChunks};
pub use memmap_slice::{MemoryMappedSlice, MemoryMappedSlicing};
pub use out_of_core::{create_disk_array, load_chunks, DiskBackedArray, OutOfCoreArray};
pub use pattern_recognition::{
    ComplexPattern, Confidence, PatternRecognitionConfig, PatternRecognizer, RecognizedPattern,
};
pub use prefetch::{
    AccessPattern, PrefetchConfig, PrefetchConfigBuilder, PrefetchStats, Prefetching,
    PrefetchingCompressedArray,
};
pub use resource_aware::{
    ResourceAwareConfig, ResourceAwareConfigBuilder, ResourceAwarePrefetcher, ResourceMonitor,
    ResourceSnapshot, ResourceSummary, ResourceType,
};
pub use streaming::{
    create_pipeline, create_stream_processor, Pipeline, PipelineBuilder, PipelineStats,
    StreamConfig, StreamConfigBuilder, StreamMode, StreamProcessor, StreamSource, StreamState,
    StreamStats,
};
pub use views::{diagonal_view, transpose_view, view_as, view_mut_as, ArrayView, ViewMut};
pub use zero_serialization::{ZeroCopySerializable, ZeroCopySerialization};
pub use zerocopy::{ArithmeticOps, BroadcastOps, ZeroCopyOps};

// Re-export commonly used items in a prelude module for convenience
pub mod prelude {
    pub use super::{
        chunk_wise_binary_op,
        chunk_wise_op,
        chunk_wise_reduce,
        create_cpu_array,
        create_cross_device_manager,
        create_gpu_array,
        create_mmap,
        create_temp_mmap,
        evaluate,
        open_mmap,
        to_best_device,
        view_as,
        view_mut_as,
        AccessMode,
        AdaptiveChunking,
        AdaptiveChunkingBuilder,
        // Advanced prefetching
        AdaptivePatternTracker,
        AdaptivePrefetchConfig,
        ArithmeticOps,
        ArrayView,
        BroadcastOps,
        ChunkIter,
        ChunkedArray,
        ComplexPattern,
        CompressedMemMapBuilder,
        CompressionAlgorithm,
        CrossDeviceManager,
        CrossFilePrefetchManager,
        DatasetId,
        DatasetPrefetcher,
        DeviceArray,
        DeviceBuffer,
        DeviceType,
        LazyArray,
        MemoryMappedArray,
        MemoryMappedChunkIter,
        MemoryMappedChunks,
        MemoryMappedSlice,
        MemoryMappedSlicing,
        OutOfCoreArray,
        PatternRecognizer,
        PrefetchConfig,
        PrefetchConfigBuilder,
        PrefetchStrategy,
        Prefetching,
        PrefetchingCompressedArray,
        ResourceAwareConfig,
        ResourceAwarePrefetcher,
        StreamConfig,
        StreamConfigBuilder,
        StreamMode,
        StreamProcessor,
        StreamState,
        ToDevice,
        ToHost,
        TransferMode,
        TransferOptions,
        TransferOptionsBuilder,
        ViewMut,
        ZeroCopyOps,
        ZeroCopySerializable,
        ZeroCopySerialization,
    };

    #[cfg(feature = "parallel")]
    pub use super::MemoryMappedChunksParallel;
}
