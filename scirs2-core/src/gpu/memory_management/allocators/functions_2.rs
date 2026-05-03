//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{BuddyAllocator, CompactionAllocator, HybridAllocator, SlabAllocator};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_buddy_allocator_creation() {
        let result = BuddyAllocator::new(1024, 64);
        assert!(result.is_ok());
        let allocator = result.expect("Failed to create allocator");
        assert_eq!(allocator.available_memory(), 1024);
        assert_eq!(allocator.current_usage(), 0);
    }
    #[test]
    fn test_buddy_allocator_invalid_sizes() {
        let result = BuddyAllocator::new(1024, 63);
        assert!(result.is_err());
        let result = BuddyAllocator::new(1000, 64);
        assert!(result.is_err());
        let result = BuddyAllocator::new(64, 128);
        assert!(result.is_err());
    }
    #[test]
    fn test_buddy_allocator_allocation_deallocation() {
        let allocator = BuddyAllocator::new(4096, 64).expect("Failed to create allocator");
        let offset1 = allocator.allocate(128).expect("Failed to allocate");
        assert_eq!(allocator.current_usage(), 128);
        let offset2 = allocator.allocate(256).expect("Failed to allocate");
        assert_eq!(allocator.current_usage(), 128 + 256);
        allocator.deallocate(offset1).expect("Failed to deallocate");
        assert_eq!(allocator.current_usage(), 256);
        allocator.deallocate(offset2).expect("Failed to deallocate");
        assert_eq!(allocator.current_usage(), 0);
    }
    #[test]
    fn test_buddy_allocator_coalescing() {
        let allocator = BuddyAllocator::new(4096, 64).expect("Failed to create allocator");
        let offset1 = allocator.allocate(128).expect("Failed to allocate");
        let offset2 = allocator.allocate(128).expect("Failed to allocate");
        let stats_before = allocator.statistics();
        let merges_before = stats_before.total_merges;
        allocator.deallocate(offset1).expect("Failed to deallocate");
        allocator.deallocate(offset2).expect("Failed to deallocate");
        let stats_after = allocator.statistics();
        assert!(stats_after.total_merges >= merges_before);
    }
    #[test]
    fn test_buddy_allocator_statistics() {
        let allocator = BuddyAllocator::new(4096, 64).expect("Failed to create allocator");
        let offset = allocator.allocate(256).expect("Failed to allocate");
        let stats = allocator.statistics();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.total_deallocations, 0);
        assert_eq!(stats.allocated_blocks, 1);
        allocator.deallocate(offset).expect("Failed to deallocate");
        let stats = allocator.statistics();
        assert_eq!(stats.total_deallocations, 1);
        assert_eq!(stats.allocated_blocks, 0);
    }
    #[test]
    fn test_buddy_allocator_out_of_memory() {
        let allocator = BuddyAllocator::new(1024, 64).expect("Failed to create allocator");
        let _offset = allocator.allocate(1024).expect("Failed to allocate");
        let result = allocator.allocate(64);
        assert!(result.is_err());
    }
    #[test]
    fn test_slab_allocator_creation() {
        let allocator = SlabAllocator::new();
        let stats = allocator.statistics();
        assert_eq!(stats.total_allocations, 0);
        assert_eq!(stats.current_allocated_bytes, 0);
    }
    #[test]
    fn test_slab_allocator_allocation_deallocation() {
        let allocator = SlabAllocator::new();
        let offset1 = allocator.allocate(128).expect("Failed to allocate");
        assert!(allocator.current_usage() > 0);
        let offset2 = allocator.allocate(512).expect("Failed to allocate");
        let stats = allocator.statistics();
        assert_eq!(stats.total_allocations, 2);
        allocator.deallocate(offset1).expect("Failed to deallocate");
        allocator.deallocate(offset2).expect("Failed to deallocate");
        let stats = allocator.statistics();
        assert_eq!(stats.total_deallocations, 2);
        assert_eq!(stats.current_allocated_bytes, 0);
    }
    #[test]
    fn test_slab_allocator_per_size_tracking() {
        let allocator = SlabAllocator::with_sizes(vec![64, 256, 1024]);
        allocator.allocate(60).expect("Failed to allocate");
        allocator.allocate(200).expect("Failed to allocate");
        allocator.allocate(200).expect("Failed to allocate");
        let stats = allocator.statistics();
        assert_eq!(stats.total_allocations, 3);
        let size_256_stats = stats
            .per_size_stats
            .iter()
            .find(|s| s.slab_size == 256)
            .expect("256B slab not found");
        assert_eq!(size_256_stats.allocated_slabs, 2);
    }
    #[test]
    fn test_slab_allocator_zero_fragmentation() {
        let allocator = SlabAllocator::new();
        for _ in 0..10 {
            let offset = allocator.allocate(256).expect("Failed to allocate");
            allocator.deallocate(offset).expect("Failed to deallocate");
        }
        let stats = allocator.statistics();
        assert_eq!(stats.fragmentation_ratio, 0.0);
    }
    #[test]
    fn test_slab_allocator_invalid_size() {
        let allocator = SlabAllocator::with_sizes(vec![64, 256]);
        let result = allocator.allocate(1024);
        assert!(result.is_err());
    }
    #[test]
    fn test_compaction_allocator_creation() {
        let allocator = CompactionAllocator::new(10240, 0.3);
        assert_eq!(allocator.current_usage(), 0);
        assert_eq!(allocator.available_memory(), 10240);
    }
    #[test]
    fn test_compaction_allocator_allocation_deallocation() {
        let allocator = CompactionAllocator::new(10240, 0.3);
        let offset1 = allocator.allocate(512, true).expect("Failed to allocate");
        assert_eq!(allocator.current_usage(), 512);
        let offset2 = allocator.allocate(256, true).expect("Failed to allocate");
        assert_eq!(allocator.current_usage(), 768);
        allocator.deallocate(offset1).expect("Failed to deallocate");
        assert_eq!(allocator.current_usage(), 256);
        allocator.deallocate(offset2).expect("Failed to deallocate");
        assert_eq!(allocator.current_usage(), 0);
    }
    #[test]
    fn test_compaction_allocator_fragmentation() {
        let allocator = CompactionAllocator::new(10240, 0.3);
        let offset1 = allocator.allocate(1024, true).expect("Failed to allocate");
        let offset2 = allocator.allocate(1024, true).expect("Failed to allocate");
        let offset3 = allocator.allocate(1024, true).expect("Failed to allocate");
        allocator.deallocate(offset2).expect("Failed to deallocate");
        let stats = allocator.statistics();
        assert!(stats.free_region_count >= 1);
    }
    #[test]
    fn test_compaction_allocator_compaction() {
        let allocator = CompactionAllocator::new(10240, 0.3);
        let _offset1 = allocator.allocate(512, true).expect("Failed to allocate");
        let offset2 = allocator.allocate(512, true).expect("Failed to allocate");
        let _offset3 = allocator.allocate(512, true).expect("Failed to allocate");
        allocator.deallocate(offset2).expect("Failed to deallocate");
        let _relocated = allocator.compact().expect("Failed to compact");
        let stats = allocator.statistics();
        assert!(stats.total_compactions >= 1);
    }
    #[test]
    fn test_compaction_allocator_non_relocatable() {
        let allocator = CompactionAllocator::new(10240, 0.3);
        let _offset = allocator.allocate(512, false).expect("Failed to allocate");
        let relocated = allocator.compact().expect("Failed to compact");
        assert_eq!(relocated, 0);
    }
    #[test]
    fn test_compaction_allocator_statistics() {
        let allocator = CompactionAllocator::new(10240, 0.3);
        let offset = allocator.allocate(512, true).expect("Failed to allocate");
        let stats = allocator.statistics();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.current_allocated_bytes, 512);
        assert_eq!(stats.total_memory, 10240);
        allocator.deallocate(offset).expect("Failed to deallocate");
        let stats = allocator.statistics();
        assert_eq!(stats.total_deallocations, 1);
    }
    #[test]
    fn test_hybrid_allocator_creation() {
        let allocator = HybridAllocator::new();
        assert!(allocator.is_ok());
        let allocator = allocator.expect("Failed to create allocator");
        assert_eq!(allocator.current_usage(), 0);
    }
    #[test]
    fn test_hybrid_allocator_small_allocation() {
        let allocator = HybridAllocator::new().expect("Failed to create allocator");
        let offset = allocator.allocate(256).expect("Failed to allocate");
        let stats = allocator.statistics();
        assert_eq!(stats.small_allocations, 1);
        assert_eq!(stats.medium_allocations, 0);
        assert_eq!(stats.large_allocations, 0);
        allocator
            .deallocate(offset, 256)
            .expect("Failed to deallocate");
    }
    #[test]
    fn test_hybrid_allocator_medium_allocation() {
        let allocator = HybridAllocator::new().expect("Failed to create allocator");
        let size = 128 * 1024;
        let offset = allocator.allocate(size).expect("Failed to allocate");
        let stats = allocator.statistics();
        assert_eq!(stats.small_allocations, 0);
        assert_eq!(stats.medium_allocations, 1);
        assert_eq!(stats.large_allocations, 0);
        allocator
            .deallocate(offset, size)
            .expect("Failed to deallocate");
    }
    #[test]
    fn test_hybrid_allocator_large_allocation() {
        let allocator = HybridAllocator::new().expect("Failed to create allocator");
        let size = 32 * 1024 * 1024;
        let offset = allocator.allocate(size).expect("Failed to allocate");
        let stats = allocator.statistics();
        assert_eq!(stats.small_allocations, 0);
        assert_eq!(stats.medium_allocations, 0);
        assert_eq!(stats.large_allocations, 1);
        allocator
            .deallocate(offset, size)
            .expect("Failed to deallocate");
    }
    #[test]
    fn test_hybrid_allocator_mixed_allocations() {
        let allocator = HybridAllocator::new().expect("Failed to create allocator");
        let small = allocator.allocate(128).expect("Failed to allocate");
        let medium = allocator.allocate(256 * 1024).expect("Failed to allocate");
        let large = allocator
            .allocate(20 * 1024 * 1024)
            .expect("Failed to allocate");
        let stats = allocator.statistics();
        assert_eq!(stats.small_allocations, 1);
        assert_eq!(stats.medium_allocations, 1);
        assert_eq!(stats.large_allocations, 1);
        allocator
            .deallocate(small, 128)
            .expect("Failed to deallocate");
        allocator
            .deallocate(medium, 256 * 1024)
            .expect("Failed to deallocate");
        allocator
            .deallocate(large, 20 * 1024 * 1024)
            .expect("Failed to deallocate");
        let stats = allocator.statistics();
        assert_eq!(stats.total_deallocations, 3);
    }
    #[test]
    fn test_hybrid_allocator_statistics() {
        let allocator = HybridAllocator::new().expect("Failed to create allocator");
        allocator.allocate(100).expect("Failed to allocate");
        allocator.allocate(100000).expect("Failed to allocate");
        let stats = allocator.statistics();
        assert!(stats.total_allocations >= 2);
        assert!(stats.slab_statistics.current_allocated_bytes > 0);
    }
    #[test]
    fn test_benchmark_buddy_allocator() {
        use crate::gpu::memory_management::benchmarks::*;
        let config = BenchmarkConfig {
            num_iterations: 100,
            allocation_sizes: vec![64, 256, 1024],
            pattern: AllocationPattern::Sequential,
        };
        let result = benchmark_buddy_allocator(&config);
        assert!(result.is_ok());
        let result = result.expect("Benchmark failed");
        assert_eq!(result.allocator_name, "BuddyAllocator");
        assert!(result.total_time_ns > 0);
    }
    #[test]
    fn test_benchmark_slab_allocator() {
        use crate::gpu::memory_management::benchmarks::*;
        let config = BenchmarkConfig {
            num_iterations: 100,
            allocation_sizes: vec![64, 256, 1024],
            pattern: AllocationPattern::Sequential,
        };
        let result = benchmark_slab_allocator(&config);
        assert!(result.is_ok());
        let result = result.expect("Benchmark failed");
        assert_eq!(result.allocator_name, "SlabAllocator");
        assert!(result.total_time_ns > 0);
    }
    #[test]
    fn test_benchmark_hybrid_allocator() {
        use crate::gpu::memory_management::benchmarks::*;
        let config = BenchmarkConfig {
            num_iterations: 100,
            allocation_sizes: vec![64, 256, 1024],
            pattern: AllocationPattern::Sequential,
        };
        let result = benchmark_hybrid_allocator(&config);
        assert!(result.is_ok());
        let result = result.expect("Benchmark failed");
        assert_eq!(result.allocator_name, "HybridAllocator");
        assert!(result.total_time_ns > 0);
    }
    #[test]
    fn test_fragmentation_reduction() {
        let buddy = BuddyAllocator::new(8192, 64).expect("Failed to create allocator");
        let mut allocations = Vec::new();
        for _ in 0..10 {
            if let Ok(offset) = buddy.allocate(128) {
                allocations.push(offset);
            }
        }
        for offset in allocations {
            let _ = buddy.deallocate(offset);
        }
        let stats = buddy.statistics();
        assert!(stats.fragmentation_ratio < 0.5);
    }
}
