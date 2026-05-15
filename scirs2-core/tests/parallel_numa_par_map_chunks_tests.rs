//! Integration tests for `par_map_chunks` — NUMA-locality chunk map.
//!
//! These tests only compile and run when the `parallel` feature is enabled.
#![cfg(feature = "parallel")]

use scirs2_core::par_map_chunks;

/// Basic correctness: doubling each element in parallel.
#[test]
fn par_map_chunks_typed_result_correctness() {
    let input: Vec<i32> = (0..100).collect();
    let result = par_map_chunks(&input, 10, |chunk| chunk.iter().map(|&x| x * 2).collect());
    let expected: Vec<i32> = (0..100).map(|x| x * 2).collect();
    assert_eq!(result, expected);
}

/// Chunk concatenation must preserve element order across chunks.
#[test]
fn par_map_chunks_preserves_order() {
    let input: Vec<usize> = (0..50).collect();
    let result = par_map_chunks(&input, 5, |chunk| chunk.to_vec());
    assert_eq!(result, input);
}

/// Empty input returns an empty Vec without panicking.
#[test]
fn par_map_chunks_empty_input() {
    let input: Vec<i32> = vec![];
    let result = par_map_chunks(&input, 10, |chunk| chunk.to_vec());
    assert!(result.is_empty());
}

/// Last chunk is smaller than `chunk_size` when `len` is not a multiple.
#[test]
fn par_map_chunks_smaller_last_chunk() {
    let input: Vec<i32> = (0..17).collect();
    // chunk sizes: 5, 5, 5, 2
    let result = par_map_chunks(&input, 5, |chunk| vec![chunk.len() as i32]);
    assert_eq!(result, vec![5, 5, 5, 2]);
}

/// When `SCIRS2_FORCE_SERIAL` is set, execution is serial (result still correct).
#[test]
fn par_map_chunks_single_thread_fallback() {
    // Safety: test-only env mutation; acceptable in integration test binary.
    unsafe {
        std::env::set_var("SCIRS2_FORCE_SERIAL", "1");
    }
    let input: Vec<i32> = (0..20).collect();
    let result = par_map_chunks(&input, 5, |chunk| chunk.iter().map(|&x| x + 1).collect());
    let expected: Vec<i32> = (1..21).collect();
    assert_eq!(result, expected);
    unsafe {
        std::env::remove_var("SCIRS2_FORCE_SERIAL");
    }
}

/// The function works with non-primitive output types.
#[test]
fn par_map_chunks_returns_vec_of_structs() {
    #[derive(Debug, PartialEq)]
    struct Item {
        val: i32,
    }
    let input: Vec<i32> = (0..10).collect();
    let result = par_map_chunks(&input, 3, |chunk| {
        chunk.iter().map(|&v| Item { val: v }).collect()
    });
    assert_eq!(result.len(), 10);
    assert_eq!(result[0].val, 0);
    assert_eq!(result[9].val, 9);
}

/// When `chunk_size` exceeds the input length, a single chunk is processed.
#[test]
fn par_map_chunks_large_chunk_size() {
    let input: Vec<i32> = (0..5).collect();
    let result = par_map_chunks(&input, 100, |chunk| chunk.to_vec());
    assert_eq!(result, input);
}

/// When `chunk_size == 1`, each element is its own chunk.
#[test]
fn par_map_chunks_chunk_size_one() {
    let input: Vec<i32> = (0..10).collect();
    let result = par_map_chunks(&input, 1, |chunk| vec![chunk[0] * 3]);
    let expected: Vec<i32> = (0..10).map(|x| x * 3).collect();
    assert_eq!(result, expected);
}
