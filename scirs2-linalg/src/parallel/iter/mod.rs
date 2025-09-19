//! Parallel iterator utilities for matrix operations

use scirs2_core::parallel_ops::*;

/// Process chunks of work in parallel
///
/// # Arguments
///
/// * `items` - Items to process
/// * `chunksize` - Size of each chunk
/// * `f` - Function to apply to each chunk
///
/// # Returns
///
/// * Vector of results from each chunk
pub fn parallel_chunks<T, R, F>(_items: &[T], chunksize: usize, f: F) -> Vec<R>
where
    T: Send + Sync,
    R: Send,
    F: Fn(&[T]) -> R + Send + Sync,
{
    _items
        .chunks(chunksize)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(f)
        .collect()
}

/// Process items in parallel with index information
///
/// # Arguments
///
/// * `items` - Items to process
/// * `f` - Function to apply to each (index, item) pair
///
/// # Returns
///
/// * Vector of results
pub fn parallel_enumerate<T, R, F>(items: &[T], f: F) -> Vec<R>
where
    T: Send + Sync,
    R: Send,
    F: Fn(usize, &T) -> R + Send + Sync,
{
    items
        .par_iter()
        .enumerate()
        .map(|(i, item)| f(i, item))
        .collect()
}
