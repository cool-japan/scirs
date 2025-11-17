//! Memory-aligned SIMD operations for optimal performance
//!
//! This module provides SIMD operations that work with properly aligned memory
//! for maximum performance. These operations are designed to be faster than
//! the general SIMD operations when you control the memory layout.

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::{self, NonNull};
use std::slice;

/// Memory alignment for SIMD operations (32 bytes for AVX2)
pub const SIMD_ALIGNMENT: usize = 32;

/// A memory-aligned vector for optimal SIMD performance
pub struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
}

impl<T> AlignedVec<T> {
    /// Create a new aligned vector with the specified capacity
    pub fn with_capacity(capacity: usize) -> Result<Self, Box<dyn std::error::Error>> {
        if capacity == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len: 0,
                capacity: 0,
            });
        }

        // Handle Zero-Sized Types (ZST) to avoid undefined behavior
        // When size_of::<T>() == 0, we must not call alloc()
        if std::mem::size_of::<T>() == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len: 0,
                capacity,
            });
        }

        let layout = Layout::from_size_align(capacity * std::mem::size_of::<T>(), SIMD_ALIGNMENT)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err("Memory allocation failed".into());
        }

        Ok(Self {
            ptr: unsafe { NonNull::new_unchecked(ptr as *mut T) },
            len: 0,
            capacity,
        })
    }

    /// Create a new aligned vector from an existing vector
    pub fn from_vec(vec: Vec<T>) -> Result<Self, Box<dyn std::error::Error>>
    where
        T: Copy,
    {
        let mut aligned = Self::with_capacity(vec.len())?;
        for item in vec {
            aligned.push(item);
        }
        Ok(aligned)
    }

    /// Push an element to the vector
    pub fn push(&mut self, value: T) {
        if self.len >= self.capacity {
            panic!("AlignedVec capacity exceeded");
        }

        unsafe {
            ptr::write(self.ptr.as_ptr().add(self.len), value);
        }
        self.len += 1;
    }

    /// Get the length of the vector
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the capacity of the vector
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get a slice of the vector
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    /// Get a mutable slice of the vector
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }

    /// Convert to a regular Vec
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.as_slice().to_vec()
    }

    /// Unsafe method to set the length directly
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - `new_len` is <= capacity
    /// - Elements from index 0 to new_len-1 are properly initialized
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.capacity);
        self.len = new_len;
    }

    /// Get a mutable pointer to the underlying data
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Get a pointer to the underlying data
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.capacity != 0 {
            unsafe {
                // Drop all elements
                for i in 0..self.len {
                    ptr::drop_in_place(self.ptr.as_ptr().add(i));
                }

                // Deallocate memory
                let layout = Layout::from_size_align_unchecked(
                    self.capacity * std::mem::size_of::<T>(),
                    SIMD_ALIGNMENT,
                );
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

unsafe impl<T: Send> Send for AlignedVec<T> {}
unsafe impl<T: Sync> Sync for AlignedVec<T> {}

/// High-performance SIMD addition for aligned f32 vectors
pub fn simd_add_aligned_f32(a: &[f32], b: &[f32]) -> Result<AlignedVec<f32>, &'static str> {
    if a.len() != b.len() {
        return Err("Arrays must have the same length");
    }

    let len = a.len();
    let mut result: AlignedVec<f32> =
        AlignedVec::with_capacity(len).map_err(|_| "Failed to allocate aligned memory")?;

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut i = 0;

                // Process 8 f32s at a time with AVX2
                while i + 8 <= len {
                    let a_ptr = a.as_ptr().add(i);
                    let b_ptr = b.as_ptr().add(i);
                    let result_ptr = result.ptr.as_ptr().add(i);

                    // Use aligned loads if possible
                    let a_vec = if (a_ptr as usize) % 32 == 0 {
                        _mm256_load_ps(a_ptr)
                    } else {
                        _mm256_loadu_ps(a_ptr)
                    };

                    let b_vec = if (b_ptr as usize) % 32 == 0 {
                        _mm256_load_ps(b_ptr)
                    } else {
                        _mm256_loadu_ps(b_ptr)
                    };

                    let result_vec = _mm256_add_ps(a_vec, b_vec);

                    // Store aligned result
                    _mm256_store_ps(result_ptr, result_vec);

                    i += 8;
                }

                // Update length for the SIMD-processed elements
                result.len = i;

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] + b[j]);
                }
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe {
                let mut i = 0;

                // Process 4 f32s at a time with SSE
                while i + 4 <= len {
                    let a_ptr = a.as_ptr().add(i);
                    let b_ptr = b.as_ptr().add(i);
                    let result_ptr = result.ptr.as_ptr().add(i);

                    let a_vec = if (a_ptr as usize) % 16 == 0 {
                        _mm_load_ps(a_ptr)
                    } else {
                        _mm_loadu_ps(a_ptr)
                    };

                    let b_vec = if (b_ptr as usize) % 16 == 0 {
                        _mm_load_ps(b_ptr)
                    } else {
                        _mm_loadu_ps(b_ptr)
                    };

                    let result_vec = _mm_add_ps(a_vec, b_vec);
                    _mm_store_ps(result_ptr, result_vec);

                    i += 4;
                }

                result.len = i;

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] + b[j]);
                }
            }
        } else {
            // Scalar fallback
            for i in 0..len {
                result.push(a[i] + b[i]);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let mut i = 0;

                // Process 4 f32s at a time with NEON
                while i + 4 <= len {
                    let a_ptr = a.as_ptr().add(i);
                    let b_ptr = b.as_ptr().add(i);
                    let result_ptr = result.ptr.as_ptr().add(i);

                    let a_vec = vld1q_f32(a_ptr);
                    let b_vec = vld1q_f32(b_ptr);
                    let result_vec = vaddq_f32(a_vec, b_vec);
                    vst1q_f32(result_ptr, result_vec);

                    i += 4;
                }

                result.len = i;

                // Handle remaining elements
                for j in i..len {
                    result.push(a[j] + b[j]);
                }
            }
        } else {
            // Scalar fallback
            for i in 0..len {
                result.push(a[i] + b[i]);
            }
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Scalar fallback for other architectures
        for i in 0..len {
            result.push(a[i] + b[i]);
        }
    }

    Ok(result)
}

/// High-performance SIMD multiplication for aligned f32 vectors
pub fn simd_mul_aligned_f32(a: &[f32], b: &[f32]) -> Result<AlignedVec<f32>, &'static str> {
    if a.len() != b.len() {
        return Err("Arrays must have the same length");
    }

    let len = a.len();
    let mut result: AlignedVec<f32> =
        AlignedVec::with_capacity(len).map_err(|_| "Failed to allocate aligned memory")?;

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut i = 0;

                while i + 8 <= len {
                    let a_ptr = a.as_ptr().add(i);
                    let b_ptr = b.as_ptr().add(i);
                    let result_ptr = result.ptr.as_ptr().add(i);

                    let a_vec = if (a_ptr as usize) % 32 == 0 {
                        _mm256_load_ps(a_ptr)
                    } else {
                        _mm256_loadu_ps(a_ptr)
                    };

                    let b_vec = if (b_ptr as usize) % 32 == 0 {
                        _mm256_load_ps(b_ptr)
                    } else {
                        _mm256_loadu_ps(b_ptr)
                    };

                    let result_vec = _mm256_mul_ps(a_vec, b_vec);
                    _mm256_store_ps(result_ptr, result_vec);

                    i += 8;
                }

                result.len = i;

                for j in i..len {
                    result.push(a[j] * b[j]);
                }
            }
        } else {
            // Fallback
            for i in 0..len {
                result.push(a[i] * b[i]);
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // Fallback for other architectures
        for i in 0..len {
            result.push(a[i] * b[i]);
        }
    }

    Ok(result)
}

/// High-performance SIMD dot product for aligned f32 vectors
pub fn simd_dot_aligned_f32(a: &[f32], b: &[f32]) -> Result<f32, &'static str> {
    if a.len() != b.len() {
        return Err("Arrays must have the same length");
    }

    let len = a.len();

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut sums = _mm256_setzero_ps();
                let mut i = 0;

                while i + 8 <= len {
                    let a_ptr = a.as_ptr().add(i);
                    let b_ptr = b.as_ptr().add(i);

                    let a_vec = if (a_ptr as usize) % 32 == 0 {
                        _mm256_load_ps(a_ptr)
                    } else {
                        _mm256_loadu_ps(a_ptr)
                    };

                    let b_vec = if (b_ptr as usize) % 32 == 0 {
                        _mm256_load_ps(b_ptr)
                    } else {
                        _mm256_loadu_ps(b_ptr)
                    };

                    let product = _mm256_mul_ps(a_vec, b_vec);
                    sums = _mm256_add_ps(sums, product);

                    i += 8;
                }

                // Horizontal sum
                let high = _mm256_extractf128_ps(sums, 1);
                let low = _mm256_castps256_ps128(sums);
                let sum128 = _mm_add_ps(low, high);

                let shuf = _mm_shuffle_ps(sum128, sum128, 0b1110);
                let sum_partial = _mm_add_ps(sum128, shuf);
                let shuf2 = _mm_shuffle_ps(sum_partial, sum_partial, 0b0001);
                let final_sum = _mm_add_ps(sum_partial, shuf2);

                let mut result = _mm_cvtss_f32(final_sum);

                // Handle remaining elements
                for j in i..len {
                    result += a[j] * b[j];
                }

                return Ok(result);
            }
        }
    }

    // Fallback
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_vec_creation() {
        let mut vec = AlignedVec::<f32>::with_capacity(16).unwrap();
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 16);

        vec.push(1.0);
        vec.push(2.0);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.as_slice(), &[1.0, 2.0]);
    }

    #[test]
    fn test_simd_add_aligned() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = vec![10.0f32, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let result = simd_add_aligned_f32(&a, &b).unwrap();
        let expected = vec![11.0f32; 10];

        assert_eq!(result.as_slice(), &expected);
    }

    #[test]
    fn test_simd_dot_aligned() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];

        let result = simd_dot_aligned_f32(&a, &b).unwrap();
        let expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0; // = 70.0

        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_alignment() {
        let mut vec = AlignedVec::<f32>::with_capacity(32).unwrap();
        // Add some elements to ensure non-empty vector
        vec.push(1.0);
        vec.push(2.0);
        vec.push(3.0);
        vec.push(4.0);

        let ptr = vec.as_slice().as_ptr() as usize;
        assert_eq!(ptr % SIMD_ALIGNMENT, 0, "Vector should be properly aligned");
    }
}
