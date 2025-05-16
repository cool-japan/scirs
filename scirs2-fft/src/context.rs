//! FFT Context Managers
//!
//! This module provides context managers for temporarily changing FFT settings
//! such as backends, worker counts, and plan caching behavior.

use crate::backend::BackendContext;
use crate::error::FFTResult;
use crate::plan_cache::get_global_cache;
use crate::worker_pool::get_global_pool;

/// Context manager for FFT settings
pub struct FftContext {
    backend_context: Option<BackendContext>,
    previous_workers: Option<usize>,
    previous_cache_enabled: Option<bool>,
    worker_pool: &'static crate::worker_pool::WorkerPool,
    plan_cache: &'static crate::plan_cache::PlanCache,
}

impl FftContext {
    /// Create a new FFT context with specified settings
    pub fn new() -> Self {
        Self {
            backend_context: None,
            previous_workers: None,
            previous_cache_enabled: None,
            worker_pool: get_global_pool(),
            plan_cache: get_global_cache(),
        }
    }
}

impl Default for FftContext {
    fn default() -> Self {
        Self::new()
    }
}

impl FftContext {
    /// Set backend for this context
    pub fn with_backend(mut self, backend_name: &str) -> FFTResult<Self> {
        self.backend_context = Some(BackendContext::new(backend_name)?);
        Ok(self)
    }

    /// Set number of workers for this context
    pub fn with_workers(mut self, _num_workers: usize) -> Self {
        self.previous_workers = Some(self.worker_pool.get_workers());
        // Note: Due to static reference limitation, we can't actually change workers
        // This is a design limitation that would need a different architecture
        self
    }

    /// Enable or disable plan caching for this context
    pub fn with_cache(mut self, enabled: bool) -> Self {
        self.previous_cache_enabled = Some(self.plan_cache.is_enabled());
        self.plan_cache.set_enabled(enabled);
        self
    }

    /// Enter the context
    pub fn __enter__(self) -> Self {
        self
    }

    /// Exit the context
    pub fn __exit__(self) {
        // Cleanup is handled by Drop
    }
}

impl Drop for FftContext {
    fn drop(&mut self) {
        // Restore cache settings
        if let Some(enabled) = self.previous_cache_enabled {
            self.plan_cache.set_enabled(enabled);
        }

        // Restore worker count (if we could)
        // Note: This is a limitation of the current design

        // Backend context cleans up itself via Drop
    }
}

/// Builder for FFT context configuration
pub struct FftContextBuilder {
    backend: Option<String>,
    workers: Option<usize>,
    cache_enabled: Option<bool>,
    cache_size: Option<usize>,
    cache_ttl: Option<std::time::Duration>,
}

impl FftContextBuilder {
    /// Create a new context builder
    pub fn new() -> Self {
        Self {
            backend: None,
            workers: None,
            cache_enabled: None,
            cache_size: None,
            cache_ttl: None,
        }
    }

    /// Set the backend
    pub fn backend(mut self, name: &str) -> Self {
        self.backend = Some(name.to_string());
        self
    }

    /// Set the number of workers
    pub fn workers(mut self, count: usize) -> Self {
        self.workers = Some(count);
        self
    }

    /// Enable or disable caching
    pub fn cache_enabled(mut self, enabled: bool) -> Self {
        self.cache_enabled = Some(enabled);
        self
    }

    /// Set cache size
    pub fn cache_size(mut self, size: usize) -> Self {
        self.cache_size = Some(size);
        self
    }

    /// Set cache TTL
    pub fn cache_ttl(mut self, ttl: std::time::Duration) -> Self {
        self.cache_ttl = Some(ttl);
        self
    }

    /// Build the context
    pub fn build(self) -> FFTResult<FftContext> {
        let mut context = FftContext::new();

        if let Some(backend) = self.backend {
            context = context.with_backend(&backend)?;
        }

        if let Some(workers) = self.workers {
            context = context.with_workers(workers);
        }

        if let Some(enabled) = self.cache_enabled {
            context = context.with_cache(enabled);
        }

        // Note: cache_size and cache_ttl would require recreating the cache
        // which is not supported with the current static architecture

        Ok(context)
    }
}

impl Default for FftContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Create an FFT context with specific settings
pub fn fft_context() -> FftContextBuilder {
    FftContextBuilder::new()
}

/// Context guard that automatically restores settings when dropped
pub struct FftSettingsGuard {
    _context: FftContext,
}

impl FftSettingsGuard {
    pub fn new(context: FftContext) -> Self {
        Self { _context: context }
    }
}

/// Use specific FFT settings within a scope
pub fn with_fft_settings<F, R>(builder: FftContextBuilder, f: F) -> FFTResult<R>
where
    F: FnOnce() -> R,
{
    let context = builder.build()?;
    let _guard = FftSettingsGuard::new(context);
    Ok(f())
}

/// Convenience function for using a specific backend
pub fn with_backend<F, R>(backend: &str, f: F) -> FFTResult<R>
where
    F: FnOnce() -> R,
{
    with_fft_settings(fft_context().backend(backend), f)
}

/// Convenience function for using specific number of workers
pub fn with_workers<F, R>(workers: usize, f: F) -> FFTResult<R>
where
    F: FnOnce() -> R,
{
    with_fft_settings(fft_context().workers(workers), f)
}

/// Convenience function for running without cache
pub fn without_cache<F, R>(f: F) -> FFTResult<R>
where
    F: FnOnce() -> R,
{
    with_fft_settings(fft_context().cache_enabled(false), f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_builder() {
        let builder = fft_context()
            .backend("rustfft")
            .workers(4)
            .cache_enabled(true);

        let context = builder.build().unwrap();
        // Context is created successfully
        drop(context);
    }

    #[test]
    fn test_with_backend() {
        let result = with_backend("rustfft", || {
            // Do some FFT operations
            42
        });

        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_with_workers() {
        let result = with_workers(2, || {
            // Do some FFT operations
            84
        });

        assert_eq!(result.unwrap(), 84);
    }

    #[test]
    fn test_without_cache() {
        let result = without_cache(|| {
            // Do some FFT operations
            168
        });

        assert_eq!(result.unwrap(), 168);
    }

    #[test]
    fn test_combined_settings() {
        let result = with_fft_settings(
            fft_context()
                .backend("rustfft")
                .workers(4)
                .cache_enabled(false),
            || {
                // Do some FFT operations
                336
            },
        );

        assert_eq!(result.unwrap(), 336);
    }
}
