//! FFT Backend System
//!
//! This module provides a pluggable backend system for FFT implementations,
//! similar to SciPy's backend model. This allows users to choose between
//! different FFT implementations at runtime.

use crate::error::{FFTError, FFTResult};
use num_complex::Complex64;
use rustfft::FftPlanner;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// FFT Backend trait for implementing different FFT algorithms
pub trait FftBackend: Send + Sync {
    /// Name of the backend
    fn name(&self) -> &str;

    /// Description of the backend
    fn description(&self) -> &str;

    /// Check if this backend is available
    fn is_available(&self) -> bool;

    /// Perform forward FFT
    fn fft(&self, input: &[Complex64], output: &mut [Complex64]) -> FFTResult<()>;

    /// Perform inverse FFT
    fn ifft(&self, input: &[Complex64], output: &mut [Complex64]) -> FFTResult<()>;

    /// Perform FFT with specific size (may be cached)
    fn fft_sized(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
        size: usize,
    ) -> FFTResult<()>;

    /// Perform inverse FFT with specific size (may be cached)
    fn ifft_sized(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
        size: usize,
    ) -> FFTResult<()>;

    /// Check if this backend supports a specific feature
    fn supports_feature(&self, feature: &str) -> bool;
}

/// RustFFT backend implementation
pub struct RustFftBackend {
    planner: Arc<Mutex<FftPlanner<f64>>>,
}

impl RustFftBackend {
    pub fn new() -> Self {
        Self {
            planner: Arc::new(Mutex::new(FftPlanner::new())),
        }
    }
}

impl Default for RustFftBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl FftBackend for RustFftBackend {
    fn name(&self) -> &str {
        "rustfft"
    }

    fn description(&self) -> &str {
        "Pure Rust FFT implementation using RustFFT library"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn fft(&self, input: &[Complex64], output: &mut [Complex64]) -> FFTResult<()> {
        self.fft_sized(input, output, input.len())
    }

    fn ifft(&self, input: &[Complex64], output: &mut [Complex64]) -> FFTResult<()> {
        self.ifft_sized(input, output, input.len())
    }

    fn fft_sized(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
        size: usize,
    ) -> FFTResult<()> {
        if input.len() != size || output.len() != size {
            return Err(FFTError::ValueError(
                "Input and output sizes must match the specified size".to_string(),
            ));
        }

        // Get cached plan from the planner
        let mut planner = self.planner.lock().unwrap();
        let fft = planner.plan_fft_forward(size);

        // Convert to rustfft's Complex type
        let mut buffer: Vec<rustfft::num_complex::Complex<f64>> = input
            .iter()
            .map(|&c| rustfft::num_complex::Complex::new(c.re, c.im))
            .collect();

        // Perform FFT
        fft.process(&mut buffer);

        // Copy to output
        for (i, &c) in buffer.iter().enumerate() {
            output[i] = Complex64::new(c.re, c.im);
        }

        Ok(())
    }

    fn ifft_sized(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
        size: usize,
    ) -> FFTResult<()> {
        if input.len() != size || output.len() != size {
            return Err(FFTError::ValueError(
                "Input and output sizes must match the specified size".to_string(),
            ));
        }

        // Get cached plan from the planner
        let mut planner = self.planner.lock().unwrap();
        let fft = planner.plan_fft_inverse(size);

        // Convert to rustfft's Complex type
        let mut buffer: Vec<rustfft::num_complex::Complex<f64>> = input
            .iter()
            .map(|&c| rustfft::num_complex::Complex::new(c.re, c.im))
            .collect();

        // Perform IFFT
        fft.process(&mut buffer);

        // Copy to output with normalization
        let scale = 1.0 / size as f64;
        for (i, &c) in buffer.iter().enumerate() {
            output[i] = Complex64::new(c.re * scale, c.im * scale);
        }

        Ok(())
    }

    fn supports_feature(&self, feature: &str) -> bool {
        matches!(feature, "1d_fft" | "2d_fft" | "nd_fft" | "cached_plans")
    }
}

/// Backend manager for FFT operations
pub struct BackendManager {
    backends: Arc<Mutex<HashMap<String, Arc<dyn FftBackend>>>>,
    current_backend: Arc<Mutex<String>>,
}

impl BackendManager {
    /// Create a new backend manager with default backends
    pub fn new() -> Self {
        let mut backends = HashMap::new();

        // Add default RustFFT backend
        let rustfft_backend = Arc::new(RustFftBackend::new()) as Arc<dyn FftBackend>;
        backends.insert("rustfft".to_string(), rustfft_backend);

        Self {
            backends: Arc::new(Mutex::new(backends)),
            current_backend: Arc::new(Mutex::new("rustfft".to_string())),
        }
    }

    /// Register a new backend
    pub fn register_backend(&self, name: String, backend: Arc<dyn FftBackend>) -> FFTResult<()> {
        let mut backends = self.backends.lock().unwrap();
        if backends.contains_key(&name) {
            return Err(FFTError::ValueError(format!(
                "Backend '{}' already exists",
                name
            )));
        }
        backends.insert(name, backend);
        Ok(())
    }

    /// Get available backends
    pub fn list_backends(&self) -> Vec<String> {
        let backends = self.backends.lock().unwrap();
        backends.keys().cloned().collect()
    }

    /// Set the current backend
    pub fn set_backend(&self, name: &str) -> FFTResult<()> {
        let backends = self.backends.lock().unwrap();
        if !backends.contains_key(name) {
            return Err(FFTError::ValueError(format!(
                "Backend '{}' not found",
                name
            )));
        }

        // Check if backend is available
        if let Some(backend) = backends.get(name) {
            if !backend.is_available() {
                return Err(FFTError::ValueError(format!(
                    "Backend '{}' is not available",
                    name
                )));
            }
        }

        *self.current_backend.lock().unwrap() = name.to_string();
        Ok(())
    }

    /// Get current backend name
    pub fn get_backend_name(&self) -> String {
        self.current_backend.lock().unwrap().clone()
    }

    /// Get current backend
    pub fn get_backend(&self) -> Arc<dyn FftBackend> {
        let current_name = self.current_backend.lock().unwrap();
        let backends = self.backends.lock().unwrap();
        backends
            .get(&*current_name)
            .cloned()
            .expect("Current backend should always exist")
    }

    /// Get backend info
    pub fn get_backend_info(&self, name: &str) -> Option<BackendInfo> {
        let backends = self.backends.lock().unwrap();
        backends.get(name).map(|backend| BackendInfo {
            name: backend.name().to_string(),
            description: backend.description().to_string(),
            available: backend.is_available(),
        })
    }
}

impl Default for BackendManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a backend
#[derive(Debug, Clone)]
pub struct BackendInfo {
    pub name: String,
    pub description: String,
    pub available: bool,
}

impl std::fmt::Display for BackendInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} - {} ({})",
            self.name,
            self.description,
            if self.available {
                "available"
            } else {
                "not available"
            }
        )
    }
}

/// Global backend manager instance
static GLOBAL_BACKEND_MANAGER: OnceLock<BackendManager> = OnceLock::new();

/// Get the global backend manager
pub fn get_backend_manager() -> &'static BackendManager {
    GLOBAL_BACKEND_MANAGER.get_or_init(BackendManager::new)
}

/// Initialize global backend manager with custom configuration
pub fn init_backend_manager(manager: BackendManager) -> Result<(), &'static str> {
    GLOBAL_BACKEND_MANAGER
        .set(manager)
        .map_err(|_| "Global backend manager already initialized")
}

/// List available backends
pub fn list_backends() -> Vec<String> {
    get_backend_manager().list_backends()
}

/// Set the current backend
pub fn set_backend(name: &str) -> FFTResult<()> {
    get_backend_manager().set_backend(name)
}

/// Get current backend name
pub fn get_backend_name() -> String {
    get_backend_manager().get_backend_name()
}

/// Get backend information
pub fn get_backend_info(name: &str) -> Option<BackendInfo> {
    get_backend_manager().get_backend_info(name)
}

/// Context manager for temporarily using a different backend
pub struct BackendContext {
    previous_backend: String,
    manager: &'static BackendManager,
}

impl BackendContext {
    /// Create a new backend context
    pub fn new(backend_name: &str) -> FFTResult<Self> {
        let manager = get_backend_manager();
        let previous_backend = manager.get_backend_name();

        // Set the new backend
        manager.set_backend(backend_name)?;

        Ok(Self {
            previous_backend,
            manager,
        })
    }
}

impl Drop for BackendContext {
    fn drop(&mut self) {
        // Restore previous backend
        let _ = self.manager.set_backend(&self.previous_backend);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rustfft_backend() {
        let backend = RustFftBackend::new();
        assert_eq!(backend.name(), "rustfft");
        assert!(backend.is_available());
        assert!(backend.supports_feature("1d_fft"));
    }

    #[test]
    fn test_backend_manager() {
        let manager = BackendManager::new();

        // Check default backend
        assert_eq!(manager.get_backend_name(), "rustfft");

        // List backends
        let backends = manager.list_backends();
        assert!(backends.contains(&"rustfft".to_string()));

        // Get backend info
        let info = manager.get_backend_info("rustfft").unwrap();
        assert!(info.available);
    }

    #[test]
    fn test_backend_context() {
        let manager = get_backend_manager();
        let original = manager.get_backend_name();

        {
            let _ctx = BackendContext::new("rustfft").unwrap();
            assert_eq!(manager.get_backend_name(), "rustfft");
        }

        // Backend should be restored
        assert_eq!(manager.get_backend_name(), original);
    }
}
