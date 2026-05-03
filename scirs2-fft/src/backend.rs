//! FFT Backend System
//!
//! This module provides a pluggable backend system for FFT implementations,
//! similar to SciPy's backend model. This allows users to choose between
//! different FFT implementations at runtime.
//!
//! Currently, OxiFFT is the sole backend (COOLJAPAN Pure Rust policy).

use crate::error::{FFTError, FFTResult};
use scirs2_core::numeric::Complex64;
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

/// Backend manager for FFT operations
pub struct BackendManager {
    backends: Arc<Mutex<HashMap<String, Arc<dyn FftBackend>>>>,
    current_backend: Arc<Mutex<String>>,
}

impl BackendManager {
    /// Create a new backend manager
    pub fn new() -> Self {
        let backends = HashMap::new();
        let default_backend = "none".to_string();

        Self {
            backends: Arc::new(Mutex::new(backends)),
            current_backend: Arc::new(Mutex::new(default_backend)),
        }
    }

    /// Register a new backend
    pub fn register_backend(&self, name: String, backend: Arc<dyn FftBackend>) -> FFTResult<()> {
        let mut backends = self.backends.lock().expect("Operation failed");
        if backends.contains_key(&name) {
            return Err(FFTError::ValueError(format!(
                "Backend '{name}' already exists"
            )));
        }
        backends.insert(name, backend);
        Ok(())
    }

    /// Get available backends
    pub fn list_backends(&self) -> Vec<String> {
        let backends = self.backends.lock().expect("Operation failed");
        backends.keys().cloned().collect()
    }

    /// Set the current backend
    pub fn set_backend(&self, name: &str) -> FFTResult<()> {
        let backends = self.backends.lock().expect("Operation failed");
        if !backends.contains_key(name) {
            // If no backends are registered, silently accept (no-op)
            if backends.is_empty() || name == "none" {
                return Ok(());
            }
            return Err(FFTError::ValueError(format!("Backend '{name}' not found")));
        }

        // Check if backend is available
        if let Some(backend) = backends.get(name) {
            if !backend.is_available() {
                return Err(FFTError::ValueError(format!(
                    "Backend '{name}' is not available"
                )));
            }
        }

        *self.current_backend.lock().expect("Operation failed") = name.to_string();
        Ok(())
    }

    /// Get current backend name
    pub fn get_backend_name(&self) -> String {
        self.current_backend
            .lock()
            .expect("Operation failed")
            .clone()
    }

    /// Get current backend (returns None if no backend is registered)
    #[allow(dead_code)]
    pub fn get_backend(&self) -> Option<Arc<dyn FftBackend>> {
        let current_name = self.current_backend.lock().expect("Operation failed");
        let backends = self.backends.lock().expect("Operation failed");
        backends.get(&*current_name).cloned()
    }

    /// Get backend info
    pub fn get_backend_info(&self, name: &str) -> Option<BackendInfo> {
        let backends = self.backends.lock().expect("Operation failed");
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
#[allow(dead_code)]
pub fn get_backend_manager() -> &'static BackendManager {
    GLOBAL_BACKEND_MANAGER.get_or_init(BackendManager::new)
}

/// Initialize global backend manager with custom configuration
#[allow(dead_code)]
pub fn init_backend_manager(manager: BackendManager) -> Result<(), &'static str> {
    GLOBAL_BACKEND_MANAGER
        .set(manager)
        .map_err(|_| "Global backend _manager already initialized")
}

/// List available backends
#[allow(dead_code)]
pub fn list_backends() -> Vec<String> {
    get_backend_manager().list_backends()
}

/// Set the current backend
#[allow(dead_code)]
pub fn set_backend(name: &str) -> FFTResult<()> {
    get_backend_manager().set_backend(name)
}

/// Get current backend name
#[allow(dead_code)]
pub fn get_backend_name() -> String {
    get_backend_manager().get_backend_name()
}

/// Get backend information
#[allow(dead_code)]
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
    ///
    /// Note: If the backend name is not registered, this is a no-op context
    /// (the backend name is stored but the active backend remains unchanged).
    pub fn new(_backendname: &str) -> FFTResult<Self> {
        let manager = get_backend_manager();
        let previous_backend = manager.get_backend_name();

        // Attempt to set the new backend (gracefully handles unknown backends)
        let _ = manager.set_backend(_backendname);

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
    fn test_backend_manager_new() {
        let manager = BackendManager::new();
        assert_eq!(manager.get_backend_name(), "none");
        assert!(manager.list_backends().is_empty());
    }

    #[test]
    fn test_backend_context_unknown() {
        // BackendContext with unknown backend should succeed (no-op)
        let ctx = BackendContext::new("oxifft");
        assert!(ctx.is_ok());
    }
}
