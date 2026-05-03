//! OxiFFT Plan Cache for performance optimization
//!
//! Caches OxiFFT plans to avoid expensive plan creation on repeated FFT calls.
//! Plans are stored in global caches keyed by size (1D) or dimensions (2D).
//!
//! This replaces the FFTW plan cache while maintaining Pure Rust Policy compliance.

// Allow complex type definitions for cache types - intentional for performance
#![allow(clippy::type_complexity)]

use oxifft::{Complex, Direction, Flags, Plan, Plan2D, R2rPlan, RealPlan, RealPlan2D};
// Use the internal R2rKind which has FFTW-compatible naming (Redft10, Redft01, etc.)
use oxifft::rdft::solvers::R2rKind;
use std::collections::HashMap;
use std::sync::Mutex;

use crate::error::{FFTError, FFTResult};

// ========================================
// 1D PLAN CACHES
// ========================================

/// Global cache for R2C (real-to-complex) plans, keyed by input size
static R2C_CACHE: Mutex<Option<HashMap<usize, RealPlan<f64>>>> = Mutex::new(None);

/// Global cache for C2C forward plans, keyed by size
static C2C_FWD_CACHE: Mutex<Option<HashMap<usize, Plan<f64>>>> = Mutex::new(None);

/// Global cache for C2C backward plans, keyed by size
static C2C_BWD_CACHE: Mutex<Option<HashMap<usize, Plan<f64>>>> = Mutex::new(None);

/// Global cache for C2R (complex-to-real) plans, keyed by output size
static C2R_CACHE: Mutex<Option<HashMap<usize, RealPlan<f64>>>> = Mutex::new(None);

/// Global cache for R2R DCT-II plans
static DCT2_CACHE: Mutex<Option<HashMap<usize, R2rPlan<f64>>>> = Mutex::new(None);

/// Global cache for R2R IDCT-II (DCT-III) plans
static IDCT2_CACHE: Mutex<Option<HashMap<usize, R2rPlan<f64>>>> = Mutex::new(None);

/// Global cache for R2R DST-II plans
static DST2_CACHE: Mutex<Option<HashMap<usize, R2rPlan<f64>>>> = Mutex::new(None);

/// Global cache for R2R IDST-II (DST-III) plans
static IDST2_CACHE: Mutex<Option<HashMap<usize, R2rPlan<f64>>>> = Mutex::new(None);

// ========================================
// 2D PLAN CACHES
// ========================================

/// Global cache for 2D R2C plans, keyed by (rows, cols)
static R2C_2D_CACHE: Mutex<Option<HashMap<(usize, usize), RealPlan2D<f64>>>> = Mutex::new(None);

/// Global cache for 2D C2C forward plans
static C2C_2D_FWD_CACHE: Mutex<Option<HashMap<(usize, usize), Plan2D<f64>>>> = Mutex::new(None);

/// Global cache for 2D C2C backward plans
static C2C_2D_BWD_CACHE: Mutex<Option<HashMap<(usize, usize), Plan2D<f64>>>> = Mutex::new(None);

/// Global cache for 2D C2R plans
static C2R_2D_CACHE: Mutex<Option<HashMap<(usize, usize), RealPlan2D<f64>>>> = Mutex::new(None);

// ========================================
// 1D PLAN EXECUTION FUNCTIONS
// ========================================

/// Execute R2C FFT with cached plan
pub fn execute_r2c(input: &[f64], output: &mut [Complex<f64>]) -> FFTResult<()> {
    let n = input.len();

    let mut cache = R2C_CACHE
        .lock()
        .map_err(|e| FFTError::ComputationError(format!("Failed to lock R2C cache: {}", e)))?;

    // Initialize cache if needed
    if cache.is_none() {
        *cache = Some(HashMap::new());
    }

    let cache_map = cache
        .as_mut()
        .ok_or_else(|| FFTError::ComputationError("Cache initialization failed".to_string()))?;

    // Create plan if not cached
    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(n) {
        let plan = RealPlan::r2c_1d(n, Flags::ESTIMATE).ok_or_else(|| {
            FFTError::ComputationError(format!("Failed to create R2C plan for size {}", n))
        })?;
        e.insert(plan);
    }

    // Execute with cached plan
    let plan = cache_map
        .get(&n)
        .ok_or_else(|| FFTError::ComputationError("Failed to get cached plan".to_string()))?;
    plan.execute_r2c(input, output);
    Ok(())
}

/// Execute C2C FFT with cached plan
pub fn execute_c2c(
    input: &[Complex<f64>],
    output: &mut [Complex<f64>],
    direction: Direction,
) -> FFTResult<()> {
    let n = input.len();

    let cache = match direction {
        Direction::Forward => &C2C_FWD_CACHE,
        Direction::Backward => &C2C_BWD_CACHE,
        _ => {
            return Err(FFTError::ComputationError(format!(
                "Unsupported FFT direction: {:?}",
                direction
            )))
        }
    };

    let mut cache_guard = cache.lock().map_err(|e| {
        FFTError::ComputationError(format!("Failed to lock C2C {:?} cache: {}", direction, e))
    })?;

    if cache_guard.is_none() {
        *cache_guard = Some(HashMap::new());
    }

    let cache_map = cache_guard
        .as_mut()
        .ok_or_else(|| FFTError::ComputationError("Cache initialization failed".to_string()))?;

    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(n) {
        let plan = Plan::dft_1d(n, direction, Flags::ESTIMATE).ok_or_else(|| {
            FFTError::ComputationError(format!(
                "Failed to create C2C {:?} plan for size {}",
                direction, n
            ))
        })?;
        e.insert(plan);
    }

    let plan = cache_map
        .get(&n)
        .ok_or_else(|| FFTError::ComputationError("Failed to get cached plan".to_string()))?;
    plan.execute(input, output);
    Ok(())
}

/// Execute C2R (inverse real) FFT with cached plan
pub fn execute_c2r(input: &[Complex<f64>], output: &mut [f64], n: usize) -> FFTResult<()> {
    let mut cache = C2R_CACHE
        .lock()
        .map_err(|e| FFTError::ComputationError(format!("Failed to lock C2R cache: {}", e)))?;

    if cache.is_none() {
        *cache = Some(HashMap::new());
    }

    let cache_map = cache
        .as_mut()
        .ok_or_else(|| FFTError::ComputationError("Cache initialization failed".to_string()))?;

    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(n) {
        let plan = RealPlan::c2r_1d(n, Flags::ESTIMATE).ok_or_else(|| {
            FFTError::ComputationError(format!("Failed to create C2R plan for size {}", n))
        })?;
        e.insert(plan);
    }

    let plan = cache_map
        .get(&n)
        .ok_or_else(|| FFTError::ComputationError("Failed to get cached plan".to_string()))?;
    plan.execute_c2r_unnormalized(input, output);
    Ok(())
}

/// Execute DCT-II with cached plan
pub fn execute_dct2(input: &[f64], output: &mut [f64]) -> FFTResult<()> {
    let n = input.len();

    let mut cache = DCT2_CACHE
        .lock()
        .map_err(|e| FFTError::ComputationError(format!("Failed to lock DCT2 cache: {}", e)))?;

    if cache.is_none() {
        *cache = Some(HashMap::new());
    }

    let cache_map = cache
        .as_mut()
        .ok_or_else(|| FFTError::ComputationError("Cache initialization failed".to_string()))?;

    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(n) {
        let plan = R2rPlan::r2r_1d(n, R2rKind::Redft10, Flags::ESTIMATE).ok_or_else(|| {
            FFTError::ComputationError(format!("Failed to create DCT2 plan for size {}", n))
        })?;
        e.insert(plan);
    }

    let plan = cache_map
        .get(&n)
        .ok_or_else(|| FFTError::ComputationError("Failed to get cached plan".to_string()))?;
    plan.execute(input, output);
    Ok(())
}

/// Execute IDCT-II (DCT-III) with cached plan
pub fn execute_idct2(input: &[f64], output: &mut [f64]) -> FFTResult<()> {
    let n = input.len();

    let mut cache = IDCT2_CACHE
        .lock()
        .map_err(|e| FFTError::ComputationError(format!("Failed to lock IDCT2 cache: {}", e)))?;

    if cache.is_none() {
        *cache = Some(HashMap::new());
    }

    let cache_map = cache
        .as_mut()
        .ok_or_else(|| FFTError::ComputationError("Cache initialization failed".to_string()))?;

    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(n) {
        let plan = R2rPlan::r2r_1d(n, R2rKind::Redft01, Flags::ESTIMATE).ok_or_else(|| {
            FFTError::ComputationError(format!("Failed to create IDCT2 plan for size {}", n))
        })?;
        e.insert(plan);
    }

    let plan = cache_map
        .get(&n)
        .ok_or_else(|| FFTError::ComputationError("Failed to get cached plan".to_string()))?;
    plan.execute(input, output);
    Ok(())
}

/// Execute DST-II with cached plan
pub fn execute_dst2(input: &[f64], output: &mut [f64]) -> FFTResult<()> {
    let n = input.len();

    let mut cache = DST2_CACHE
        .lock()
        .map_err(|e| FFTError::ComputationError(format!("Failed to lock DST2 cache: {}", e)))?;

    if cache.is_none() {
        *cache = Some(HashMap::new());
    }

    let cache_map = cache
        .as_mut()
        .ok_or_else(|| FFTError::ComputationError("Cache initialization failed".to_string()))?;

    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(n) {
        let plan = R2rPlan::r2r_1d(n, R2rKind::Rodft10, Flags::ESTIMATE).ok_or_else(|| {
            FFTError::ComputationError(format!("Failed to create DST2 plan for size {}", n))
        })?;
        e.insert(plan);
    }

    let plan = cache_map
        .get(&n)
        .ok_or_else(|| FFTError::ComputationError("Failed to get cached plan".to_string()))?;
    plan.execute(input, output);
    Ok(())
}

/// Execute IDST-II (DST-III) with cached plan
pub fn execute_idst2(input: &[f64], output: &mut [f64]) -> FFTResult<()> {
    let n = input.len();

    let mut cache = IDST2_CACHE
        .lock()
        .map_err(|e| FFTError::ComputationError(format!("Failed to lock IDST2 cache: {}", e)))?;

    if cache.is_none() {
        *cache = Some(HashMap::new());
    }

    let cache_map = cache
        .as_mut()
        .ok_or_else(|| FFTError::ComputationError("Cache initialization failed".to_string()))?;

    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(n) {
        let plan = R2rPlan::r2r_1d(n, R2rKind::Rodft01, Flags::ESTIMATE).ok_or_else(|| {
            FFTError::ComputationError(format!("Failed to create IDST2 plan for size {}", n))
        })?;
        e.insert(plan);
    }

    let plan = cache_map
        .get(&n)
        .ok_or_else(|| FFTError::ComputationError("Failed to get cached plan".to_string()))?;
    plan.execute(input, output);
    Ok(())
}

// ========================================
// 2D PLAN EXECUTION FUNCTIONS
// ========================================

/// Execute 2D R2C FFT with cached plan
pub fn execute_r2c_2d(
    input: &[f64],
    output: &mut [Complex<f64>],
    rows: usize,
    cols: usize,
) -> FFTResult<()> {
    let key = (rows, cols);

    let mut cache = R2C_2D_CACHE
        .lock()
        .map_err(|e| FFTError::ComputationError(format!("Failed to lock 2D R2C cache: {}", e)))?;

    if cache.is_none() {
        *cache = Some(HashMap::new());
    }

    let cache_map = cache
        .as_mut()
        .ok_or_else(|| FFTError::ComputationError("Cache initialization failed".to_string()))?;

    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(key) {
        let plan = RealPlan2D::r2c(rows, cols, Flags::ESTIMATE).ok_or_else(|| {
            FFTError::ComputationError(format!(
                "Failed to create 2D R2C plan for size {}x{}",
                rows, cols
            ))
        })?;
        e.insert(plan);
    }

    let plan = cache_map
        .get(&key)
        .ok_or_else(|| FFTError::ComputationError("Failed to get cached plan".to_string()))?;
    plan.execute_r2c(input, output);
    Ok(())
}

/// Execute 2D C2C FFT with cached plan
pub fn execute_c2c_2d(
    input: &[Complex<f64>],
    output: &mut [Complex<f64>],
    rows: usize,
    cols: usize,
    direction: Direction,
) -> FFTResult<()> {
    let key = (rows, cols);

    let cache = match direction {
        Direction::Forward => &C2C_2D_FWD_CACHE,
        Direction::Backward => &C2C_2D_BWD_CACHE,
        _ => {
            return Err(FFTError::ComputationError(format!(
                "Unsupported FFT direction: {:?}",
                direction
            )))
        }
    };

    let mut cache_guard = cache.lock().map_err(|e| {
        FFTError::ComputationError(format!(
            "Failed to lock 2D C2C {:?} cache: {}",
            direction, e
        ))
    })?;

    if cache_guard.is_none() {
        *cache_guard = Some(HashMap::new());
    }

    let cache_map = cache_guard
        .as_mut()
        .ok_or_else(|| FFTError::ComputationError("Cache initialization failed".to_string()))?;

    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(key) {
        let plan = Plan2D::new(rows, cols, direction, Flags::ESTIMATE).ok_or_else(|| {
            FFTError::ComputationError(format!(
                "Failed to create 2D C2C {:?} plan for size {}x{}",
                direction, rows, cols
            ))
        })?;
        e.insert(plan);
    }

    let plan = cache_map
        .get(&key)
        .ok_or_else(|| FFTError::ComputationError("Failed to get cached plan".to_string()))?;
    plan.execute(input, output);
    Ok(())
}

/// Execute 2D C2R FFT with cached plan
pub fn execute_c2r_2d(
    input: &[Complex<f64>],
    output: &mut [f64],
    rows: usize,
    cols: usize,
) -> FFTResult<()> {
    let key = (rows, cols);

    let mut cache = C2R_2D_CACHE
        .lock()
        .map_err(|e| FFTError::ComputationError(format!("Failed to lock 2D C2R cache: {}", e)))?;

    if cache.is_none() {
        *cache = Some(HashMap::new());
    }

    let cache_map = cache
        .as_mut()
        .ok_or_else(|| FFTError::ComputationError("Cache initialization failed".to_string()))?;

    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(key) {
        let plan = RealPlan2D::c2r(rows, cols, Flags::ESTIMATE).ok_or_else(|| {
            FFTError::ComputationError(format!(
                "Failed to create 2D C2R plan for size {}x{}",
                rows, cols
            ))
        })?;
        e.insert(plan);
    }

    let plan = cache_map
        .get(&key)
        .ok_or_else(|| FFTError::ComputationError("Failed to get cached plan".to_string()))?;
    plan.execute_c2r(input, output);
    Ok(())
}

// ========================================
// CACHE MANAGEMENT
// ========================================

/// Clear all cached plans (useful for testing or memory management)
pub fn clear_all_caches() {
    if let Ok(mut cache) = R2C_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = C2C_FWD_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = C2C_BWD_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = C2R_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = DCT2_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = IDCT2_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = DST2_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = IDST2_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = R2C_2D_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = C2C_2D_FWD_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = C2C_2D_BWD_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = C2R_2D_CACHE.lock() {
        *cache = None;
    }
}

/// Get cache statistics for debugging/monitoring
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub r2c_count: usize,
    pub c2c_fwd_count: usize,
    pub c2c_bwd_count: usize,
    pub c2r_count: usize,
    pub dct2_count: usize,
    pub idct2_count: usize,
    pub dst2_count: usize,
    pub idst2_count: usize,
    pub r2c_2d_count: usize,
    pub c2c_2d_fwd_count: usize,
    pub c2c_2d_bwd_count: usize,
    pub c2r_2d_count: usize,
}

/// Get current cache statistics
pub fn get_cache_stats() -> CacheStats {
    let mut stats = CacheStats::default();

    if let Ok(cache) = R2C_CACHE.lock() {
        if let Some(ref map) = *cache {
            stats.r2c_count = map.len();
        }
    }
    if let Ok(cache) = C2C_FWD_CACHE.lock() {
        if let Some(ref map) = *cache {
            stats.c2c_fwd_count = map.len();
        }
    }
    if let Ok(cache) = C2C_BWD_CACHE.lock() {
        if let Some(ref map) = *cache {
            stats.c2c_bwd_count = map.len();
        }
    }
    if let Ok(cache) = C2R_CACHE.lock() {
        if let Some(ref map) = *cache {
            stats.c2r_count = map.len();
        }
    }
    if let Ok(cache) = DCT2_CACHE.lock() {
        if let Some(ref map) = *cache {
            stats.dct2_count = map.len();
        }
    }
    if let Ok(cache) = IDCT2_CACHE.lock() {
        if let Some(ref map) = *cache {
            stats.idct2_count = map.len();
        }
    }
    if let Ok(cache) = DST2_CACHE.lock() {
        if let Some(ref map) = *cache {
            stats.dst2_count = map.len();
        }
    }
    if let Ok(cache) = IDST2_CACHE.lock() {
        if let Some(ref map) = *cache {
            stats.idst2_count = map.len();
        }
    }
    if let Ok(cache) = R2C_2D_CACHE.lock() {
        if let Some(ref map) = *cache {
            stats.r2c_2d_count = map.len();
        }
    }
    if let Ok(cache) = C2C_2D_FWD_CACHE.lock() {
        if let Some(ref map) = *cache {
            stats.c2c_2d_fwd_count = map.len();
        }
    }
    if let Ok(cache) = C2C_2D_BWD_CACHE.lock() {
        if let Some(ref map) = *cache {
            stats.c2c_2d_bwd_count = map.len();
        }
    }
    if let Ok(cache) = C2R_2D_CACHE.lock() {
        if let Some(ref map) = *cache {
            stats.c2r_2d_count = map.len();
        }
    }

    stats
}
