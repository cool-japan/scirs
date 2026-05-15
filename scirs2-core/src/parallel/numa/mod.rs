//! NUMA-locality parallel map over chunks.
//!
//! This module exposes [`fn@par_map_chunks`] — a typed-result chunk map that uses
//! a NUMA-aware rayon thread pool on Linux (best-effort CPU pinning via
//! `pthread_setaffinity_np`) and falls back to plain rayon on other platforms.

pub mod par_map_chunks;
pub use par_map_chunks::par_map_chunks;
