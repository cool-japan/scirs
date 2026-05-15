//! Shared structural hashing for EML IR.
//!
//! Implements a two-seed `ahash` u128 composition. We use u128 because the
//! workspace can have ~10^6 subexpressions in flight; u64 birthday collision
//! at 2^32 is too close, while u128 puts collision at ~2^96 — negligible.
//!
//! ## Determinism
//!
//! Both seeds are fixed compile-time literals (NOT [`ahash::RandomState::new`]
//! which is process-random). Structural hashes are therefore reproducible
//! across runs — important for cache files, debug output, and test stability.

use ahash::RandomState;
use once_cell::sync::Lazy;
use std::hash::{BuildHasher, Hash};

static SEED1: Lazy<RandomState> = Lazy::new(|| {
    RandomState::with_seeds(
        0xDEAD_BEEF_DEAD_BEEF,
        0xDEAD_BEEF_DEAD_BEEF,
        0xDEAD_BEEF_DEAD_BEEF,
        0xDEAD_BEEF_DEAD_BEEF,
    )
});
static SEED2: Lazy<RandomState> = Lazy::new(|| {
    RandomState::with_seeds(
        0xCAFE_BABE_CAFE_BABE,
        0xCAFE_BABE_CAFE_BABE,
        0xCAFE_BABE_CAFE_BABE,
        0xCAFE_BABE_CAFE_BABE,
    )
});

/// Compose two `ahash` u64 seeds into a u128.
///
/// The two seeds are deterministic across process restarts (we use literal
/// constants, not [`RandomState::new`] which is process-random). This makes
/// structural hashes reproducible across runs — important for cache files
/// and debug output.
pub(crate) fn compose_u128(h1: u64, h2: u64) -> u128 {
    ((h1 as u128) << 64) | (h2 as u128)
}

/// Hash an arbitrary `T: Hash` value with both seeds and compose to u128.
pub(crate) fn hash_u128<T: Hash + ?Sized>(value: &T) -> u128 {
    compose_u128(SEED1.hash_one(value), SEED2.hash_one(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_across_calls() {
        assert_eq!(hash_u128(&42u32), hash_u128(&42u32));
        assert_eq!(hash_u128(&"hello"), hash_u128(&"hello"));
    }

    #[test]
    fn different_inputs_different_hashes() {
        assert_ne!(hash_u128(&1u32), hash_u128(&2u32));
        assert_ne!(hash_u128(&"a"), hash_u128(&"b"));
    }

    #[test]
    fn compose_is_lossless() {
        let combined = compose_u128(0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555);
        assert_eq!(combined >> 64, 0xAAAA_AAAA_AAAA_AAAA);
        assert_eq!(combined & 0xFFFF_FFFF_FFFF_FFFF, 0x5555_5555_5555_5555);
    }

    #[test]
    fn high_low_halves_distinct() {
        // The two seeds must produce distinguishable u64s for non-trivial inputs;
        // otherwise the u128 collapses to a u64-equivalent.
        let h = hash_u128(&"distinguishable");
        let high = (h >> 64) as u64;
        let low = (h & 0xFFFF_FFFF_FFFF_FFFF) as u64;
        assert_ne!(high, low);
    }
}
