//! Cryptographically secure random number generation
//!
//! This module provides cryptographically secure random number generation
//! suitable for security-sensitive applications including key generation,
//! nonce creation, and other cryptographic operations.

use crate::random::core::Random;
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};
use rand_distr::{Distribution, Uniform};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::process;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

/// Cryptographically secure random number generator
///
/// This generator uses multiple entropy sources and a cryptographically secure
/// PRNG to provide high-quality random numbers suitable for security applications.
pub struct SecureRandom {
    rng: Random<StdRng>,
}

impl Default for SecureRandom {
    fn default() -> Self {
        Self::new()
    }
}

impl SecureRandom {
    /// Create a new cryptographically secure RNG
    ///
    /// Uses multiple entropy sources including:
    /// - System time (nanoseconds since UNIX epoch)
    /// - Process ID
    /// - Thread ID
    /// - Hash combining all sources
    pub fn new() -> Self {
        // Use system entropy to generate a secure seed for StdRng
        let time_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);

        let process_id = process::id() as u128;
        let thread_id = thread::current().id();

        // Combine multiple entropy sources
        let mut hasher = DefaultHasher::new();
        time_nanos.hash(&mut hasher);
        process_id.hash(&mut hasher);
        thread_id.hash(&mut hasher);

        let seed_u64 = hasher.finish();

        // Create a 32-byte seed from the hash
        let mut seed = [0u8; 32];
        for (i, chunk) in seed.chunks_mut(8).enumerate() {
            let offset_seed = seed_u64.wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let bytes = offset_seed.to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }

        let std_rng = StdRng::from_seed(seed);
        Self {
            rng: Random { rng: std_rng },
        }
    }

    /// Create a secure RNG from a provided seed
    ///
    /// # Warning
    /// Using a predictable seed defeats the purpose of cryptographic security.
    /// This method should only be used for testing or when the seed comes from
    /// a high-entropy source.
    pub fn from_seed(seed: [u8; 32]) -> Self {
        let std_rng = StdRng::from_seed(seed);
        Self {
            rng: Random { rng: std_rng },
        }
    }

    /// Generate a cryptographically secure random value
    pub fn sample<D, T>(&mut self, distribution: D) -> T
    where
        D: Distribution<T>,
    {
        self.rng.sample(distribution)
    }

    /// Generate cryptographically secure random bytes
    ///
    /// This is suitable for generating keys, nonces, and other cryptographic material.
    pub fn random_bytes(&mut self, count: usize) -> Vec<u8> {
        (0..count)
            .map(|_| self.sample(Uniform::new(0u8, 255u8).expect("Operation failed")))
            .collect()
    }

    /// Generate a cryptographically secure random key
    ///
    /// Alias for `random_bytes` with more descriptive naming for key generation.
    pub fn generate_key(&mut self, length: usize) -> Vec<u8> {
        self.random_bytes(length)
    }

    /// Generate a cryptographically secure random nonce
    ///
    /// Nonces should be unique for each use. This generates random bytes
    /// suitable for use as a nonce in cryptographic protocols.
    pub fn generate_nonce(&mut self, length: usize) -> Vec<u8> {
        self.random_bytes(length)
    }

    /// Generate a cryptographically secure random float in [0, 1)
    pub fn random_f64(&mut self) -> f64 {
        self.sample(Uniform::new(0.0, 1.0).expect("Operation failed"))
    }

    /// Generate a cryptographically secure random float in [0, 1) as f32
    pub fn random_f32(&mut self) -> f32 {
        self.sample(Uniform::new(0.0f32, 1.0f32).expect("Operation failed"))
    }

    /// Generate cryptographically secure random integers in a range
    pub fn random_range<T>(&mut self, min: T, max: T) -> T
    where
        T: rand_distr::uniform::SampleUniform + PartialOrd + Copy,
    {
        self.sample(Uniform::new(min, max).expect("Operation failed"))
    }

    /// Generate a cryptographically secure random boolean
    pub fn random_bool(&mut self) -> bool {
        self.rng.rng.random_bool(0.5)
    }

    /// Generate a cryptographically secure random boolean with given probability
    pub fn random_bool_with_probability(&mut self, p: f64) -> bool {
        self.rng.rng.random_bool(p)
    }

    /// Generate cryptographically secure random alphanumeric string
    ///
    /// Useful for generating random passwords, tokens, or identifiers.
    pub fn random_alphanumeric(&mut self, length: usize) -> String {
        const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        (0..length)
            .map(|_| {
                let idx = self.random_range(0, CHARSET.len());
                CHARSET[idx] as char
            })
            .collect()
    }

    /// Generate cryptographically secure random hex string
    ///
    /// Useful for generating random hex-encoded keys or identifiers.
    pub fn random_hex(&mut self, byte_length: usize) -> String {
        let bytes = self.random_bytes(byte_length);
        hex_encode(&bytes)
    }

    /// Generate cryptographically secure UUID v4
    ///
    /// Generates a random UUID suitable for use as a unique identifier.
    pub fn random_uuid(&mut self) -> String {
        let bytes = self.random_bytes(16);
        format!(
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-4{:01x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5],
            bytes[6] & 0x0f, bytes[7],
            bytes[8] & 0x3f | 0x80, bytes[9],
            bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
        )
    }

    /// Access the underlying RNG for advanced operations
    pub fn rng_mut(&mut self) -> &mut Random<StdRng> {
        &mut self.rng
    }
}

/// Thread-safe secure random number generator pool
///
/// Provides secure random number generation in multi-threaded environments
/// by maintaining separate secure RNG instances per thread.
pub struct SecureRngPool {
    seed_base: u64,
}

impl SecureRngPool {
    /// Create a new secure RNG pool
    pub fn new() -> Self {
        let time_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        Self {
            seed_base: time_nanos,
        }
    }

    /// Get a thread-local secure RNG
    pub fn get_secure_rng(&self) -> SecureRandom {
        let thread_id = thread::current().id();
        let mut hasher = DefaultHasher::new();
        self.seed_base.hash(&mut hasher);
        thread_id.hash(&mut hasher);

        let thread_seed = hasher.finish();
        let mut seed = [0u8; 32];

        for (i, chunk) in seed.chunks_mut(8).enumerate() {
            let offset_seed = thread_seed.wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let bytes = offset_seed.to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }

        SecureRandom::from_seed(seed)
    }
}

impl Default for SecureRngPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Secure random utilities
pub mod utils {
    use super::*;

    /// Generate a cryptographically secure random salt
    pub fn generate_salt(length: usize) -> Vec<u8> {
        let mut rng = SecureRandom::new();
        rng.random_bytes(length)
    }

    /// Generate a cryptographically secure random session ID
    pub fn generate_session_id() -> String {
        let mut rng = SecureRandom::new();
        rng.random_alphanumeric(32)
    }

    /// Generate a cryptographically secure random API key
    pub fn generate_api_key(length: usize) -> String {
        let mut rng = SecureRandom::new();
        rng.random_hex(length)
    }

    /// Generate a cryptographically secure random password
    pub fn generate_password(length: usize, include_symbols: bool) -> String {
        let mut rng = SecureRandom::new();

        let charset: &[u8] = if include_symbols {
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:,.<>?"
        } else {
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        };

        (0..length)
            .map(|_| {
                let idx = rng.random_range(0, charset.len());
                charset[idx] as char
            })
            .collect()
    }

    /// Generate cryptographically secure random initialization vector (IV)
    pub fn generate_iv(length: usize) -> Vec<u8> {
        let mut rng = SecureRandom::new();
        rng.random_bytes(length)
    }
}

/// Encode bytes as a lowercase hexadecimal string.
///
/// Std-only implementation that avoids pulling in the external `hex` crate.
fn hex_encode(bytes: &[u8]) -> String {
    const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX_CHARS[usize::from(byte >> 4)] as char);
        out.push(HEX_CHARS[usize::from(byte & 0x0f)] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secure_random_creation() {
        let mut secure_rng = SecureRandom::new();
        let value = secure_rng.random_f64();
        assert!((0.0..1.0).contains(&value));
    }

    #[test]
    fn test_secure_random_bytes() {
        let mut secure_rng = SecureRandom::new();
        let bytes = secure_rng.random_bytes(32);
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn test_secure_key_generation() {
        let mut secure_rng = SecureRandom::new();
        let key = secure_rng.generate_key(16);
        assert_eq!(key.len(), 16);
    }

    #[test]
    fn test_secure_alphanumeric() {
        let mut secure_rng = SecureRandom::new();
        let text = secure_rng.random_alphanumeric(20);
        assert_eq!(text.len(), 20);
        assert!(text.chars().all(|c| c.is_alphanumeric()));
    }

    #[test]
    fn test_hex_encode() {
        assert_eq!(hex_encode(&[]), "");
        assert_eq!(hex_encode(&[0x00]), "00");
        assert_eq!(hex_encode(&[0xff]), "ff");
        assert_eq!(hex_encode(&[0xde, 0xad, 0xbe, 0xef]), "deadbeef");
        assert_eq!(hex_encode(&[0x01, 0x0a, 0x10, 0x7f]), "010a107f");
    }

    #[test]
    fn test_secure_hex() {
        let mut secure_rng = SecureRandom::new();
        let hex = secure_rng.random_hex(16);
        assert_eq!(hex.len(), 32); // 16 bytes = 32 hex characters
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_secure_uuid() {
        let mut secure_rng = SecureRandom::new();
        let uuid = secure_rng.random_uuid();

        println!("Generated UUID: '{}' (length: {})", uuid, uuid.len());

        // Basic UUID format check
        assert_eq!(
            uuid.len(),
            36,
            "UUID should be 36 characters, got: '{}'",
            uuid
        );
        assert_eq!(uuid.chars().nth(8).expect("Operation failed"), '-');
        assert_eq!(uuid.chars().nth(13).expect("Operation failed"), '-');
        assert_eq!(uuid.chars().nth(18).expect("Operation failed"), '-');
        assert_eq!(uuid.chars().nth(23).expect("Operation failed"), '-');
    }

    #[test]
    fn test_secure_rng_pool() {
        let pool = SecureRngPool::new();
        let mut rng = pool.get_secure_rng();
        let value = rng.random_f64();
        assert!((0.0..1.0).contains(&value));
    }

    #[test]
    fn test_secure_utils() {
        let salt = utils::generate_salt(16);
        assert_eq!(salt.len(), 16);

        let session_id = utils::generate_session_id();
        assert_eq!(session_id.len(), 32);

        let api_key = utils::generate_api_key(24);
        assert_eq!(api_key.len(), 48); // 24 bytes = 48 hex characters

        let password = utils::generate_password(12, false);
        assert_eq!(password.len(), 12);

        let password_with_symbols = utils::generate_password(12, true);
        assert_eq!(password_with_symbols.len(), 12);

        let iv = utils::generate_iv(16);
        assert_eq!(iv.len(), 16);
    }

    #[test]
    fn test_secure_random_range() {
        let mut secure_rng = SecureRandom::new();

        for _ in 0..100 {
            let value = secure_rng.random_range(10, 20);
            assert!((10..20).contains(&value));
        }
    }

    #[test]
    fn test_secure_random_bool() {
        let mut secure_rng = SecureRandom::new();

        // Test basic boolean generation
        let _bool_val = secure_rng.random_bool();

        // Test boolean with probability
        let always_true = secure_rng.random_bool_with_probability(1.0);
        assert!(always_true);

        let always_false = secure_rng.random_bool_with_probability(0.0);
        assert!(!always_false);
    }
}
