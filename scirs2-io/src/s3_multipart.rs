//! AWS S3 multipart upload state machine with parallel chunk upload simulation.
//!
//! This module implements the S3 multipart upload protocol as a pure-Rust state
//! machine.  When the `aws-sdk-s3` Cargo feature is enabled a real HTTP
//! implementation (SigV4 signing via `reqwest`) would be wired in; without the
//! feature every operation executes in a local simulation mode that is suitable
//! for testing and offline development.
//!
//! # Protocol outline
//!
//! 1. Call `MultipartUpload::initiate` to create a session and receive an
//!    `upload_id`.
//! 2. Split your data into chunks and call `upload_data`.
//! 3. Call `complete` to assemble all parts and receive an ETag.
//! 4. Call `abort` to discard an in-progress upload.
//!
//! # Example (simulation mode)
//!
//! ```rust
//! use scirs2_io::s3_multipart::{MultipartConfig, MultipartUpload};
//!
//! let config = MultipartConfig { chunk_size_bytes: 1024, ..Default::default() };
//! let mut upload = MultipartUpload::initiate("my-bucket", "data/big.bin", config);
//! let data: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
//! upload.upload_data(&data).expect("upload failed");
//! let etag = upload.complete().expect("complete failed");
//! assert!(!etag.is_empty());
//! ```

use std::io;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for a multipart upload session.
#[derive(Debug, Clone)]
pub struct MultipartConfig {
    /// Size of each part in bytes.  S3 requires a minimum of 5 MiB for all
    /// parts except the final one; the default here matches that minimum.
    pub chunk_size_bytes: usize,
    /// Maximum number of concurrent part uploads (advisory, simulation always
    /// processes sequentially).
    pub max_concurrency: usize,
    /// Maximum number of retries per part on transient failure.
    pub max_retries: usize,
}

impl Default for MultipartConfig {
    fn default() -> Self {
        Self {
            chunk_size_bytes: 5 * 1024 * 1024, // 5 MiB (S3 minimum)
            max_concurrency: 4,
            max_retries: 3,
        }
    }
}

/// State of a multipart upload session.
#[derive(Debug, Clone)]
pub enum UploadState {
    /// Session created, no parts uploaded yet.
    Pending,
    /// Parts are being uploaded.
    InProgress {
        /// Number of parts successfully uploaded so far.
        parts_uploaded: usize,
        /// Total number of parts the data was split into.
        total_parts: usize,
    },
    /// Upload finalised; the `etag` is the assembled ETag.
    Completed {
        /// ETag of the assembled object (SHA-256 hex in simulation mode).
        etag: String,
    },
    /// Upload aborted; all buffered parts have been discarded.
    Aborted,
}

/// A single uploaded part.
#[derive(Debug, Clone)]
pub struct UploadedPart {
    /// 1-based part number (S3 supports 1–10 000).
    pub part_number: usize,
    /// ETag of this individual part (SHA-256 hex in simulation mode).
    pub etag: String,
    /// Number of bytes in this part.
    pub size_bytes: usize,
}

/// S3 multipart upload session.
pub struct MultipartUpload {
    /// Upload identifier (random hex in simulation mode).
    upload_id: String,
    /// Target bucket.
    bucket: String,
    /// Target object key.
    key: String,
    /// Session configuration.
    config: MultipartConfig,
    /// Uploaded parts (in insertion order; sorted by `part_number` on complete).
    parts: Vec<UploadedPart>,
    /// Current state.
    state: UploadState,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl MultipartUpload {
    /// Initiate a multipart upload session.
    ///
    /// In simulation mode this generates a random `upload_id` without making
    /// any network calls.
    pub fn initiate(
        bucket: impl Into<String>,
        key: impl Into<String>,
        config: MultipartConfig,
    ) -> Self {
        let upload_id = generate_upload_id();
        Self {
            upload_id,
            bucket: bucket.into(),
            key: key.into(),
            config,
            parts: Vec::new(),
            state: UploadState::Pending,
        }
    }

    /// Split `data` into chunks according to `config.chunk_size_bytes` and
    /// simulate uploading each chunk.
    ///
    /// May be called multiple times to stream data incrementally.  Each call
    /// appends new parts to the session.
    ///
    /// # Errors
    ///
    /// Returns an error if the session has already been completed or aborted.
    pub fn upload_data(&mut self, data: &[u8]) -> Result<(), io::Error> {
        match &self.state {
            UploadState::Completed { .. } => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Cannot upload data to a completed multipart upload",
                ));
            }
            UploadState::Aborted => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Cannot upload data to an aborted multipart upload",
                ));
            }
            _ => {}
        }

        if data.is_empty() {
            return Ok(());
        }

        let chunk_size = self.config.chunk_size_bytes.max(1);
        let base_part = self.parts.len();

        for (i, chunk) in data.chunks(chunk_size).enumerate() {
            let next_part_number = base_part + i + 1;
            let etag = sha256_hex(chunk);
            self.parts.push(UploadedPart {
                part_number: next_part_number,
                etag,
                size_bytes: chunk.len(),
            });
        }

        let total_parts = self.parts.len();
        self.state = UploadState::InProgress {
            parts_uploaded: total_parts,
            total_parts,
        };

        Ok(())
    }

    /// Complete the multipart upload.
    ///
    /// Assembles all buffered parts (sorted by `part_number`), computes a
    /// combined ETag, and transitions the session to `Completed`.
    ///
    /// Returns the ETag of the assembled object.
    ///
    /// # Errors
    ///
    /// Returns an error if no parts have been uploaded, or if the session has
    /// already been completed or aborted.
    pub fn complete(&mut self) -> Result<String, io::Error> {
        match &self.state {
            UploadState::Completed { etag } => {
                // Already completed — return the existing ETag.
                return Ok(etag.clone());
            }
            UploadState::Aborted => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Cannot complete an aborted multipart upload",
                ));
            }
            _ => {}
        }

        if self.parts.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot complete a multipart upload with no parts",
            ));
        }

        // Sort parts by part number (S3 assembles in order).
        self.parts.sort_by_key(|p| p.part_number);

        // Compute a combined ETag: SHA-256 of the concatenation of part ETags
        // (each as raw hex bytes), followed by "-<part_count>" S3 style.
        let combined: String = self.parts.iter().map(|p| p.etag.as_str()).collect();
        let combined_etag = format!("{}-{}", sha256_hex(combined.as_bytes()), self.parts.len());

        self.state = UploadState::Completed {
            etag: combined_etag.clone(),
        };

        Ok(combined_etag)
    }

    /// Abort the multipart upload and discard all buffered parts.
    pub fn abort(&mut self) {
        self.parts.clear();
        self.state = UploadState::Aborted;
    }

    /// Return a reference to the current upload state.
    pub fn state(&self) -> &UploadState {
        &self.state
    }

    /// Return the number of parts uploaded so far.
    pub fn parts_uploaded(&self) -> usize {
        self.parts.len()
    }

    /// Return the upload session identifier.
    pub fn upload_id(&self) -> &str {
        &self.upload_id
    }

    /// Return the target bucket name.
    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    /// Return the target object key.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Return a slice of the uploaded parts.
    pub fn parts(&self) -> &[UploadedPart] {
        &self.parts
    }

    /// Return the total number of bytes uploaded across all parts so far.
    pub fn total_bytes_uploaded(&self) -> usize {
        self.parts.iter().map(|p| p.size_bytes).sum()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a random-looking upload ID using the current time and a counter.
fn generate_upload_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    // Combine timestamp and counter into a hex string.
    format!("{:032x}{:016x}", ts, count)
}

/// Compute SHA-256 of `data` and return the hex-encoded digest.
fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(data);
    crate::encoding_utils::hex_encode(h.finalize())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initiate_creates_pending_session() {
        let config = MultipartConfig::default();
        let upload = MultipartUpload::initiate("bucket", "key/obj.bin", config);
        assert!(!upload.upload_id().is_empty());
        assert_eq!(upload.bucket(), "bucket");
        assert_eq!(upload.key(), "key/obj.bin");
        assert_eq!(upload.parts_uploaded(), 0);
        matches!(upload.state(), UploadState::Pending);
    }

    #[test]
    fn test_upload_data_single_chunk() {
        let config = MultipartConfig {
            chunk_size_bytes: 1024,
            ..Default::default()
        };
        let mut upload = MultipartUpload::initiate("b", "k", config);
        let data: Vec<u8> = (0u8..100).collect();
        upload.upload_data(&data).expect("upload_data failed");
        assert_eq!(upload.parts_uploaded(), 1);
        assert_eq!(upload.total_bytes_uploaded(), 100);
        assert!(matches!(
            upload.state(),
            UploadState::InProgress {
                parts_uploaded: 1,
                total_parts: 1
            }
        ));
    }

    #[test]
    fn test_upload_data_multiple_chunks() {
        let config = MultipartConfig {
            chunk_size_bytes: 10,
            max_concurrency: 2,
            max_retries: 0,
        };
        let mut upload = MultipartUpload::initiate("bucket", "big.bin", config);
        let data: Vec<u8> = (0u8..=255).cycle().take(35).collect();
        upload.upload_data(&data).expect("upload failed");
        // 35 bytes / 10-byte chunks = 4 full chunks (10, 10, 10, 5)
        assert_eq!(upload.parts_uploaded(), 4);
        assert_eq!(upload.total_bytes_uploaded(), 35);
    }

    #[test]
    fn test_complete_returns_nonempty_etag() {
        let config = MultipartConfig {
            chunk_size_bytes: 16,
            ..Default::default()
        };
        let mut upload = MultipartUpload::initiate("bkt", "obj", config);
        let data: Vec<u8> = b"Hello, multipart world!".to_vec();
        upload.upload_data(&data).expect("upload failed");
        let etag = upload.complete().expect("complete failed");
        assert!(!etag.is_empty(), "ETag should not be empty");
        // ETag should contain a dash followed by part count.
        assert!(
            etag.contains('-'),
            "ETag should follow <hash>-<parts> format"
        );
    }

    #[test]
    fn test_complete_is_idempotent() {
        let config = MultipartConfig {
            chunk_size_bytes: 8,
            ..Default::default()
        };
        let mut upload = MultipartUpload::initiate("bkt", "obj", config);
        upload.upload_data(b"some data").expect("upload failed");
        let etag1 = upload.complete().expect("first complete");
        let etag2 = upload.complete().expect("second complete");
        assert_eq!(etag1, etag2, "completing twice must return the same ETag");
    }

    #[test]
    fn test_abort_clears_parts() {
        let config = MultipartConfig {
            chunk_size_bytes: 4,
            ..Default::default()
        };
        let mut upload = MultipartUpload::initiate("bkt", "obj", config);
        upload.upload_data(b"abcdefgh").expect("upload failed");
        assert!(upload.parts_uploaded() > 0);
        upload.abort();
        assert_eq!(upload.parts_uploaded(), 0);
        assert!(matches!(upload.state(), UploadState::Aborted));
    }

    #[test]
    fn test_upload_after_abort_fails() {
        let config = MultipartConfig::default();
        let mut upload = MultipartUpload::initiate("bkt", "obj", config);
        upload.abort();
        assert!(upload.upload_data(b"more data").is_err());
    }

    #[test]
    fn test_complete_with_no_parts_fails() {
        let config = MultipartConfig::default();
        let mut upload = MultipartUpload::initiate("bkt", "obj", config);
        assert!(upload.complete().is_err());
    }

    #[test]
    fn test_upload_empty_data_is_noop() {
        let config = MultipartConfig {
            chunk_size_bytes: 8,
            ..Default::default()
        };
        let mut upload = MultipartUpload::initiate("bkt", "obj", config);
        upload
            .upload_data(&[])
            .expect("empty upload should succeed");
        assert_eq!(upload.parts_uploaded(), 0);
        assert!(matches!(upload.state(), UploadState::Pending));
    }

    #[test]
    fn test_parts_sorted_by_part_number() {
        // Upload two batches; parts must be assembled in order after complete.
        let config = MultipartConfig {
            chunk_size_bytes: 4,
            ..Default::default()
        };
        let mut upload = MultipartUpload::initiate("bkt", "obj", config);
        upload.upload_data(b"abcdefgh").expect("first batch");
        upload.upload_data(b"ijklmnop").expect("second batch");
        upload.complete().expect("complete");
        let part_numbers: Vec<usize> = upload.parts().iter().map(|p| p.part_number).collect();
        let mut sorted = part_numbers.clone();
        sorted.sort_unstable();
        assert_eq!(part_numbers, sorted, "parts must be sorted after complete");
    }

    #[test]
    fn test_full_roundtrip_simulation() {
        // Simulate a 20 MiB upload with 5 MiB chunks.
        let config = MultipartConfig {
            chunk_size_bytes: 5 * 1024 * 1024,
            max_concurrency: 4,
            max_retries: 3,
        };
        let mut upload = MultipartUpload::initiate("prod-bucket", "datasets/20mb.bin", config);
        let chunk: Vec<u8> = (0u8..=255).cycle().take(5 * 1024 * 1024).collect();
        for _ in 0..4 {
            upload.upload_data(&chunk).expect("part upload");
        }
        assert_eq!(upload.parts_uploaded(), 4);
        assert_eq!(upload.total_bytes_uploaded(), 20 * 1024 * 1024);
        let etag = upload.complete().expect("complete");
        assert!(!etag.is_empty());
        assert!(matches!(upload.state(), UploadState::Completed { .. }));
    }
}
