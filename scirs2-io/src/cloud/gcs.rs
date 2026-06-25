//! GCS Resumable Upload state machine (simulation mode).
//!
//! Implements the Google Cloud Storage resumable upload protocol as a pure-Rust
//! state machine.  When the `google-cloud-storage` Cargo feature is enabled a
//! real HTTP implementation (OAuth2 + service-account JSON credentials) would be
//! wired in.  Without the feature every operation executes in a local simulation
//! mode suitable for testing and offline development.
//!
//! # Protocol outline (real GCS)
//!
//! 1. POST to `https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o`
//!    with `uploadType=resumable` → receive a `Location` header containing the
//!    **resumable session URI**.
//! 2. PUT each byte range to the session URI using a `Content-Range` header.
//! 3. The final PUT (with `Content-Range: bytes N-M/TOTAL`) returns `200 OK`
//!    and the completed object JSON.
//! 4. Query progress at any time with a zero-length PUT and a
//!    `Content-Range: bytes */{total_or_*}` header → `308 Resume Incomplete`
//!    with `Range: bytes=0-{last_received}`.
//! 5. Abort by sending a DELETE to the session URI.
//!
//! # Simulation mode
//!
//! In simulation mode no network calls are made.  An in-memory `Vec<u8>` stores
//! the assembled object data.  The `upload_id` plays the role of the resumable
//! session URI (it is a UUID-v4-like string).
//!
//! # Example
//!
//! ```rust
//! use scirs2_io::cloud::gcs::{GcsResumableUpload, UploadStatus};
//!
//! let mut upload = GcsResumableUpload::initiate("my-bucket", "data/file.bin", Some(6));
//! assert!(!upload.upload_id().is_empty());
//!
//! let status = upload.upload_chunk(0, b"hello ").expect("chunk");
//! assert!(matches!(status, UploadStatus::Complete { total_bytes: 6 }));
//!
//! let total = upload.finalize().expect("finalize");
//! assert_eq!(total, 6);
//! assert_eq!(upload.assembled_data(), b"hello ");
//! ```

use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by GCS resumable upload operations.
#[derive(Debug, thiserror::Error)]
pub enum GcsError {
    /// The upload session has been aborted; no further operations are possible.
    #[error("upload aborted")]
    Aborted,
    /// The upload has already been finalized; no further mutations are allowed.
    #[error("already finalized")]
    AlreadyFinalized,
    /// The supplied byte offset does not match the server's received byte count.
    ///
    /// GCS requires that each PUT starts exactly where the previous one ended.
    #[error("chunk out of range: offset {offset} but server has {server_bytes} bytes")]
    ChunkOutOfRange {
        /// The offset supplied by the caller.
        offset: usize,
        /// The number of bytes the server (simulation) has received.
        server_bytes: usize,
    },
    /// The total uploaded size differs from the value declared at initiation.
    #[error("size mismatch: declared {declared}, uploaded {uploaded}")]
    SizeMismatch {
        /// Size declared at `initiate`.
        declared: usize,
        /// Actual bytes uploaded before `finalize` was called.
        uploaded: usize,
    },
}

// ---------------------------------------------------------------------------
// UploadStatus
// ---------------------------------------------------------------------------

/// Status returned after uploading a chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UploadStatus {
    /// Upload is still in progress.
    Incomplete {
        /// Total bytes received so far (cumulative across all chunks).
        bytes_received: usize,
    },
    /// Upload is fully streamed (all declared bytes received).
    Complete {
        /// Total bytes in the object.
        total_bytes: usize,
    },
}

// ---------------------------------------------------------------------------
// GcsResumableUpload
// ---------------------------------------------------------------------------

/// Simulation-mode GCS resumable upload session.
///
/// Holds all in-memory state for a single upload.  Chunks must be delivered in
/// order (each chunk's `offset` must equal `query_status()`).
pub struct GcsResumableUpload {
    /// Simulated resumable session URI (UUID-like identifier).
    upload_id: String,
    /// Destination object name within the bucket.
    object_name: String,
    /// Destination bucket name.
    bucket: String,
    /// Uploaded chunks stored in the order they were received.
    chunks: Vec<Vec<u8>>,
    /// Optional total object size declared at initiation (in bytes).
    total_size: Option<usize>,
    /// Whether the upload has been finalized.
    finalized: bool,
    /// Whether the upload has been aborted.
    aborted: bool,
}

impl GcsResumableUpload {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Initiate a new resumable upload session.
    ///
    /// `total_size` is the number of bytes that will be uploaded in total, or
    /// `None` if the size is not known in advance.
    pub fn initiate(bucket: &str, object_name: &str, total_size: Option<usize>) -> Self {
        Self {
            upload_id: generate_session_id(),
            object_name: object_name.to_owned(),
            bucket: bucket.to_owned(),
            chunks: Vec::new(),
            total_size,
            finalized: false,
            aborted: false,
        }
    }

    // -----------------------------------------------------------------------
    // Mutating operations
    // -----------------------------------------------------------------------

    /// Upload a chunk starting at byte `offset`.
    ///
    /// `offset` **must** equal the number of bytes already received
    /// (`self.query_status()`).  GCS enforces this invariant so that chunks
    /// are always contiguous — there are no holes in the object.
    ///
    /// Returns `UploadStatus::Complete` when all `total_size` bytes (if
    /// declared) have been received, otherwise returns
    /// `UploadStatus::Incomplete`.
    ///
    /// # Errors
    ///
    /// - [`GcsError::Aborted`] — session has been aborted.
    /// - [`GcsError::AlreadyFinalized`] — session is already closed.
    /// - [`GcsError::ChunkOutOfRange`] — `offset` does not match `query_status()`.
    pub fn upload_chunk(&mut self, offset: usize, data: &[u8]) -> Result<UploadStatus, GcsError> {
        self.guard_active()?;

        let server_bytes = self.query_status();
        if offset != server_bytes {
            return Err(GcsError::ChunkOutOfRange {
                offset,
                server_bytes,
            });
        }

        if !data.is_empty() {
            self.chunks.push(data.to_vec());
        }

        let bytes_received = self.query_status();
        match self.total_size {
            Some(total) if bytes_received >= total => Ok(UploadStatus::Complete {
                total_bytes: bytes_received,
            }),
            _ => Ok(UploadStatus::Incomplete { bytes_received }),
        }
    }

    /// Finalize the upload and return the total number of bytes.
    ///
    /// If `total_size` was declared at initiation the method verifies that the
    /// uploaded byte count matches; returns [`GcsError::SizeMismatch`] if not.
    ///
    /// # Errors
    ///
    /// - [`GcsError::Aborted`]
    /// - [`GcsError::AlreadyFinalized`]
    /// - [`GcsError::SizeMismatch`]
    pub fn finalize(&mut self) -> Result<usize, GcsError> {
        self.guard_active()?;

        let uploaded = self.query_status();

        if let Some(declared) = self.total_size {
            if uploaded != declared {
                return Err(GcsError::SizeMismatch { declared, uploaded });
            }
        }

        self.finalized = true;
        Ok(uploaded)
    }

    /// Abort the upload, discarding all buffered data.
    ///
    /// After a successful abort the session is permanently closed; any
    /// subsequent call will return [`GcsError::Aborted`].
    ///
    /// # Errors
    ///
    /// Returns [`GcsError::AlreadyFinalized`] if the upload was already
    /// successfully finalized.
    pub fn abort(&mut self) -> Result<(), GcsError> {
        if self.finalized {
            return Err(GcsError::AlreadyFinalized);
        }
        self.chunks.clear();
        self.aborted = true;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Query / read operations
    // -----------------------------------------------------------------------

    /// Return the total number of bytes received so far.
    ///
    /// This mirrors the GCS `308 Resume Incomplete` response, which includes a
    /// `Range: bytes=0-{last_received}` header.
    pub fn query_status(&self) -> usize {
        self.chunks.iter().map(|c| c.len()).sum()
    }

    /// Assemble and return the full object data from all uploaded chunks.
    ///
    /// The result is the concatenation of all chunks in the order they were
    /// uploaded.  Returns an empty `Vec` if no chunks have been uploaded.
    pub fn assembled_data(&self) -> Vec<u8> {
        let total = self.query_status();
        let mut out = Vec::with_capacity(total);
        for chunk in &self.chunks {
            out.extend_from_slice(chunk);
        }
        out
    }

    /// Return the simulated session URI / upload ID.
    pub fn upload_id(&self) -> &str {
        &self.upload_id
    }

    /// Return the destination bucket name.
    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    /// Return the destination object name.
    pub fn object_name(&self) -> &str {
        &self.object_name
    }

    /// Return `true` if the upload has been successfully finalized.
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// Return `true` if the upload has been aborted.
    pub fn is_aborted(&self) -> bool {
        self.aborted
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Return an error if the session is in a terminal state.
    fn guard_active(&self) -> Result<(), GcsError> {
        if self.aborted {
            return Err(GcsError::Aborted);
        }
        if self.finalized {
            return Err(GcsError::AlreadyFinalized);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a unique session identifier (simulates a resumable session URI).
///
/// Uses a monotonic counter combined with the current nanosecond timestamp to
/// produce a 48-hex-character string that is unique within a process.
fn generate_session_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("gcs-sim-{ts:032x}{count:016x}")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Initiate → upload single chunk → finalize → assembled_data correct.
    #[test]
    fn test_single_chunk_roundtrip() {
        let data = b"hello, gcs!";
        let mut upload =
            GcsResumableUpload::initiate("my-bucket", "obj/data.bin", Some(data.len()));

        assert!(!upload.upload_id().is_empty());
        assert_eq!(upload.bucket(), "my-bucket");
        assert_eq!(upload.object_name(), "obj/data.bin");

        let status = upload.upload_chunk(0, data).expect("upload_chunk");
        assert_eq!(
            status,
            UploadStatus::Complete {
                total_bytes: data.len()
            }
        );

        let total = upload.finalize().expect("finalize");
        assert_eq!(total, data.len());
        assert_eq!(upload.assembled_data(), data);
        assert!(upload.is_finalized());
    }

    /// Declaring a total_size and uploading a different size returns SizeMismatch.
    #[test]
    fn test_size_mismatch_on_finalize() {
        let mut upload = GcsResumableUpload::initiate("b", "k", Some(100));
        upload
            .upload_chunk(0, b"only 9 bytes")
            .expect("upload chunk");
        let err = upload.finalize().expect_err("should fail");
        assert!(matches!(err, GcsError::SizeMismatch { declared: 100, .. }));
    }

    /// After abort, further upload_chunk and finalize calls return errors.
    #[test]
    fn test_operations_after_abort_fail() {
        let mut upload = GcsResumableUpload::initiate("b", "k", None);
        upload.upload_chunk(0, b"some data").expect("first chunk");
        upload.abort().expect("abort");

        assert!(upload.is_aborted());

        let err_chunk = upload.upload_chunk(9, b"more").expect_err("should fail");
        assert!(matches!(err_chunk, GcsError::Aborted));

        let err_finalize = upload.finalize().expect_err("should fail");
        assert!(matches!(err_finalize, GcsError::Aborted));
    }

    /// Operations after finalize return AlreadyFinalized.
    #[test]
    fn test_operations_after_finalize_fail() {
        let data = b"finalized";
        let mut upload = GcsResumableUpload::initiate("b", "k", Some(data.len()));
        upload.upload_chunk(0, data).expect("upload");
        upload.finalize().expect("first finalize");

        let err_upload = upload
            .upload_chunk(data.len(), b"x")
            .expect_err("should fail");
        assert!(matches!(err_upload, GcsError::AlreadyFinalized));

        let err_finalize = upload.finalize().expect_err("should fail");
        assert!(matches!(err_finalize, GcsError::AlreadyFinalized));

        // Abort after finalize also errors.
        let err_abort = upload
            .abort()
            .expect_err("abort after finalize should fail");
        assert!(matches!(err_abort, GcsError::AlreadyFinalized));
    }

    /// query_status returns cumulative bytes uploaded so far.
    #[test]
    fn test_query_status_tracks_bytes() {
        let mut upload = GcsResumableUpload::initiate("b", "k", None);
        assert_eq!(upload.query_status(), 0);

        upload.upload_chunk(0, b"abc").expect("chunk 1");
        assert_eq!(upload.query_status(), 3);

        upload.upload_chunk(3, b"defgh").expect("chunk 2");
        assert_eq!(upload.query_status(), 8);
    }

    /// Two sequential chunks are assembled in order.
    #[test]
    fn test_two_chunks_assemble_in_order() {
        let mut upload = GcsResumableUpload::initiate("b", "k", None);

        let part1 = b"FIRST_";
        let part2 = b"SECOND";
        upload.upload_chunk(0, part1).expect("chunk 1");
        upload.upload_chunk(part1.len(), part2).expect("chunk 2");

        upload.finalize().expect("finalize");

        let assembled = upload.assembled_data();
        let expected: Vec<u8> = [part1.as_ref(), part2.as_ref()].concat();
        assert_eq!(assembled, expected);
    }

    /// Out-of-order chunk (wrong offset) returns ChunkOutOfRange.
    #[test]
    fn test_wrong_offset_returns_chunk_out_of_range() {
        let mut upload = GcsResumableUpload::initiate("b", "k", None);
        upload.upload_chunk(0, b"first").expect("chunk 1");

        // Wrong offset — should be 5.
        let err = upload.upload_chunk(99, b"second").expect_err("should fail");
        assert!(matches!(
            err,
            GcsError::ChunkOutOfRange {
                offset: 99,
                server_bytes: 5
            }
        ));
    }

    /// Unknown total size: can finalize at any point with no mismatch check.
    #[test]
    fn test_unknown_total_size_finalize() {
        let mut upload = GcsResumableUpload::initiate("b", "k", None);
        upload.upload_chunk(0, b"anything").expect("chunk");
        let total = upload.finalize().expect("finalize");
        assert_eq!(total, 8);
    }

    /// Multiple chunks: UploadStatus::Incomplete until all bytes arrive.
    #[test]
    fn test_incomplete_status_until_last_chunk() {
        let total_size = 10usize;
        let mut upload = GcsResumableUpload::initiate("b", "k", Some(total_size));

        let s1 = upload.upload_chunk(0, b"hello").expect("chunk 1");
        assert_eq!(s1, UploadStatus::Incomplete { bytes_received: 5 });

        let s2 = upload.upload_chunk(5, b"world").expect("chunk 2");
        assert_eq!(s2, UploadStatus::Complete { total_bytes: 10 });
    }
}
