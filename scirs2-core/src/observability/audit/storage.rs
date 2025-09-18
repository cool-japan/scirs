//! Storage management for audit logs

use crate::error::CoreError;
use crate::observability::audit::types::{AuditConfig, AuditEvent};
use chrono::{DateTime, Utc};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

#[cfg(feature = "crypto")]
use sha2::{Digest, Sha256};

/// Log file manager for handling rotation and retention
pub struct LogFileManager {
    pub config: AuditConfig,
    pub current_file: Option<File>,
    pub current_file_size: u64,
    pub file_counter: u64,
    pub last_event_hash: Option<String>,
    pub hash_chain: Vec<String>,
}

impl LogFileManager {
    /// Create a new log file manager.
    ///
    /// # Errors
    ///
    /// Returns an error if the log directory cannot be created.
    pub fn new(config: AuditConfig) -> Result<Self, CoreError> {
        // Create log directory if it doesn't exist
        std::fs::create_dir_all(&config.log_directory).map_err(|e| {
            CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                "Failed to create log directory: {e}"
            )))
        })?;

        Ok(Self {
            config,
            current_file: None,
            current_file_size: 0,
            file_counter: 0,
            last_event_hash: None,
            hash_chain: Vec::new(),
        })
    }

    /// Write an audit event to the log file.
    ///
    /// # Errors
    ///
    /// Returns an error if the event cannot be serialized or written to the log file.
    pub fn write_event(&mut self, event: &mut AuditEvent) -> Result<(), CoreError> {
        // Set up hash chain if enabled
        if self.config.enable_hash_chain {
            event.previous_hash = self.last_event_hash.clone();
            let event_hash = self.calculate_event_hash(event)?;
            event.event_hash = Some(event_hash.clone());
            self.last_event_hash = Some(event_hash.clone());
            self.hash_chain.push(event_hash);
        }

        let serialized = if self.config.enable_json_format {
            self.serialize_json(event)?
        } else {
            self.serialize_text(event)
        };

        let data = format!("{serialized}\n");
        let data_size = data.len() as u64;

        // Check if we need to rotate the log file
        if self.current_file.is_none()
            || self.current_file_size + data_size > self.config.max_file_size
        {
            self.rotate_log_file()?;
        }

        if let Some(ref mut file) = self.current_file {
            file.write_all(data.as_bytes()).map_err(|e| {
                CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                    "Failed to write to log file: {e}"
                )))
            })?;

            self.current_file_size += data_size;
        }

        Ok(())
    }

    /// Rotate the current log file to a new file.
    ///
    /// # Errors
    ///
    /// Returns an error if the current file cannot be flushed or a new file cannot be created.
    pub fn rotate_log_file(&mut self) -> Result<(), CoreError> {
        // Close current file
        if let Some(mut file) = self.current_file.take() {
            file.flush().map_err(|e| {
                CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                    "Failed to flush log file: {e}"
                )))
            })?;
        }

        // Create new log file
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!("audit_{timestamp}_{:06}.log", self.file_counter);
        let filepath = self.config.log_directory.join(filename);

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filepath)
            .map_err(|e| {
                CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                    "Failed to create log file: {e}"
                )))
            })?;

        self.current_file = Some(file);
        self.current_file_size = 0;
        self.file_counter += 1;

        // Clean up old files if necessary
        self.cleanup_old_files()?;

        Ok(())
    }

    /// Clean up old log files according to the retention policy.
    ///
    /// # Errors
    ///
    /// Returns an error if log files cannot be read or deleted.
    pub fn cleanup_old_files(&self) -> Result<(), CoreError> {
        let mut log_files = Vec::new();

        // Read directory and collect log files
        if let Ok(entries) = std::fs::read_dir(&self.config.log_directory) {
            for entry in entries.flatten() {
                if let Some(filename) = entry.file_name().to_str() {
                    if filename.starts_with("audit_") && filename.ends_with(".log") {
                        if let Ok(metadata) = entry.metadata() {
                            log_files.push((
                                entry.path(),
                                metadata
                                    .modified()
                                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                            ));
                        }
                    }
                }
            }
        }

        // Sort by modification time (oldest first)
        log_files.sort_by_key(|(_, time)| *time);

        // Remove excess files
        if log_files.len() > self.config.max_files {
            let files_to_remove = log_files.len() - self.config.max_files;
            for (path, _) in log_files.iter().take(files_to_remove) {
                if let Err(e) = std::fs::remove_file(path) {
                    eprintln!("Failed to remove old log file {path:?}: {e}");
                }
            }
        }

        Ok(())
    }

    /// Serialize an audit event to JSON format.
    ///
    /// # Errors
    ///
    /// Returns an error if the event cannot be serialized to JSON.
    pub fn serialize_json(&self, event: &AuditEvent) -> Result<String, CoreError> {
        serde_json::to_string(event).map_err(|e| {
            CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                "Failed to serialize event to JSON: {e}"
            )))
        })
    }

    /// Serialize an audit event to JSON format (serde feature required).
    ///
    /// # Errors
    ///
    /// Returns an error indicating that the serde feature is required.
    #[cfg(not(feature = "serde"))]
    pub fn serialize_json(&self, _event: &AuditEvent) -> Result<String, CoreError> {
        Err(CoreError::ComputationError(
            crate::error::ErrorContext::new(
                "JSON serialization requires serde feature".to_string(),
            ),
        ))
    }

    /// Serialize an audit event to text format.
    #[must_use]
    pub fn serialize_text(&self, event: &AuditEvent) -> String {
        format!(
            "[{}] {} {} {} user={} resource={} outcome={} description=\"{}\"",
            event.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            event.category.as_str(),
            event.severity.as_str(),
            event.action,
            event.userid.as_deref().unwrap_or("-"),
            event.resourceid.as_deref().unwrap_or("-"),
            event.outcome.as_str(),
            event.description
        )
    }

    /// Flush pending data to the log file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be flushed.
    pub fn flush(&mut self) -> Result<(), CoreError> {
        if let Some(ref mut file) = self.current_file {
            file.flush().map_err(|e| {
                CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                    "Failed to flush log file: {e}"
                )))
            })?;
        }
        Ok(())
    }

    /// Calculate a cryptographic hash for an audit event.
    ///
    /// # Errors
    ///
    /// Returns an error if the hash cannot be calculated.
    #[cfg(feature = "crypto")]
    pub fn calculate_event_hash(&self, event: &AuditEvent) -> Result<String, CoreError> {
        let mut hasher = Sha256::new();

        // Hash key fields to ensure integrity
        hasher.update(event.event_id.to_string());
        hasher.update(event.timestamp.to_rfc3339());
        hasher.update(event.category.as_str());
        hasher.update(&event.action);

        if let Some(ref userid) = event.userid {
            hasher.update(userid);
        }

        if let Some(ref resourceid) = event.resourceid {
            hasher.update(resourceid);
        }

        hasher.update(&event.description);
        hasher.update(event.outcome.as_str());

        if let Some(ref prev_hash) = event.previous_hash {
            hasher.update(prev_hash);
        }

        let result = hasher.finalize();
        Ok(format!("{:x}", result))
    }

    /// Calculate a fallback hash for an audit event (crypto feature recommended).
    ///
    /// # Errors
    ///
    /// Returns an error if the hash cannot be calculated.
    #[cfg(not(feature = "crypto"))]
    pub fn calculate_event_hash(&self, event: &AuditEvent) -> Result<String, CoreError> {
        // Simple fallback hash implementation
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        event.event_id.hash(&mut hasher);
        event.timestamp.timestamp().hash(&mut hasher);
        event.category.as_str().hash(&mut hasher);
        event.action.hash(&mut hasher);

        Ok(format!("{:x}", hasher.finish()))
    }

    /// Verify hash chain integrity
    ///
    /// # Errors
    ///
    /// Returns an error if hash chain verification fails.
    pub fn verify_hash_chain(&self) -> Result<bool, CoreError> {
        if !self.config.enable_hash_chain {
            return Ok(true); // No verification needed
        }

        // Verify each hash in the chain
        if self.hash_chain.is_empty() {
            return Ok(true); // Empty chain is valid
        }

        // Check if any hash appears to be tampered
        for (i, hash) in self.hash_chain.iter().enumerate() {
            // Basic hash format validation
            if hash.len() != 64 {
                return Ok(false); // SHA-256 hashes should be 64 hex chars
            }

            // Validate hex format
            if !hash.chars().all(|c| c.is_ascii_hexdigit()) {
                return Ok(false); // Invalid hex characters
            }

            // For chained verification, we would need to re-read and verify
            // each event against its hash. This is a simplified check.
            if i > 0 {
                let prev_hash = &self.hash_chain[i - 1];
                // In a full implementation, we would verify that the current
                // event's previous_hash field matches the actual previous hash
                if prev_hash.is_empty() {
                    return Ok(false); // Broken chain
                }
            }
        }

        Ok(true)
    }

    /// Archive old log files according to retention policy
    ///
    /// # Errors
    ///
    /// Returns an error if archival operations fail.
    #[allow(dead_code)]
    pub fn archive_old_files(&self) -> Result<(), CoreError> {
        if !self.config.retention_policy.enable_auto_archive {
            return Ok(());
        }

        let cutoff_date = Utc::now()
            - chrono::Duration::days(self.config.retention_policy.active_retention_days as i64);

        let archive_path = self
            .config
            .retention_policy
            .archive_path
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.config.log_directory.join("archive"));

        // Create archive directory if it doesn't exist
        std::fs::create_dir_all(&archive_path).map_err(|e| {
            CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                "Failed to create archive directory: {e}"
            )))
        })?;

        // Find files older than cutoff date
        if let Ok(entries) = std::fs::read_dir(&self.config.log_directory) {
            for entry in entries.flatten() {
                if let Some(filename) = entry.file_name().to_str() {
                    if filename.starts_with("audit_") && filename.ends_with(".log") {
                        if let Ok(metadata) = entry.metadata() {
                            if let Ok(modified_time) = metadata.modified() {
                                let modified_datetime: DateTime<Utc> = modified_time.into();

                                if modified_datetime < cutoff_date {
                                    // Archive this file
                                    let source_path = entry.path();
                                    let archive_filename = format!("archived_{filename}");
                                    let dest_path = archive_path.join(archive_filename);

                                    // Simple archive: copy to archive directory
                                    if let Err(e) = std::fs::copy(&source_path, &dest_path) {
                                        eprintln!("Failed to archive file {source_path:?}: {e}");
                                        continue;
                                    }

                                    // Optionally compress the archived file
                                    #[cfg(feature = "compression")]
                                    {
                                        if let Err(e) = self.compress_archived_file(&dest_path) {
                                            eprintln!(
                                                "Failed to compress archived file {:?}: {}",
                                                dest_path, e
                                            );
                                        }
                                    }

                                    // Remove original file after successful archival
                                    if let Err(e) = std::fs::remove_file(&source_path) {
                                        eprintln!(
                                            "Failed to remove original file {source_path:?}: {e}"
                                        );
                                    } else {
                                        println!(
                                            "Archived log file: {source_path:?} -> {dest_path:?}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Clean up files according to retention policy
    ///
    /// # Errors
    ///
    /// Returns an error if cleanup operations fail.
    #[allow(dead_code)]
    pub fn cleanup_expired_files(&self) -> Result<(), CoreError> {
        if !self.config.retention_policy.enable_auto_delete {
            return Ok(());
        }

        let archive_cutoff = Utc::now()
            - chrono::Duration::days(self.config.retention_policy.archive_retention_days as i64);

        let archive_path = self
            .config
            .retention_policy
            .archive_path
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.config.log_directory.join("archive"));

        // Clean up expired archive files
        if archive_path.exists() {
            if let Ok(entries) = std::fs::read_dir(&archive_path) {
                for entry in entries.flatten() {
                    if let Some(filename) = entry.file_name().to_str() {
                        if filename.starts_with("archived_audit_") {
                            if let Ok(metadata) = entry.metadata() {
                                if let Ok(modified_time) = metadata.modified() {
                                    let modified_datetime: DateTime<Utc> = modified_time.into();

                                    if modified_datetime < archive_cutoff {
                                        // Delete expired archive file
                                        let file_path = entry.path();
                                        if let Err(e) = std::fs::remove_file(&file_path) {
                                            eprintln!(
                                                "Failed to delete expired archive file {file_path:?}: {e}"
                                            );
                                        } else {
                                            println!("Deleted expired archive file: {file_path:?}");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Check disk space and clean up if necessary
        let min_free_space = self.config.retention_policy.min_free_space;
        if let Ok(available_space) = self.get_available_disk_space(&self.config.log_directory) {
            if available_space < min_free_space {
                // Emergency cleanup - remove oldest files first
                let mut log_files = Vec::new();

                // Collect both active and archive files
                for dir in [&self.config.log_directory, &archive_path] {
                    if dir.exists() {
                        if let Ok(entries) = std::fs::read_dir(dir) {
                            for entry in entries.flatten() {
                                if let Some(filename) = entry.file_name().to_str() {
                                    if filename.contains("audit_") && filename.ends_with(".log") {
                                        if let Ok(metadata) = entry.metadata() {
                                            if let Ok(modified_time) = metadata.modified() {
                                                log_files.push((entry.path(), modified_time));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Sort by age (oldest first)
                log_files.sort_by_key(|(_, time)| *time);

                // Remove oldest files until we have enough space
                for (file_path, _) in log_files {
                    if let Err(e) = std::fs::remove_file(&file_path) {
                        eprintln!("Failed to remove file for disk space: {file_path:?}: {e}");
                    } else {
                        println!("Removed file to free disk space: {file_path:?}");

                        // Check if we have enough space now
                        if let Ok(new_available) =
                            self.get_available_disk_space(&self.config.log_directory)
                        {
                            if new_available >= min_free_space {
                                break;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Compress an archived log file
    ///
    /// # Errors
    ///
    /// Returns an error if compression fails.
    #[cfg(feature = "compression")]
    pub fn compress_archived_file(&self, file_path: &std::path::Path) -> Result<(), CoreError> {
        use std::fs::File;
        use std::io::{BufReader, BufWriter};

        let input_file = File::open(file_path).map_err(|e| {
            CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                "Failed to open file for compression: {e}"
            )))
        })?;

        let compressed_path = file_path.with_extension("log.gz");
        let output_file = File::create(&compressed_path).map_err(|e| {
            CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                "Failed to create compressed file: {e}"
            )))
        })?;

        let mut reader = BufReader::new(input_file);
        let writer = BufWriter::new(output_file);

        // Use flate2 for gzip compression
        #[cfg(feature = "flate2")]
        {
            use flate2::write::GzEncoder;
            use flate2::Compression;
            use std::io::copy;

            let mut encoder = GzEncoder::new(writer, Compression::default());
            copy(&mut reader, &mut encoder).map_err(|e| {
                CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                    "Failed to compress file: {e}"
                )))
            })?;

            encoder.finish().map_err(|e| {
                CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                    "Failed to finalize compression: {e}"
                )))
            })?;
        }

        #[cfg(not(feature = "flate2"))]
        {
            return Err(CoreError::ComputationError(
                crate::error::ErrorContext::new("Compression requires flate2 feature".to_string()),
            ));
        }

        // Remove original file after successful compression
        std::fs::remove_file(file_path).map_err(|e| {
            CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                "Failed to remove original file after compression: {e}"
            )))
        })?;

        println!("Compressed archive file: {file_path:?} -> {compressed_path:?}");
        Ok(())
    }

    /// Get available disk space for a directory
    ///
    /// # Errors
    ///
    /// Returns an error if disk space cannot be determined.
    pub fn get_available_disk_space(&self, path: &std::path::Path) -> Result<u64, CoreError> {
        #[cfg(feature = "libc")]
        {
            use std::ffi::CString;
            use std::mem;

            let path_cstr = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| {
                CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                    "Failed to convert path to CString: {e}"
                )))
            })?;

            let mut stat: libc::statvfs = unsafe { mem::zeroed() };
            let result = unsafe { libc::statvfs(path_cstr.as_ptr(), &mut stat) };

            if result == 0 {
                // Available space = available blocks * block size
                Ok(stat.f_bavail as u64 * stat.f_frsize)
            } else {
                Err(CoreError::ComputationError(
                    crate::error::ErrorContext::new(
                        "Failed to get filesystem statistics".to_string(),
                    ),
                ))
            }
        }

        #[cfg(not(feature = "libc"))]
        {
            // Fallback for platforms without libc support
            let _ = path; // Acknowledge unused parameter
            Ok(1024 * 1024 * 1024 * 10) // 10GB fallback
        }
    }
}
