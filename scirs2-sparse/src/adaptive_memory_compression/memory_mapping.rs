//! Memory-Mapped File Operations
//!
//! This module provides safe wrappers around memory-mapped file operations
//! for efficient large-scale data access.

use crate::error::{SparseError, SparseResult};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::path::PathBuf;

#[cfg(unix)]
use std::os::unix::fs::FileExt;

#[cfg(windows)]
use std::os::windows::fs::FileExt;

/// Memory-mapped file wrapper providing cross-platform file access
#[derive(Debug)]
pub struct MemoryMappedFile {
    filepath: PathBuf,
    file: File,
    size: usize,
    mapped: bool,
    access_count: u64,
    _phantom: PhantomData<()>,
}

/// Memory mapping configuration
#[derive(Debug, Clone)]
pub struct MemoryMappingConfig {
    /// Enable read-only mapping
    pub read_only: bool,
    /// Enable write-through mapping
    pub write_through: bool,
    /// Prefetch pages on mapping
    pub prefetch: bool,
    /// Page size hint for mapping
    pub page_size_hint: usize,
}

/// Statistics for memory mapping operations
#[derive(Debug, Clone)]
pub struct MemoryMappingStats {
    pub total_files: usize,
    pub total_mapped_size: usize,
    pub read_operations: u64,
    pub write_operations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

impl Default for MemoryMappingConfig {
    fn default() -> Self {
        Self {
            read_only: false,
            write_through: true,
            prefetch: false,
            page_size_hint: 4096, // 4KB default page size
        }
    }
}

impl MemoryMappedFile {
    /// Create a new memory-mapped file
    pub fn new(filepath: PathBuf, size: usize) -> SparseResult<Self> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&filepath)
            .map_err(|e| SparseError::Io(format!("Failed to create file {filepath:?}: {e}")))?;

        // Set file size if creating new file
        file.set_len(size as u64)
            .map_err(|e| SparseError::Io(format!("Failed to set file size: {e}")))?;

        Ok(Self {
            filepath,
            file,
            size,
            mapped: true, // We'll treat buffered I/O as "mapped" for this implementation
            access_count: 0,
            _phantom: PhantomData,
        })
    }

    /// Create memory-mapped file with configuration
    pub fn new_with_config(
        filepath: PathBuf,
        size: usize,
        config: MemoryMappingConfig,
    ) -> SparseResult<Self> {
        let mut options = OpenOptions::new();
        options.create(true).read(true);

        if !config.read_only {
            options.write(true);
        }

        let file = options
            .open(&filepath)
            .map_err(|e| SparseError::Io(format!("Failed to create file {filepath:?}: {e}")))?;

        // Set file size if creating new file
        if !config.read_only {
            file.set_len(size as u64)
                .map_err(|e| SparseError::Io(format!("Failed to set file size: {e}")))?;
        }

        Ok(Self {
            filepath,
            file,
            size,
            mapped: true,
            access_count: 0,
            _phantom: PhantomData,
        })
    }

    /// Open existing memory-mapped file
    pub fn open(filepath: PathBuf) -> SparseResult<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&filepath)
            .map_err(|e| SparseError::Io(format!("Failed to open file {filepath:?}: {e}")))?;

        let metadata = file
            .metadata()
            .map_err(|e| SparseError::Io(format!("Failed to get file metadata: {e}")))?;

        Ok(Self {
            filepath,
            file,
            size: metadata.len() as usize,
            mapped: true,
            access_count: 0,
            _phantom: PhantomData,
        })
    }

    /// Read data from the mapped file at offset
    pub fn read_at(&mut self, offset: usize, buffer: &mut [u8]) -> SparseResult<usize> {
        self.access_count += 1;

        if offset >= self.size {
            return Ok(0);
        }

        let read_size = buffer.len().min(self.size - offset);
        let buffer = &mut buffer[..read_size];

        #[cfg(unix)]
        {
            self.file
                .read_at(buffer, offset as u64)
                .map_err(|e| SparseError::Io(format!("Failed to read at offset {offset}: {e}")))
        }

        #[cfg(windows)]
        {
            self.file
                .seek_read(buffer, offset as u64)
                .map_err(|e| SparseError::Io(format!("Failed to read at offset {offset}: {e}")))
        }

        #[cfg(not(any(unix, windows)))]
        {
            // Fallback for other platforms - use regular seeking
            let mut file_clone = self
                .file
                .try_clone()
                .map_err(|e| SparseError::Io(format!("Failed to clone file handle: {e}")))?;
            file_clone
                .seek(SeekFrom::Start(offset as u64))
                .map_err(|e| SparseError::Io(format!("Failed to seek to offset {offset}: {e}")))?;
            file_clone
                .read(buffer)
                .map_err(|e| SparseError::Io(format!("Failed to read data: {e}")))
        }
    }

    /// Write data to the mapped file at offset
    pub fn write_at(&mut self, offset: usize, data: &[u8]) -> SparseResult<usize> {
        self.access_count += 1;

        if offset >= self.size {
            return Err(SparseError::Io(format!(
                "Write offset {offset} exceeds file size {}",
                self.size
            )));
        }

        let write_size = data.len().min(self.size - offset);
        let data = &data[..write_size];

        #[cfg(unix)]
        {
            self.file
                .write_at(data, offset as u64)
                .map_err(|e| SparseError::Io(format!("Failed to write at offset {offset}: {e}")))
        }

        #[cfg(windows)]
        {
            self.file
                .seek_write(data, offset as u64)
                .map_err(|e| SparseError::Io(format!("Failed to write at offset {offset}: {e}")))
        }

        #[cfg(not(any(unix, windows)))]
        {
            // Fallback for other platforms - use regular seeking
            let mut file_clone = self
                .file
                .try_clone()
                .map_err(|e| SparseError::Io(format!("Failed to clone file handle: {e}")))?;
            file_clone
                .seek(SeekFrom::Start(offset as u64))
                .map_err(|e| SparseError::Io(format!("Failed to seek to offset {offset}: {e}")))?;
            file_clone
                .write(data)
                .map_err(|e| SparseError::Io(format!("Failed to write data: {e}")))
        }
    }

    /// Read entire file contents
    pub fn read_all(&mut self) -> SparseResult<Vec<u8>> {
        let mut buffer = vec![0u8; self.size];
        let bytes_read = self.read_at(0, &mut buffer)?;
        buffer.truncate(bytes_read);
        Ok(buffer)
    }

    /// Write entire file contents
    pub fn write_all(&mut self, data: &[u8]) -> SparseResult<()> {
        if data.len() > self.size {
            // Resize file if necessary
            self.resize(data.len())?;
        }

        self.write_at(0, data)?;
        Ok(())
    }

    /// Resize the mapped file
    pub fn resize(&mut self, new_size: usize) -> SparseResult<()> {
        self.file
            .set_len(new_size as u64)
            .map_err(|e| SparseError::Io(format!("Failed to resize file: {e}")))?;
        self.size = new_size;
        Ok(())
    }

    /// Flush data to disk
    pub fn flush(&self) -> SparseResult<()> {
        self.file
            .sync_all()
            .map_err(|e| SparseError::Io(format!("Failed to flush file: {e}")))
    }

    /// Flush data to disk (metadata only)
    pub fn flush_data(&self) -> SparseResult<()> {
        self.file
            .sync_data()
            .map_err(|e| SparseError::Io(format!("Failed to flush data: {e}")))
    }

    /// Get file size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get file path
    pub fn path(&self) -> &PathBuf {
        &self.filepath
    }

    /// Check if file is mapped
    pub fn is_mapped(&self) -> bool {
        self.mapped
    }

    /// Get access count
    pub fn access_count(&self) -> u64 {
        self.access_count
    }

    /// Reset access count
    pub fn reset_access_count(&mut self) {
        self.access_count = 0;
    }

    /// Get file metadata
    pub fn metadata(&self) -> SparseResult<std::fs::Metadata> {
        self.file
            .metadata()
            .map_err(|e| SparseError::Io(format!("Failed to get metadata: {e}")))
    }

    /// Check if file exists
    pub fn exists(&self) -> bool {
        self.filepath.exists()
    }

    /// Read data in chunks for better memory efficiency
    pub fn read_chunked<F>(&mut self, chunk_size: usize, mut callback: F) -> SparseResult<()>
    where
        F: FnMut(&[u8], usize) -> SparseResult<()>,
    {
        let mut offset = 0;
        let mut buffer = vec![0u8; chunk_size];

        while offset < self.size {
            let bytes_read = self.read_at(offset, &mut buffer)?;
            if bytes_read == 0 {
                break;
            }

            callback(&buffer[..bytes_read], offset)?;
            offset += bytes_read;
        }

        Ok(())
    }

    /// Write data in chunks for better memory efficiency
    pub fn write_chunked<F>(&mut self, chunk_size: usize, mut data_provider: F) -> SparseResult<()>
    where
        F: FnMut(usize) -> SparseResult<Option<Vec<u8>>>,
    {
        let mut offset = 0;

        loop {
            match data_provider(offset)? {
                Some(chunk) => {
                    if chunk.is_empty() {
                        break;
                    }
                    let bytes_written = self.write_at(offset, &chunk)?;
                    offset += bytes_written;
                }
                None => break,
            }
        }

        Ok(())
    }

    /// Prefetch data into memory (hint to OS)
    pub fn prefetch(&self, offset: usize, length: usize) -> SparseResult<()> {
        // For now, this is a no-op since we're using buffered I/O
        // In a real implementation, this would use platform-specific prefetch hints
        let _end_offset = offset + length;
        Ok(())
    }

    /// Advise access pattern to OS
    pub fn advise_access_pattern(&self, pattern: AccessPattern) -> SparseResult<()> {
        // Platform-specific advice would go here
        // For now, this is a no-op
        let _pattern = pattern;
        Ok(())
    }
}

/// Access pattern hints for memory mapping
#[derive(Debug, Clone, Copy)]
pub enum AccessPattern {
    /// Sequential access pattern
    Sequential,
    /// Random access pattern
    Random,
    /// Will need data soon
    WillNeed,
    /// Won't need data anymore
    DontNeed,
}

/// Memory mapping manager for multiple files
#[derive(Debug)]
pub struct MemoryMappingManager {
    mapped_files: std::collections::HashMap<PathBuf, MemoryMappedFile>,
    config: MemoryMappingConfig,
    stats: MemoryMappingStats,
}

impl MemoryMappingManager {
    /// Create new memory mapping manager
    pub fn new(config: MemoryMappingConfig) -> Self {
        Self {
            mapped_files: std::collections::HashMap::new(),
            config,
            stats: MemoryMappingStats {
                total_files: 0,
                total_mapped_size: 0,
                read_operations: 0,
                write_operations: 0,
                cache_hits: 0,
                cache_misses: 0,
            },
        }
    }

    /// Map a file
    pub fn map_file(&mut self, filepath: PathBuf, size: usize) -> SparseResult<()> {
        if self.mapped_files.contains_key(&filepath) {
            return Ok(()); // Already mapped
        }

        let mapped_file =
            MemoryMappedFile::new_with_config(filepath.clone(), size, self.config.clone())?;

        self.stats.total_files += 1;
        self.stats.total_mapped_size += size;

        self.mapped_files.insert(filepath, mapped_file);
        Ok(())
    }

    /// Unmap a file
    pub fn unmap_file(&mut self, filepath: &PathBuf) -> SparseResult<()> {
        if let Some(mapped_file) = self.mapped_files.remove(filepath) {
            self.stats.total_files -= 1;
            self.stats.total_mapped_size -= mapped_file.size();
            mapped_file.flush()?;
        }
        Ok(())
    }

    /// Get mapped file reference
    pub fn get_file(&mut self, filepath: &PathBuf) -> Option<&mut MemoryMappedFile> {
        self.mapped_files.get_mut(filepath)
    }

    /// Get statistics
    pub fn get_stats(&self) -> &MemoryMappingStats {
        &self.stats
    }

    /// Flush all mapped files
    pub fn flush_all(&self) -> SparseResult<()> {
        for mapped_file in self.mapped_files.values() {
            mapped_file.flush()?;
        }
        Ok(())
    }

    /// Close all mapped files
    pub fn close_all(&mut self) -> SparseResult<()> {
        self.flush_all()?;
        self.mapped_files.clear();
        self.stats.total_files = 0;
        self.stats.total_mapped_size = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_memory_mapped_file_creation() {
        let temp_dir = TempDir::new().unwrap();
        let filepath = temp_dir.path().join("test.dat");

        let mapped_file = MemoryMappedFile::new(filepath, 1024);
        assert!(mapped_file.is_ok());

        let mapped_file = mapped_file.unwrap();
        assert_eq!(mapped_file.size(), 1024);
        assert!(mapped_file.is_mapped());
    }

    #[test]
    fn test_read_write_operations() {
        let temp_dir = TempDir::new().unwrap();
        let filepath = temp_dir.path().join("test.dat");

        let mut mapped_file = MemoryMappedFile::new(filepath, 1024).unwrap();

        // Write some data
        let write_data = b"Hello, World!";
        let bytes_written = mapped_file.write_at(0, write_data).unwrap();
        assert_eq!(bytes_written, write_data.len());

        // Read it back
        let mut read_buffer = vec![0u8; write_data.len()];
        let bytes_read = mapped_file.read_at(0, &mut read_buffer).unwrap();
        assert_eq!(bytes_read, write_data.len());
        assert_eq!(&read_buffer, write_data);
    }

    #[test]
    fn test_file_resize() {
        let temp_dir = TempDir::new().unwrap();
        let filepath = temp_dir.path().join("test.dat");

        let mut mapped_file = MemoryMappedFile::new(filepath, 1024).unwrap();
        assert_eq!(mapped_file.size(), 1024);

        // Resize to larger
        mapped_file.resize(2048).unwrap();
        assert_eq!(mapped_file.size(), 2048);

        // Resize to smaller
        mapped_file.resize(512).unwrap();
        assert_eq!(mapped_file.size(), 512);
    }

    #[test]
    fn test_chunked_operations() {
        let temp_dir = TempDir::new().unwrap();
        let filepath = temp_dir.path().join("test.dat");

        let mut mapped_file = MemoryMappedFile::new(filepath, 1024).unwrap();

        // Write chunked data
        let test_data = b"This is a test string for chunked operations";
        let chunk_size = 10;
        let mut offset = 0;

        mapped_file
            .write_chunked(chunk_size, |current_offset| {
                if current_offset != offset {
                    return Ok(None);
                }

                let start = current_offset;
                let end = (start + chunk_size).min(test_data.len());

                if start >= test_data.len() {
                    Ok(None)
                } else {
                    offset = end;
                    Ok(Some(test_data[start..end].to_vec()))
                }
            })
            .unwrap();

        // Read back and verify
        let mut read_data = Vec::new();
        mapped_file
            .read_chunked(chunk_size, |chunk, _offset| {
                read_data.extend_from_slice(chunk);
                Ok(())
            })
            .unwrap();

        assert_eq!(&read_data[..test_data.len()], test_data);
    }

    #[test]
    fn test_memory_mapping_manager() {
        let temp_dir = TempDir::new().unwrap();
        let config = MemoryMappingConfig::default();
        let mut manager = MemoryMappingManager::new(config);

        // Map a file
        let filepath = temp_dir.path().join("test.dat");
        manager.map_file(filepath.clone(), 1024).unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.total_files, 1);
        assert_eq!(stats.total_mapped_size, 1024);

        // Unmap the file
        manager.unmap_file(&filepath).unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.total_files, 0);
        assert_eq!(stats.total_mapped_size, 0);
    }

    #[test]
    fn test_access_count_tracking() {
        let temp_dir = TempDir::new().unwrap();
        let filepath = temp_dir.path().join("test.dat");

        let mut mapped_file = MemoryMappedFile::new(filepath, 1024).unwrap();
        assert_eq!(mapped_file.access_count(), 0);

        // Perform some operations
        let mut buffer = vec![0u8; 10];
        mapped_file.read_at(0, &mut buffer).unwrap();
        assert_eq!(mapped_file.access_count(), 1);

        mapped_file.write_at(0, b"test").unwrap();
        assert_eq!(mapped_file.access_count(), 2);

        // Reset counter
        mapped_file.reset_access_count();
        assert_eq!(mapped_file.access_count(), 0);
    }
}
