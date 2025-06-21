//! I/O utilities for ``SciRS2``
//!
//! This module provides utilities for input/output operations.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::error::CoreResult;

/// Opens a file for reading
///
/// # Arguments
///
/// * `path` - Path to the file
///
/// # Returns
///
/// * `Ok(BufReader<File>)` if the file was opened successfully
/// * `Err(CoreError::IoError)` if the file could not be opened
///
/// # Errors
/// Returns `CoreError::IoError` if the file could not be opened.
pub fn open_file<P: AsRef<Path>>(path: P) -> CoreResult<BufReader<File>> {
    let file = File::open(path.as_ref())?;
    Ok(BufReader::new(file))
}

/// Opens a file for writing
///
/// # Arguments
///
/// * `path` - Path to the file
///
/// # Returns
///
/// * `Ok(BufWriter<File>)` if the file was opened successfully
/// * `Err(CoreError::IoError)` if the file could not be opened
///
/// # Errors
/// Returns `CoreError::IoError` if the file could not be opened.
pub fn create_file<P: AsRef<Path>>(path: P) -> CoreResult<BufWriter<File>> {
    let file = File::create(path.as_ref())?;
    Ok(BufWriter::new(file))
}

/// Reads an entire file into a string
///
/// # Arguments
///
/// * `path` - Path to the file
///
/// # Returns
///
/// * `Ok(String)` if the file was read successfully
/// * `Err(CoreError::IoError)` if the file could not be read
///
/// # Errors
/// Returns `CoreError::IoError` if the file could not be read.
pub fn read_to_string<P: AsRef<Path>>(path: P) -> CoreResult<String> {
    let mut file = open_file(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

/// Reads an entire file into a byte vector
///
/// # Arguments
///
/// * `path` - Path to the file
///
/// # Returns
///
/// * `Ok(Vec<u8>)` if the file was read successfully
/// * `Err(CoreError::IoError)` if the file could not be read
///
/// # Errors
/// Returns `CoreError::IoError` if the file could not be read.
pub fn read_to_bytes<P: AsRef<Path>>(path: P) -> CoreResult<Vec<u8>> {
    let mut file = open_file(path)?;
    let mut contents = Vec::new();
    file.read_to_end(&mut contents)?;
    Ok(contents)
}

/// Writes a string to a file
///
/// # Arguments
///
/// * `path` - Path to the file
/// * `contents` - String to write
///
/// # Returns
///
/// * `Ok(())` if the file was written successfully
/// * `Err(CoreError::IoError)` if the file could not be written
pub fn write_string<P: AsRef<Path>, S: AsRef<str>>(path: P, contents: S) -> CoreResult<()> {
    let mut file = create_file(path)?;
    file.write_all(contents.as_ref().as_bytes())?;
    file.flush()?;
    Ok(())
}

/// Writes bytes to a file
///
/// # Arguments
///
/// * `path` - Path to the file
/// * `contents` - Bytes to write
///
/// # Returns
///
/// * `Ok(())` if the file was written successfully
/// * `Err(CoreError::IoError)` if the file could not be written
pub fn write_bytes<P: AsRef<Path>, B: AsRef<[u8]>>(path: P, contents: B) -> CoreResult<()> {
    let mut file = create_file(path)?;
    file.write_all(contents.as_ref())?;
    file.flush()?;
    Ok(())
}

/// Reads a file line by line
///
/// # Arguments
///
/// * `path` - Path to the file
/// * `callback` - Function to call for each line
///
/// # Returns
///
/// * `Ok(())` if the file was read successfully
/// * `Err(CoreError::IoError)` if the file could not be read
pub fn read_lines<P, F>(path: P, mut callback: F) -> CoreResult<()>
where
    P: AsRef<Path>,
    F: FnMut(String) -> CoreResult<()>,
{
    let file = open_file(path)?;
    for line in file.lines() {
        let line = line?;
        callback(line)?;
    }
    Ok(())
}

/// Checks if a file exists
///
/// # Arguments
///
/// * `path` - Path to the file
///
/// # Returns
///
/// * `true` if the file exists
/// * `false` if the file does not exist
#[must_use]
pub fn file_exists<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().exists() && path.as_ref().is_file()
}

/// Checks if a directory exists
///
/// # Arguments
///
/// * `path` - Path to the directory
///
/// # Returns
///
/// * `true` if the directory exists
/// * `false` if the directory does not exist
#[must_use]
pub fn directory_exists<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().exists() && path.as_ref().is_dir()
}

/// Creates a directory if it doesn't exist
///
/// # Arguments
///
/// * `path` - Path to the directory
///
/// # Returns
///
/// * `Ok(())` if the directory was created successfully or already exists
/// * `Err(CoreError::IoError)` if the directory could not be created
pub fn create_directory<P: AsRef<Path>>(path: P) -> CoreResult<()> {
    if !directory_exists(&path) {
        std::fs::create_dir_all(path.as_ref())?;
    }
    Ok(())
}

/// Returns the file size in bytes
///
/// # Arguments
///
/// * `path` - Path to the file
///
/// # Returns
///
/// * `Ok(u64)` if the file size was determined successfully
/// * `Err(CoreError::IoError)` if the file size could not be determined
///
/// # Errors
/// Returns `CoreError::IoError` if the file size could not be determined.
pub fn file_size<P: AsRef<Path>>(path: P) -> CoreResult<u64> {
    let metadata = std::fs::metadata(path.as_ref())?;
    Ok(metadata.len())
}

/// Pretty-prints a file size with appropriate units
///
/// # Arguments
///
/// * `size` - File size in bytes
///
/// # Returns
///
/// * A formatted string with appropriate units
#[must_use]
pub fn format_file_size(size: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if size >= TB {
        format!("{:.2} TB", size as f64 / TB as f64)
    } else if size >= GB {
        format!("{:.2} GB", size as f64 / GB as f64)
    } else if size >= MB {
        format!("{:.2} MB", size as f64 / MB as f64)
    } else if size >= KB {
        format!("{:.2} KB", size as f64 / KB as f64)
    } else {
        format!("{} B", size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_read_write_string() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");

        let test_str = "Hello, world!";
        write_string(&file_path, test_str).unwrap();

        let contents = read_to_string(&file_path).unwrap();
        assert_eq!(contents, test_str);
    }

    #[test]
    fn test_read_write_bytes() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.bin");

        let test_bytes = vec![1, 2, 3, 4, 5];
        write_bytes(&file_path, &test_bytes).unwrap();

        let contents = read_to_bytes(&file_path).unwrap();
        assert_eq!(contents, test_bytes);
    }

    #[test]
    fn test_read_lines() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_lines.txt");

        {
            let mut file = File::create(&file_path).unwrap();
            writeln!(file, "Line 1").unwrap();
            writeln!(file, "Line 2").unwrap();
            writeln!(file, "Line 3").unwrap();
        }

        let mut lines = Vec::new();
        read_lines(&file_path, |line| {
            lines.push(line);
            Ok(())
        })
        .unwrap();

        assert_eq!(lines, vec!["Line 1", "Line 2", "Line 3"]);
    }

    #[test]
    fn test_file_exists() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_exists.txt");

        assert!(!file_exists(&file_path));

        File::create(&file_path).unwrap();

        assert!(file_exists(&file_path));
    }

    #[test]
    fn test_directory_exists() {
        let dir = tempdir().unwrap();
        let dir_path = dir.path().join("test_dir");

        assert!(!directory_exists(&dir_path));

        std::fs::create_dir(&dir_path).unwrap();

        assert!(directory_exists(&dir_path));
    }

    #[test]
    fn test_create_directory() {
        let dir = tempdir().unwrap();
        let dir_path = dir.path().join("test_create_dir");

        assert!(!directory_exists(&dir_path));

        create_directory(&dir_path).unwrap();

        assert!(directory_exists(&dir_path));
    }

    #[test]
    fn test_file_size() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_size.txt");

        let test_str = "Hello, world!";
        write_string(&file_path, test_str).unwrap();

        let size = file_size(&file_path).unwrap();
        assert_eq!(size, test_str.len() as u64);
    }

    #[test]
    fn test_format_file_size() {
        assert_eq!(format_file_size(500), "500 B");
        assert_eq!(format_file_size(1024), "1.00 KB");
        assert_eq!(format_file_size(1500), "1.46 KB");
        assert_eq!(format_file_size(1024 * 1024), "1.00 MB");
        assert_eq!(format_file_size(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_file_size(1024 * 1024 * 1024 * 1024), "1.00 TB");
    }
}
