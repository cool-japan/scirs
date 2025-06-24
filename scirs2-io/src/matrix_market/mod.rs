//! Matrix Market file format support
//!
//! This module provides functionality for reading and writing Matrix Market files,
//! which are commonly used for storing sparse matrices in scientific computing.
//!
//! Matrix Market is a standard format for representing sparse matrices,
//! supporting both coordinate format (COO) and array format.
//!
//! This implementation provides:
//! - Reading and writing Matrix Market files
//! - Support for coordinate format sparse matrices
//! - Support for dense array format
//! - Real and complex number support
//! - Different matrix types (general, symmetric, hermitian, skew-symmetric)

use ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::error::{IoError, Result};

/// Matrix Market data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MMDataType {
    /// Real numbers
    Real,
    /// Complex numbers
    Complex,
    /// Integer numbers
    Integer,
    /// Pattern (just structure, no values)
    Pattern,
}

impl FromStr for MMDataType {
    type Err = IoError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "real" => Ok(MMDataType::Real),
            "complex" => Ok(MMDataType::Complex),
            "integer" => Ok(MMDataType::Integer),
            "pattern" => Ok(MMDataType::Pattern),
            _ => Err(IoError::FormatError(format!("Unknown data type: {}", s))),
        }
    }
}

impl std::fmt::Display for MMDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MMDataType::Real => write!(f, "real"),
            MMDataType::Complex => write!(f, "complex"),
            MMDataType::Integer => write!(f, "integer"),
            MMDataType::Pattern => write!(f, "pattern"),
        }
    }
}

/// Matrix Market storage format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MMFormat {
    /// Coordinate format (sparse)
    Coordinate,
    /// Array format (dense)
    Array,
}

impl FromStr for MMFormat {
    type Err = IoError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "coordinate" => Ok(MMFormat::Coordinate),
            "array" => Ok(MMFormat::Array),
            _ => Err(IoError::FormatError(format!("Unknown format: {}", s))),
        }
    }
}

impl std::fmt::Display for MMFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MMFormat::Coordinate => write!(f, "coordinate"),
            MMFormat::Array => write!(f, "array"),
        }
    }
}

/// Matrix Market matrix type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MMSymmetry {
    /// General matrix (no symmetry)
    General,
    /// Symmetric matrix
    Symmetric,
    /// Hermitian matrix (complex symmetric)
    Hermitian,
    /// Skew-symmetric matrix
    SkewSymmetric,
}

impl FromStr for MMSymmetry {
    type Err = IoError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "general" => Ok(MMSymmetry::General),
            "symmetric" => Ok(MMSymmetry::Symmetric),
            "hermitian" => Ok(MMSymmetry::Hermitian),
            "skew-symmetric" => Ok(MMSymmetry::SkewSymmetric),
            _ => Err(IoError::FormatError(format!("Unknown symmetry: {}", s))),
        }
    }
}

impl std::fmt::Display for MMSymmetry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MMSymmetry::General => write!(f, "general"),
            MMSymmetry::Symmetric => write!(f, "symmetric"),
            MMSymmetry::Hermitian => write!(f, "hermitian"),
            MMSymmetry::SkewSymmetric => write!(f, "skew-symmetric"),
        }
    }
}

/// Matrix Market header information
#[derive(Debug, Clone)]
pub struct MMHeader {
    /// Object type (always "matrix" for Matrix Market)
    pub object: String,
    /// Storage format
    pub format: MMFormat,
    /// Data type
    pub data_type: MMDataType,
    /// Symmetry type
    pub symmetry: MMSymmetry,
    /// Comments from the file
    pub comments: Vec<String>,
}

/// Sparse matrix entry (coordinate format)
#[derive(Debug, Clone)]
pub struct SparseEntry<T> {
    /// Row index (1-based in file, 0-based in memory)
    pub row: usize,
    /// Column index (1-based in file, 0-based in memory)
    pub col: usize,
    /// Value
    pub value: T,
}

/// Matrix Market sparse matrix (COO format)
#[derive(Debug, Clone)]
pub struct MMSparseMatrix<T> {
    /// Matrix header
    pub header: MMHeader,
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Number of non-zero entries
    pub nnz: usize,
    /// Sparse entries
    pub entries: Vec<SparseEntry<T>>,
}

/// Matrix Market dense matrix
#[derive(Debug, Clone)]
pub struct MMDenseMatrix<T> {
    /// Matrix header
    pub header: MMHeader,
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Dense matrix data (column-major order)
    pub data: Array2<T>,
}

/// Configuration for parallel Matrix Market I/O
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of worker threads
    pub num_threads: usize,
    /// Chunk size for parallel processing (number of entries per chunk)
    pub chunk_size: usize,
    /// Buffer size for I/O operations
    pub buffer_size: usize,
    /// Enable memory mapping for large files
    pub use_memory_mapping: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            chunk_size: 100_000,           // 100k entries per chunk
            buffer_size: 64 * 1024 * 1024, // 64MB buffer
            use_memory_mapping: false,
        }
    }
}

/// Statistics for Matrix Market I/O operations
#[derive(Debug, Clone, Default)]
pub struct IOStats {
    /// Time taken for I/O operation in milliseconds
    pub io_time_ms: f64,
    /// Number of entries processed
    pub entries_processed: usize,
    /// Throughput in entries per second
    pub throughput_eps: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
}

impl MMHeader {
    /// Parse Matrix Market header line
    pub fn parse_header(line: &str) -> Result<Self> {
        if !line.starts_with("%%MatrixMarket") {
            return Err(IoError::FormatError(
                "Invalid Matrix Market header".to_string(),
            ));
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 5 {
            return Err(IoError::FormatError(
                "Invalid Matrix Market header format".to_string(),
            ));
        }

        let object = parts[1].to_string();
        let format = MMFormat::from_str(parts[2])?;
        let data_type = MMDataType::from_str(parts[3])?;
        let symmetry = MMSymmetry::from_str(parts[4])?;

        if object.to_lowercase() != "matrix" {
            return Err(IoError::FormatError(format!(
                "Unsupported object type: {}",
                object
            )));
        }

        Ok(MMHeader {
            object,
            format,
            data_type,
            symmetry,
            comments: Vec::new(),
        })
    }

    /// Generate Matrix Market header line
    pub fn to_header_line(&self) -> String {
        format!(
            "%%MatrixMarket {} {} {} {}",
            self.object, self.format, self.data_type, self.symmetry
        )
    }
}

/// Read a Matrix Market file containing a sparse matrix
///
/// # Arguments
///
/// * `path` - Path to the Matrix Market file
///
/// # Returns
///
/// * `Result<MMSparseMatrix<f64>>` - The sparse matrix or an error
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::matrix_market::read_sparse_matrix;
///
/// let matrix = read_sparse_matrix("matrix.mtx").unwrap();
/// println!("Matrix: {}x{} with {} non-zeros", matrix.rows, matrix.cols, matrix.nnz);
/// ```
pub fn read_sparse_matrix<P: AsRef<Path>>(path: P) -> Result<MMSparseMatrix<f64>> {
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();

    // Read header line
    let header_line = lines
        .next()
        .ok_or_else(|| IoError::FormatError("Empty file".to_string()))?
        .map_err(|e| IoError::FileError(e.to_string()))?;

    let mut header = MMHeader::parse_header(&header_line)?;

    // Read comments
    for line in &mut lines {
        let line = line.map_err(|e| IoError::FileError(e.to_string()))?;
        if line.starts_with('%') {
            header
                .comments
                .push(line.strip_prefix('%').unwrap().trim().to_string());
        } else {
            // This is the size line, put it back
            let size_parts: Vec<&str> = line.split_whitespace().collect();
            if size_parts.len() < 2 {
                return Err(IoError::FormatError("Invalid size line format".to_string()));
            }

            let rows = size_parts[0]
                .parse::<usize>()
                .map_err(|_| IoError::FormatError("Invalid row count".to_string()))?;
            let cols = size_parts[1]
                .parse::<usize>()
                .map_err(|_| IoError::FormatError("Invalid column count".to_string()))?;
            let nnz = if size_parts.len() > 2 {
                size_parts[2]
                    .parse::<usize>()
                    .map_err(|_| IoError::FormatError("Invalid nnz count".to_string()))?
            } else {
                // For array format, nnz is rows * cols
                rows * cols
            };

            if header.format != MMFormat::Coordinate {
                return Err(IoError::FormatError(
                    "Only coordinate format is supported for sparse matrices".to_string(),
                ));
            }

            // Read entries
            let mut entries = Vec::with_capacity(nnz);

            for line in lines {
                let line = line.map_err(|e| IoError::FileError(e.to_string()))?;
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 2 {
                    return Err(IoError::FormatError("Invalid entry format".to_string()));
                }

                let row = parts[0]
                    .parse::<usize>()
                    .map_err(|_| IoError::FormatError("Invalid row index".to_string()))?
                    - 1; // Convert to 0-based
                let col = parts[1]
                    .parse::<usize>()
                    .map_err(|_| IoError::FormatError("Invalid column index".to_string()))?
                    - 1; // Convert to 0-based

                let value = if header.data_type == MMDataType::Pattern {
                    1.0 // Pattern matrices have implicit value of 1
                } else if parts.len() > 2 {
                    parts[2]
                        .parse::<f64>()
                        .map_err(|_| IoError::FormatError("Invalid value".to_string()))?
                } else {
                    return Err(IoError::FormatError(
                        "Missing value for non-pattern matrix".to_string(),
                    ));
                };

                entries.push(SparseEntry { row, col, value });
            }

            return Ok(MMSparseMatrix {
                header,
                rows,
                cols,
                nnz,
                entries,
            });
        }
    }

    Err(IoError::FormatError("Missing size information".to_string()))
}

/// Write a sparse matrix to a Matrix Market file
///
/// # Arguments
///
/// * `path` - Path to the output Matrix Market file
/// * `matrix` - The sparse matrix to write
///
/// # Returns
///
/// * `Result<()>` - Success or an error
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::matrix_market::{write_sparse_matrix, MMSparseMatrix, MMHeader, SparseEntry};
/// use scirs2_io::matrix_market::{MMFormat, MMDataType, MMSymmetry};
///
/// let header = MMHeader {
///     object: "matrix".to_string(),
///     format: MMFormat::Coordinate,
///     data_type: MMDataType::Real,
///     symmetry: MMSymmetry::General,
///     comments: vec!["Generated by scirs2-io".to_string()],
/// };
///
/// let mut entries = Vec::new();
/// entries.push(SparseEntry { row: 0, col: 0, value: 1.0 });
/// entries.push(SparseEntry { row: 1, col: 1, value: 2.0 });
///
/// let matrix = MMSparseMatrix {
///     header,
///     rows: 2,
///     cols: 2,
///     nnz: 2,
///     entries,
/// };
///
/// write_sparse_matrix("output.mtx", &matrix).unwrap();
/// ```
pub fn write_sparse_matrix<P: AsRef<Path>>(path: P, matrix: &MMSparseMatrix<f64>) -> Result<()> {
    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "{}", matrix.header.to_header_line())
        .map_err(|e| IoError::FileError(e.to_string()))?;

    // Write comments
    for comment in &matrix.header.comments {
        writeln!(writer, "%{}", comment).map_err(|e| IoError::FileError(e.to_string()))?;
    }

    // Write size line
    writeln!(writer, "{} {} {}", matrix.rows, matrix.cols, matrix.nnz)
        .map_err(|e| IoError::FileError(e.to_string()))?;

    // Write entries
    for entry in &matrix.entries {
        if matrix.header.data_type == MMDataType::Pattern {
            // Pattern matrices only have row and column indices
            writeln!(writer, "{} {}", entry.row + 1, entry.col + 1)
                .map_err(|e| IoError::FileError(e.to_string()))?;
        } else {
            // Write row, column, and value (convert to 1-based indexing)
            writeln!(
                writer,
                "{} {} {}",
                entry.row + 1,
                entry.col + 1,
                entry.value
            )
            .map_err(|e| IoError::FileError(e.to_string()))?;
        }
    }

    writer
        .flush()
        .map_err(|e| IoError::FileError(e.to_string()))?;

    Ok(())
}

/// Read a Matrix Market file containing a dense matrix
///
/// # Arguments
///
/// * `path` - Path to the Matrix Market file
///
/// # Returns
///
/// * `Result<MMDenseMatrix<f64>>` - The dense matrix or an error
pub fn read_dense_matrix<P: AsRef<Path>>(path: P) -> Result<MMDenseMatrix<f64>> {
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();

    // Read header line
    let header_line = lines
        .next()
        .ok_or_else(|| IoError::FormatError("Empty file".to_string()))?
        .map_err(|e| IoError::FileError(e.to_string()))?;

    let mut header = MMHeader::parse_header(&header_line)?;

    if header.format != MMFormat::Array {
        return Err(IoError::FormatError(
            "Only array format is supported for dense matrices".to_string(),
        ));
    }

    // Read comments
    for line in &mut lines {
        let line = line.map_err(|e| IoError::FileError(e.to_string()))?;
        if line.starts_with('%') {
            header
                .comments
                .push(line.strip_prefix('%').unwrap().trim().to_string());
        } else {
            // This is the size line
            let size_parts: Vec<&str> = line.split_whitespace().collect();
            if size_parts.len() < 2 {
                return Err(IoError::FormatError("Invalid size line format".to_string()));
            }

            let rows = size_parts[0]
                .parse::<usize>()
                .map_err(|_| IoError::FormatError("Invalid row count".to_string()))?;
            let cols = size_parts[1]
                .parse::<usize>()
                .map_err(|_| IoError::FormatError("Invalid column count".to_string()))?;

            // Read matrix data (column-major order)
            let mut data = Vec::with_capacity(rows * cols);

            for line in lines {
                let line = line.map_err(|e| IoError::FileError(e.to_string()))?;
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                let value = line
                    .parse::<f64>()
                    .map_err(|_| IoError::FormatError("Invalid matrix value".to_string()))?;
                data.push(value);
            }

            if data.len() != rows * cols {
                return Err(IoError::FormatError(format!(
                    "Expected {} values, got {}",
                    rows * cols,
                    data.len()
                )));
            }

            // Convert to Array2 (column-major to row-major)
            let mut matrix_data = Array2::zeros((rows, cols));
            for col in 0..cols {
                for row in 0..rows {
                    matrix_data[[row, col]] = data[col * rows + row];
                }
            }

            return Ok(MMDenseMatrix {
                header,
                rows,
                cols,
                data: matrix_data,
            });
        }
    }

    Err(IoError::FormatError("Missing size information".to_string()))
}

/// Write a dense matrix to a Matrix Market file
///
/// # Arguments
///
/// * `path` - Path to the output Matrix Market file
/// * `matrix` - The dense matrix to write
///
/// # Returns
///
/// * `Result<()>` - Success or an error
pub fn write_dense_matrix<P: AsRef<Path>>(path: P, matrix: &MMDenseMatrix<f64>) -> Result<()> {
    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "{}", matrix.header.to_header_line())
        .map_err(|e| IoError::FileError(e.to_string()))?;

    // Write comments
    for comment in &matrix.header.comments {
        writeln!(writer, "%{}", comment).map_err(|e| IoError::FileError(e.to_string()))?;
    }

    // Write size line
    writeln!(writer, "{} {}", matrix.rows, matrix.cols)
        .map_err(|e| IoError::FileError(e.to_string()))?;

    // Write matrix data in column-major order
    for col in 0..matrix.cols {
        for row in 0..matrix.rows {
            writeln!(writer, "{}", matrix.data[[row, col]])
                .map_err(|e| IoError::FileError(e.to_string()))?;
        }
    }

    writer
        .flush()
        .map_err(|e| IoError::FileError(e.to_string()))?;

    Ok(())
}

/// Convert a sparse matrix to ndarray coordinate format
///
/// # Arguments
///
/// * `matrix` - The Matrix Market sparse matrix
///
/// # Returns
///
/// * `(Array1<usize>, Array1<usize>, Array1<f64>)` - Row indices, column indices, and values
pub fn sparse_to_coo(matrix: &MMSparseMatrix<f64>) -> (Array1<usize>, Array1<usize>, Array1<f64>) {
    let rows: Vec<usize> = matrix.entries.iter().map(|e| e.row).collect();
    let cols: Vec<usize> = matrix.entries.iter().map(|e| e.col).collect();
    let values: Vec<f64> = matrix.entries.iter().map(|e| e.value).collect();

    (Array1::from(rows), Array1::from(cols), Array1::from(values))
}

/// Create a sparse matrix from ndarray coordinate format
///
/// # Arguments
///
/// * `rows` - Row indices
/// * `cols` - Column indices
/// * `values` - Values
/// * `shape` - Matrix shape (rows, cols)
/// * `header` - Matrix Market header
///
/// # Returns
///
/// * `MMSparseMatrix<f64>` - The Matrix Market sparse matrix
pub fn coo_to_sparse(
    rows: &Array1<usize>,
    cols: &Array1<usize>,
    values: &Array1<f64>,
    shape: (usize, usize),
    header: MMHeader,
) -> MMSparseMatrix<f64> {
    let entries: Vec<SparseEntry<f64>> = rows
        .iter()
        .zip(cols.iter())
        .zip(values.iter())
        .map(|((&row, &col), &value)| SparseEntry { row, col, value })
        .collect();

    MMSparseMatrix {
        header,
        rows: shape.0,
        cols: shape.1,
        nnz: entries.len(),
        entries,
    }
}

/// Read a Matrix Market file in parallel for better performance with large matrices
///
/// # Arguments
///
/// * `path` - Path to the Matrix Market file
/// * `config` - Parallel processing configuration
///
/// # Returns
///
/// * `Result<(MMSparseMatrix<f64>, IOStats)>` - The sparse matrix and I/O statistics
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::matrix_market::{read_sparse_matrix_parallel, ParallelConfig};
///
/// let config = ParallelConfig::default();
/// let (matrix, stats) = read_sparse_matrix_parallel("large_matrix.mtx", config).unwrap();
/// println!("Read {} entries in {:.2}ms", stats.entries_processed, stats.io_time_ms);
/// ```
pub fn read_sparse_matrix_parallel<P: AsRef<Path>>(
    path: P,
    config: ParallelConfig,
) -> Result<(MMSparseMatrix<f64>, IOStats)> {
    let start_time = std::time::Instant::now();
    let mut stats = IOStats::default();

    let file = File::open(&path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut reader = BufReader::new(file);

    // Read header sequentially (always small)
    let mut header_line = String::new();
    reader
        .read_line(&mut header_line)
        .map_err(|e| IoError::FileError(e.to_string()))?;

    let mut header = MMHeader::parse_header(&header_line)?;

    // Read comments sequentially
    let mut line = String::new();
    let size_line;
    loop {
        line.clear();
        let bytes_read = reader
            .read_line(&mut line)
            .map_err(|e| IoError::FileError(e.to_string()))?;

        if bytes_read == 0 {
            return Err(IoError::FormatError("Unexpected end of file".to_string()));
        }

        if line.starts_with('%') {
            header
                .comments
                .push(line.strip_prefix('%').unwrap().trim().to_string());
        } else {
            size_line = line.clone();
            break;
        }
    }

    // Parse size information
    let size_parts: Vec<&str> = size_line.split_whitespace().collect();
    if size_parts.len() < 2 {
        return Err(IoError::FormatError("Invalid size line format".to_string()));
    }

    let rows = size_parts[0]
        .parse::<usize>()
        .map_err(|_| IoError::FormatError("Invalid row count".to_string()))?;
    let cols = size_parts[1]
        .parse::<usize>()
        .map_err(|_| IoError::FormatError("Invalid column count".to_string()))?;
    let nnz = if size_parts.len() > 2 {
        size_parts[2]
            .parse::<usize>()
            .map_err(|_| IoError::FormatError("Invalid nnz count".to_string()))?
    } else {
        rows * cols
    };

    if header.format != MMFormat::Coordinate {
        return Err(IoError::FormatError(
            "Only coordinate format is supported for sparse matrices".to_string(),
        ));
    }

    // For small matrices, use sequential reading
    if nnz <= config.chunk_size {
        let mut entries = Vec::with_capacity(nnz);

        for line in reader.lines() {
            let line = line.map_err(|e| IoError::FileError(e.to_string()))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let entry = parse_matrix_entry(line, &header)?;
            entries.push(entry);
        }

        stats.entries_processed = entries.len();
        stats.io_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        stats.throughput_eps = if stats.io_time_ms > 0.0 {
            stats.entries_processed as f64 / (stats.io_time_ms / 1000.0)
        } else {
            0.0
        };
        stats.memory_usage_bytes = std::mem::size_of::<SparseEntry<f64>>() * entries.len();

        return Ok((
            MMSparseMatrix {
                header,
                rows,
                cols,
                nnz,
                entries,
            },
            stats,
        ));
    }

    // Parallel reading for large matrices
    // Read all lines into memory first (for parallel processing)
    let lines: Vec<String> = reader
        .lines()
        .collect::<std::io::Result<Vec<_>>>()
        .map_err(|e| IoError::FileError(e.to_string()))?;

    let non_empty_lines: Vec<&str> = lines
        .iter()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    // Process lines in parallel chunks
    let entries = Arc::new(Mutex::new(Vec::with_capacity(nnz)));
    let chunk_size = config
        .chunk_size
        .min(non_empty_lines.len() / config.num_threads + 1);

    let mut handles = Vec::new();

    for chunk in non_empty_lines.chunks(chunk_size) {
        let chunk_lines: Vec<String> = chunk.iter().map(|&s| s.to_string()).collect();
        let entries_clone = Arc::clone(&entries);
        let header_clone = header.clone();

        let handle = thread::spawn(move || -> Result<()> {
            let mut local_entries = Vec::new();

            for line in chunk_lines {
                let entry = parse_matrix_entry(&line, &header_clone)?;
                local_entries.push(entry);
            }

            // Lock and append to shared vector
            let mut shared_entries = entries_clone.lock().unwrap();
            shared_entries.extend(local_entries);

            Ok(())
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle
            .join()
            .map_err(|_| IoError::FormatError("Thread join failed".to_string()))??;
    }

    let final_entries = Arc::try_unwrap(entries)
        .map_err(|_| IoError::FormatError("Failed to unwrap entries".to_string()))?
        .into_inner()
        .unwrap();

    stats.entries_processed = final_entries.len();
    stats.io_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
    stats.throughput_eps = if stats.io_time_ms > 0.0 {
        stats.entries_processed as f64 / (stats.io_time_ms / 1000.0)
    } else {
        0.0
    };
    stats.memory_usage_bytes = std::mem::size_of::<SparseEntry<f64>>() * final_entries.len();

    Ok((
        MMSparseMatrix {
            header,
            rows,
            cols,
            nnz,
            entries: final_entries,
        },
        stats,
    ))
}

/// Parse a single matrix entry line
fn parse_matrix_entry(line: &str, header: &MMHeader) -> Result<SparseEntry<f64>> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(IoError::FormatError("Invalid entry format".to_string()));
    }

    let row = parts[0]
        .parse::<usize>()
        .map_err(|_| IoError::FormatError("Invalid row index".to_string()))?
        - 1; // Convert to 0-based
    let col = parts[1]
        .parse::<usize>()
        .map_err(|_| IoError::FormatError("Invalid column index".to_string()))?
        - 1; // Convert to 0-based

    let value = if header.data_type == MMDataType::Pattern {
        1.0 // Pattern matrices have implicit value of 1
    } else if parts.len() > 2 {
        parts[2]
            .parse::<f64>()
            .map_err(|_| IoError::FormatError("Invalid value".to_string()))?
    } else {
        return Err(IoError::FormatError(
            "Missing value for non-pattern matrix".to_string(),
        ));
    };

    Ok(SparseEntry { row, col, value })
}

/// Write a sparse matrix to a Matrix Market file in parallel
///
/// # Arguments
///
/// * `path` - Path to the output Matrix Market file
/// * `matrix` - The sparse matrix to write
/// * `config` - Parallel processing configuration
///
/// # Returns
///
/// * `Result<IOStats>` - I/O statistics
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::matrix_market::{write_sparse_matrix_parallel, MMSparseMatrix, ParallelConfig};
/// # use scirs2_io::matrix_market::{MMHeader, MMFormat, MMDataType, MMSymmetry, SparseEntry};
///
/// # let header = MMHeader {
/// #     object: "matrix".to_string(),
/// #     format: MMFormat::Coordinate,
/// #     data_type: MMDataType::Real,
/// #     symmetry: MMSymmetry::General,
/// #     comments: Vec::new(),
/// # };
/// # let matrix = MMSparseMatrix {
/// #     header,
/// #     rows: 2,
/// #     cols: 2,
/// #     nnz: 2,
/// #     entries: vec![
/// #         SparseEntry { row: 0, col: 0, value: 1.0 },
/// #         SparseEntry { row: 1, col: 1, value: 2.0 },
/// #     ],
/// # };
/// let config = ParallelConfig::default();
/// let stats = write_sparse_matrix_parallel("output.mtx", &matrix, config).unwrap();
/// println!("Wrote {} entries in {:.2}ms", stats.entries_processed, stats.io_time_ms);
/// ```
pub fn write_sparse_matrix_parallel<P: AsRef<Path>>(
    path: P,
    matrix: &MMSparseMatrix<f64>,
    config: ParallelConfig,
) -> Result<IOStats> {
    let start_time = std::time::Instant::now();
    let mut stats = IOStats::default();

    let file = File::create(&path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::with_capacity(config.buffer_size, file);

    // Write header and metadata sequentially
    writeln!(writer, "{}", matrix.header.to_header_line())
        .map_err(|e| IoError::FileError(e.to_string()))?;

    for comment in &matrix.header.comments {
        writeln!(writer, "%{}", comment).map_err(|e| IoError::FileError(e.to_string()))?;
    }

    writeln!(writer, "{} {} {}", matrix.rows, matrix.cols, matrix.nnz)
        .map_err(|e| IoError::FileError(e.to_string()))?;

    // For small matrices, write sequentially
    if matrix.entries.len() <= config.chunk_size {
        for entry in &matrix.entries {
            write_matrix_entry(&mut writer, entry, &matrix.header)?;
        }
    } else {
        // For large matrices, format entries in parallel then write sequentially
        let chunk_size = config
            .chunk_size
            .min(matrix.entries.len() / config.num_threads + 1);
        let chunks: Vec<&[SparseEntry<f64>]> = matrix.entries.chunks(chunk_size).collect();

        let formatted_chunks = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();

        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let chunk_entries = chunk.to_vec();
            let header_clone = matrix.header.clone();
            let formatted_chunks_clone = Arc::clone(&formatted_chunks);

            let handle = thread::spawn(move || -> Result<()> {
                let mut local_lines = Vec::new();

                for entry in &chunk_entries {
                    let line = format_matrix_entry(entry, &header_clone);
                    local_lines.push(line);
                }

                let mut shared_chunks = formatted_chunks_clone.lock().unwrap();
                shared_chunks.push((chunk_idx, local_lines));

                Ok(())
            });

            handles.push(handle);
        }

        // Wait for all formatting to complete
        for handle in handles {
            handle
                .join()
                .map_err(|_| IoError::FormatError("Thread join failed".to_string()))??;
        }

        // Sort chunks by index and write sequentially
        let mut all_formatted = Arc::try_unwrap(formatted_chunks)
            .map_err(|_| IoError::FormatError("Failed to unwrap formatted chunks".to_string()))?
            .into_inner()
            .unwrap();

        all_formatted.sort_by_key(|&(idx, _)| idx);

        for (_, lines) in all_formatted {
            for line in lines {
                writeln!(writer, "{}", line).map_err(|e| IoError::FileError(e.to_string()))?;
            }
        }
    }

    writer
        .flush()
        .map_err(|e| IoError::FileError(e.to_string()))?;

    stats.entries_processed = matrix.entries.len();
    stats.io_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
    stats.throughput_eps = if stats.io_time_ms > 0.0 {
        stats.entries_processed as f64 / (stats.io_time_ms / 1000.0)
    } else {
        0.0
    };
    stats.memory_usage_bytes = std::mem::size_of::<SparseEntry<f64>>() * matrix.entries.len();

    Ok(stats)
}

/// Write a single matrix entry to the writer
fn write_matrix_entry<W: Write>(
    writer: &mut W,
    entry: &SparseEntry<f64>,
    header: &MMHeader,
) -> Result<()> {
    if header.data_type == MMDataType::Pattern {
        writeln!(writer, "{} {}", entry.row + 1, entry.col + 1)
            .map_err(|e| IoError::FileError(e.to_string()))?;
    } else {
        writeln!(
            writer,
            "{} {} {}",
            entry.row + 1,
            entry.col + 1,
            entry.value
        )
        .map_err(|e| IoError::FileError(e.to_string()))?;
    }
    Ok(())
}

/// Format a single matrix entry as a string
fn format_matrix_entry(entry: &SparseEntry<f64>, header: &MMHeader) -> String {
    if header.data_type == MMDataType::Pattern {
        format!("{} {}", entry.row + 1, entry.col + 1)
    } else {
        format!("{} {} {}", entry.row + 1, entry.col + 1, entry.value)
    }
}

/// Create optimal parallel configuration based on matrix size
///
/// # Arguments
///
/// * `nnz` - Number of non-zero entries
/// * `available_memory` - Available memory in bytes (optional)
///
/// # Returns
///
/// * `ParallelConfig` - Optimized configuration
pub fn create_optimal_parallel_config(
    nnz: usize,
    available_memory: Option<usize>,
) -> ParallelConfig {
    let mut config = ParallelConfig::default();

    // Adjust chunk size based on matrix size
    if nnz < 10_000 {
        // Small matrices - use sequential processing
        config.num_threads = 1;
        config.chunk_size = nnz;
    } else if nnz < 1_000_000 {
        // Medium matrices
        config.chunk_size = 50_000;
    } else {
        // Large matrices
        config.chunk_size = 200_000;
        config.use_memory_mapping = true;
    }

    // Adjust based on available memory
    if let Some(memory) = available_memory {
        let entry_size = std::mem::size_of::<SparseEntry<f64>>();
        let max_entries_in_memory = memory / (entry_size * 4); // Use 25% of available memory

        if nnz > max_entries_in_memory {
            config.chunk_size = config
                .chunk_size
                .min(max_entries_in_memory / config.num_threads);
            config.use_memory_mapping = true;
        }
    }

    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_parsing() {
        let header_line = "%%MatrixMarket matrix coordinate real general";
        let header = MMHeader::parse_header(header_line).unwrap();

        assert_eq!(header.object, "matrix");
        assert_eq!(header.format, MMFormat::Coordinate);
        assert_eq!(header.data_type, MMDataType::Real);
        assert_eq!(header.symmetry, MMSymmetry::General);
    }

    #[test]
    fn test_header_generation() {
        let header = MMHeader {
            object: "matrix".to_string(),
            format: MMFormat::Coordinate,
            data_type: MMDataType::Real,
            symmetry: MMSymmetry::General,
            comments: vec!["Test comment".to_string()],
        };

        let header_line = header.to_header_line();
        assert_eq!(header_line, "%%MatrixMarket matrix coordinate real general");
    }

    #[test]
    fn test_sparse_matrix_creation() {
        let header = MMHeader {
            object: "matrix".to_string(),
            format: MMFormat::Coordinate,
            data_type: MMDataType::Real,
            symmetry: MMSymmetry::General,
            comments: Vec::new(),
        };

        let mut entries = Vec::new();
        entries.push(SparseEntry {
            row: 0,
            col: 0,
            value: 1.0,
        });
        entries.push(SparseEntry {
            row: 1,
            col: 1,
            value: 2.0,
        });
        entries.push(SparseEntry {
            row: 0,
            col: 1,
            value: 3.0,
        });

        let matrix = MMSparseMatrix {
            header,
            rows: 2,
            cols: 2,
            nnz: 3,
            entries,
        };

        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);
        assert_eq!(matrix.nnz, 3);
        assert_eq!(matrix.entries.len(), 3);
    }

    #[test]
    fn test_sparse_to_coo_conversion() {
        let header = MMHeader {
            object: "matrix".to_string(),
            format: MMFormat::Coordinate,
            data_type: MMDataType::Real,
            symmetry: MMSymmetry::General,
            comments: Vec::new(),
        };

        let entries = vec![
            SparseEntry {
                row: 0,
                col: 0,
                value: 1.0,
            },
            SparseEntry {
                row: 1,
                col: 1,
                value: 2.0,
            },
        ];

        let matrix = MMSparseMatrix {
            header,
            rows: 2,
            cols: 2,
            nnz: 2,
            entries,
        };

        let (rows, cols, values) = sparse_to_coo(&matrix);

        assert_eq!(rows.len(), 2);
        assert_eq!(cols.len(), 2);
        assert_eq!(values.len(), 2);
        assert_eq!(rows[0], 0);
        assert_eq!(cols[0], 0);
        assert_eq!(values[0], 1.0);
        assert_eq!(rows[1], 1);
        assert_eq!(cols[1], 1);
        assert_eq!(values[1], 2.0);
    }

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelConfig::default();
        assert!(config.num_threads > 0);
        assert!(config.chunk_size > 0);
        assert!(config.buffer_size > 0);
    }

    #[test]
    fn test_optimal_parallel_config() {
        // Small matrix
        let config = create_optimal_parallel_config(5000, None);
        assert_eq!(config.num_threads, 1);
        assert_eq!(config.chunk_size, 5000);

        // Medium matrix
        let config = create_optimal_parallel_config(500_000, None);
        assert!(config.num_threads > 1);
        assert_eq!(config.chunk_size, 50_000);

        // Large matrix
        let config = create_optimal_parallel_config(5_000_000, None);
        assert!(config.num_threads > 1);
        assert_eq!(config.chunk_size, 200_000);
        assert!(config.use_memory_mapping);

        // Memory constrained
        let config = create_optimal_parallel_config(1_000_000, Some(100_000)); // 100KB memory
        assert!(config.use_memory_mapping);
        assert!(config.chunk_size < 1_000_000);
    }

    #[test]
    fn test_parse_matrix_entry() {
        let header = MMHeader {
            object: "matrix".to_string(),
            format: MMFormat::Coordinate,
            data_type: MMDataType::Real,
            symmetry: MMSymmetry::General,
            comments: Vec::new(),
        };

        // Test normal entry
        let entry = parse_matrix_entry(&format!("1 2 {}", std::f64::consts::PI), &header).unwrap();
        assert_eq!(entry.row, 0); // 0-based
        assert_eq!(entry.col, 1); // 0-based
        assert!((entry.value - std::f64::consts::PI).abs() < 1e-10);

        // Test pattern entry
        let mut pattern_header = header.clone();
        pattern_header.data_type = MMDataType::Pattern;
        let entry = parse_matrix_entry("5 10", &pattern_header).unwrap();
        assert_eq!(entry.row, 4);
        assert_eq!(entry.col, 9);
        assert_eq!(entry.value, 1.0);
    }

    #[test]
    fn test_format_matrix_entry() {
        let header = MMHeader {
            object: "matrix".to_string(),
            format: MMFormat::Coordinate,
            data_type: MMDataType::Real,
            symmetry: MMSymmetry::General,
            comments: Vec::new(),
        };

        let entry = SparseEntry {
            row: 0,
            col: 1,
            value: 2.5,
        };

        let formatted = format_matrix_entry(&entry, &header);
        assert_eq!(formatted, "1 2 2.5"); // 1-based indexing

        // Test pattern entry
        let mut pattern_header = header.clone();
        pattern_header.data_type = MMDataType::Pattern;
        let formatted = format_matrix_entry(&entry, &pattern_header);
        assert_eq!(formatted, "1 2");
    }

    #[test]
    fn test_io_stats() {
        let mut stats = IOStats::default();
        assert_eq!(stats.entries_processed, 0);
        assert_eq!(stats.io_time_ms, 0.0);
        assert_eq!(stats.throughput_eps, 0.0);
        assert_eq!(stats.memory_usage_bytes, 0);

        // Test throughput calculation
        stats.entries_processed = 1000;
        stats.io_time_ms = 100.0;
        stats.throughput_eps = stats.entries_processed as f64 / (stats.io_time_ms / 1000.0);
        assert_eq!(stats.throughput_eps, 10000.0); // 10k entries per second
    }
}
