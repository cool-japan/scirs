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
}
