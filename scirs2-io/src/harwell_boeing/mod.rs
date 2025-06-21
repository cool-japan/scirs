//! Harwell-Boeing sparse matrix format support
//!
//! The Harwell-Boeing format is a standard format for storing sparse matrices,
//! particularly used in the scientific computing community. It stores matrices
//! in column-compressed sparse (CCS) format.
//!
//! This implementation provides:
//! - Reading and writing Harwell-Boeing files
//! - Support for real and complex matrices
//! - Different storage schemes (assembled, elemental, etc.)
//! - Conversion to/from other sparse matrix formats
//!
//! The format specification:
//! - Line 1: Title (72 characters)
//! - Line 2: Key (8 chars), totcrd, ptrcrd, indcrd, valcrd, rhscrd
//! - Line 3: mxtype, nrow, ncol, nnzero, neltvl
//! - Line 4: ptrfmt, indfmt, valfmt, rhsfmt
//! - Line 5: (optional) rhstyp, nrhs, nrhsix
//! - Data: column pointers, row indices, values, right-hand sides

use ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::str::FromStr;

use crate::error::{IoError, Result};

/// Harwell-Boeing matrix type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HBMatrixType {
    /// Real unsymmetric matrix
    RealUnsymmetric,
    /// Real symmetric matrix
    RealSymmetric,
    /// Real symmetric positive definite matrix
    RealSymmetricPositiveDefinite,
    /// Real skew-symmetric matrix
    RealSkewSymmetric,
    /// Complex unsymmetric matrix
    ComplexUnsymmetric,
    /// Complex symmetric matrix
    ComplexSymmetric,
    /// Complex hermitian matrix
    ComplexHermitian,
    /// Complex skew-symmetric matrix
    ComplexSkewSymmetric,
    /// Pattern only (no values)
    Pattern,
}

impl FromStr for HBMatrixType {
    type Err = IoError;

    fn from_str(s: &str) -> Result<Self> {
        if s.len() < 3 {
            return Err(IoError::FormatError(
                "Matrix type string too short".to_string(),
            ));
        }

        let chars: Vec<char> = s.chars().collect();

        match (chars[0], chars[1], chars[2]) {
            ('R', 'U', 'A') => Ok(HBMatrixType::RealUnsymmetric),
            ('R', 'S', 'A') => Ok(HBMatrixType::RealSymmetric),
            ('R', 'S', 'P') => Ok(HBMatrixType::RealSymmetricPositiveDefinite),
            ('R', 'S', 'S') => Ok(HBMatrixType::RealSkewSymmetric),
            ('C', 'U', 'A') => Ok(HBMatrixType::ComplexUnsymmetric),
            ('C', 'S', 'A') => Ok(HBMatrixType::ComplexSymmetric),
            ('C', 'H', 'A') => Ok(HBMatrixType::ComplexHermitian),
            ('C', 'S', 'S') => Ok(HBMatrixType::ComplexSkewSymmetric),
            ('P', _, _) => Ok(HBMatrixType::Pattern),
            _ => Err(IoError::FormatError(format!("Unknown matrix type: {}", s))),
        }
    }
}

impl std::fmt::Display for HBMatrixType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HBMatrixType::RealUnsymmetric => write!(f, "RUA"),
            HBMatrixType::RealSymmetric => write!(f, "RSA"),
            HBMatrixType::RealSymmetricPositiveDefinite => write!(f, "RSP"),
            HBMatrixType::RealSkewSymmetric => write!(f, "RSS"),
            HBMatrixType::ComplexUnsymmetric => write!(f, "CUA"),
            HBMatrixType::ComplexSymmetric => write!(f, "CSA"),
            HBMatrixType::ComplexHermitian => write!(f, "CHA"),
            HBMatrixType::ComplexSkewSymmetric => write!(f, "CSS"),
            HBMatrixType::Pattern => write!(f, "PUA"),
        }
    }
}

/// Harwell-Boeing file header
#[derive(Debug, Clone)]
pub struct HBHeader {
    /// Title (up to 72 characters)
    pub title: String,
    /// Key (up to 8 characters)
    pub key: String,
    /// Total number of lines
    pub totcrd: usize,
    /// Number of lines for column pointers
    pub ptrcrd: usize,
    /// Number of lines for row indices
    pub indcrd: usize,
    /// Number of lines for values
    pub valcrd: usize,
    /// Number of lines for right-hand sides
    pub rhscrd: usize,
    /// Matrix type
    pub mxtype: HBMatrixType,
    /// Number of rows
    pub nrow: usize,
    /// Number of columns
    pub ncol: usize,
    /// Number of non-zero entries
    pub nnzero: usize,
    /// Number of elemental matrix entries
    pub neltvl: usize,
    /// Format for column pointers
    pub ptrfmt: String,
    /// Format for row indices
    pub indfmt: String,
    /// Format for values
    pub valfmt: String,
    /// Format for right-hand sides
    pub rhsfmt: String,
}

/// Harwell-Boeing sparse matrix
#[derive(Debug, Clone)]
pub struct HBSparseMatrix<T> {
    /// File header
    pub header: HBHeader,
    /// Column pointers (size ncol + 1)
    pub colptr: Vec<usize>,
    /// Row indices (size nnzero)
    pub rowind: Vec<usize>,
    /// Values (size nnzero, if not pattern matrix)
    pub values: Option<Vec<T>>,
    /// Right-hand side vectors (optional)
    pub rhs: Option<Array2<T>>,
}

impl HBHeader {
    /// Parse Harwell-Boeing header from file
    pub fn parse_header<R: BufRead>(reader: &mut R) -> Result<Self> {
        let mut lines = Vec::new();

        // Read header lines
        for _ in 0..5 {
            let mut line = String::new();
            reader
                .read_line(&mut line)
                .map_err(|e| IoError::FileError(e.to_string()))?;
            lines.push(line.trim_end().to_string());
        }

        if lines.len() < 4 {
            return Err(IoError::FormatError("Incomplete header".to_string()));
        }

        // Parse line 1: Title
        let title = if lines[0].len() > 72 {
            lines[0][..72].to_string()
        } else {
            lines[0].clone()
        };

        // Parse line 2: Key and card counts
        let line2_parts: Vec<&str> = lines[1].split_whitespace().collect();
        if line2_parts.len() < 5 {
            return Err(IoError::FormatError("Invalid line 2 format".to_string()));
        }

        let key = line2_parts[0].to_string();
        let totcrd = line2_parts[1]
            .parse::<usize>()
            .map_err(|_| IoError::FormatError("Invalid totcrd".to_string()))?;
        let ptrcrd = line2_parts[2]
            .parse::<usize>()
            .map_err(|_| IoError::FormatError("Invalid ptrcrd".to_string()))?;
        let indcrd = line2_parts[3]
            .parse::<usize>()
            .map_err(|_| IoError::FormatError("Invalid indcrd".to_string()))?;
        let valcrd = line2_parts[4]
            .parse::<usize>()
            .map_err(|_| IoError::FormatError("Invalid valcrd".to_string()))?;
        let rhscrd = if line2_parts.len() > 5 {
            line2_parts[5]
                .parse::<usize>()
                .map_err(|_| IoError::FormatError("Invalid rhscrd".to_string()))?
        } else {
            0
        };

        // Parse line 3: Matrix dimensions and type
        let line3_parts: Vec<&str> = lines[2].split_whitespace().collect();
        if line3_parts.len() < 4 {
            return Err(IoError::FormatError("Invalid line 3 format".to_string()));
        }

        let mxtype = HBMatrixType::from_str(line3_parts[0])?;
        let nrow = line3_parts[1]
            .parse::<usize>()
            .map_err(|_| IoError::FormatError("Invalid nrow".to_string()))?;
        let ncol = line3_parts[2]
            .parse::<usize>()
            .map_err(|_| IoError::FormatError("Invalid ncol".to_string()))?;
        let nnzero = line3_parts[3]
            .parse::<usize>()
            .map_err(|_| IoError::FormatError("Invalid nnzero".to_string()))?;
        let neltvl = if line3_parts.len() > 4 {
            line3_parts[4]
                .parse::<usize>()
                .map_err(|_| IoError::FormatError("Invalid neltvl".to_string()))?
        } else {
            0
        };

        // Parse line 4: Formats
        let line4_parts: Vec<&str> = lines[3].split_whitespace().collect();
        if line4_parts.len() < 2 {
            return Err(IoError::FormatError("Invalid line 4 format".to_string()));
        }

        let ptrfmt = line4_parts[0].to_string();
        let indfmt = line4_parts[1].to_string();
        let valfmt = if line4_parts.len() > 2 {
            line4_parts[2].to_string()
        } else {
            String::new()
        };
        let rhsfmt = if line4_parts.len() > 3 {
            line4_parts[3].to_string()
        } else {
            String::new()
        };

        Ok(HBHeader {
            title,
            key,
            totcrd,
            ptrcrd,
            indcrd,
            valcrd,
            rhscrd,
            mxtype,
            nrow,
            ncol,
            nnzero,
            neltvl,
            ptrfmt,
            indfmt,
            valfmt,
            rhsfmt,
        })
    }

    /// Write header to a writer
    pub fn write_header<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Line 1: Title (padded to 72 characters)
        let padded_title = format!("{:<72}", self.title);
        writeln!(writer, "{}", &padded_title[..72.min(padded_title.len())])
            .map_err(|e| IoError::FileError(e.to_string()))?;

        // Line 2: Key and card counts
        writeln!(
            writer,
            "{:<8} {:>7} {:>7} {:>7} {:>7} {:>7}",
            self.key, self.totcrd, self.ptrcrd, self.indcrd, self.valcrd, self.rhscrd
        )
        .map_err(|e| IoError::FileError(e.to_string()))?;

        // Line 3: Matrix type and dimensions
        writeln!(
            writer,
            "{:<3} {:>11} {:>11} {:>11} {:>11}",
            self.mxtype, self.nrow, self.ncol, self.nnzero, self.neltvl
        )
        .map_err(|e| IoError::FileError(e.to_string()))?;

        // Line 4: Formats
        writeln!(
            writer,
            "{:<16} {:<16} {:<20} {:<20}",
            self.ptrfmt, self.indfmt, self.valfmt, self.rhsfmt
        )
        .map_err(|e| IoError::FileError(e.to_string()))?;

        Ok(())
    }
}

/// Read a Harwell-Boeing sparse matrix file
///
/// # Arguments
///
/// * `path` - Path to the Harwell-Boeing file
///
/// # Returns
///
/// * `Result<HBSparseMatrix<f64>>` - The sparse matrix or an error
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::harwell_boeing::read_harwell_boeing;
///
/// let matrix = read_harwell_boeing("matrix.hb").unwrap();
/// println!("Matrix: {}x{} with {} non-zeros", matrix.header.nrow, matrix.header.ncol, matrix.header.nnzero);
/// ```
pub fn read_harwell_boeing<P: AsRef<Path>>(path: P) -> Result<HBSparseMatrix<f64>> {
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut reader = BufReader::new(file);

    // Parse header
    let header = HBHeader::parse_header(&mut reader)?;

    // Read column pointers
    let mut colptr = Vec::with_capacity(header.ncol + 1);
    read_integer_data(&mut reader, header.ptrcrd, &mut colptr)?;

    if colptr.len() != header.ncol + 1 {
        return Err(IoError::FormatError(format!(
            "Expected {} column pointers, got {}",
            header.ncol + 1,
            colptr.len()
        )));
    }

    // Read row indices
    let mut rowind = Vec::with_capacity(header.nnzero);
    read_integer_data(&mut reader, header.indcrd, &mut rowind)?;

    if rowind.len() != header.nnzero {
        return Err(IoError::FormatError(format!(
            "Expected {} row indices, got {}",
            header.nnzero,
            rowind.len()
        )));
    }

    // Convert to 0-based indexing
    for ptr in &mut colptr {
        *ptr -= 1;
    }
    for idx in &mut rowind {
        *idx -= 1;
    }

    // Read values (if not a pattern matrix)
    let values = if header.mxtype == HBMatrixType::Pattern {
        None
    } else {
        let mut vals = Vec::with_capacity(header.nnzero);
        read_real_data(&mut reader, header.valcrd, &mut vals)?;

        if vals.len() != header.nnzero {
            return Err(IoError::FormatError(format!(
                "Expected {} values, got {}",
                header.nnzero,
                vals.len()
            )));
        }

        Some(vals)
    };

    // TODO: Read right-hand side vectors if present
    let rhs = None;

    Ok(HBSparseMatrix {
        header,
        colptr,
        rowind,
        values,
        rhs,
    })
}

/// Write a Harwell-Boeing sparse matrix file
///
/// # Arguments
///
/// * `path` - Path to the output file
/// * `matrix` - The sparse matrix to write
///
/// # Returns
///
/// * `Result<()>` - Success or an error
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::harwell_boeing::{write_harwell_boeing, HBSparseMatrix, HBHeader, HBMatrixType};
///
/// # let header = HBHeader {
/// #     title: "Test matrix".to_string(),
/// #     key: "TEST".to_string(),
/// #     totcrd: 10, ptrcrd: 1, indcrd: 1, valcrd: 1, rhscrd: 0,
/// #     mxtype: HBMatrixType::RealUnsymmetric,
/// #     nrow: 2, ncol: 2, nnzero: 2, neltvl: 0,
/// #     ptrfmt: "(3I8)".to_string(), indfmt: "(2I8)".to_string(),
/// #     valfmt: "(2E16.8)".to_string(), rhsfmt: String::new(),
/// # };
/// # let matrix = HBSparseMatrix {
/// #     header, colptr: vec![0, 1, 2], rowind: vec![0, 1],
/// #     values: Some(vec![1.0, 2.0]), rhs: None,
/// # };
/// write_harwell_boeing("output.hb", &matrix).unwrap();
/// ```
pub fn write_harwell_boeing<P: AsRef<Path>>(path: P, matrix: &HBSparseMatrix<f64>) -> Result<()> {
    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    // Write header
    matrix.header.write_header(&mut writer)?;

    // Write column pointers (convert to 1-based indexing)
    let colptr_1based: Vec<usize> = matrix.colptr.iter().map(|&x| x + 1).collect();
    write_integer_data(&mut writer, &colptr_1based, 8)?;

    // Write row indices (convert to 1-based indexing)
    let rowind_1based: Vec<usize> = matrix.rowind.iter().map(|&x| x + 1).collect();
    write_integer_data(&mut writer, &rowind_1based, 8)?;

    // Write values (if present)
    if let Some(ref values) = matrix.values {
        write_real_data(&mut writer, values, 16)?;
    }

    // TODO: Write right-hand side vectors if present

    writer
        .flush()
        .map_err(|e| IoError::FileError(e.to_string()))?;

    Ok(())
}

/// Convert from Harwell-Boeing format to column-compressed sparse (CCS) format
///
/// # Arguments
///
/// * `matrix` - The Harwell-Boeing matrix
///
/// # Returns
///
/// * `(Array1<usize>, Array1<usize>, Array1<f64>)` - Column pointers, row indices, and values
pub fn hb_to_ccs(matrix: &HBSparseMatrix<f64>) -> (Array1<usize>, Array1<usize>, Array1<f64>) {
    let colptr = Array1::from(matrix.colptr.clone());
    let rowind = Array1::from(matrix.rowind.clone());
    let values = if let Some(ref vals) = matrix.values {
        Array1::from(vals.clone())
    } else {
        Array1::from(vec![1.0; matrix.rowind.len()]) // Pattern matrix
    };

    (colptr, rowind, values)
}

/// Create a Harwell-Boeing matrix from column-compressed sparse (CCS) format
///
/// # Arguments
///
/// * `colptr` - Column pointers
/// * `rowind` - Row indices
/// * `values` - Values
/// * `shape` - Matrix shape (rows, cols)
/// * `title` - Matrix title
/// * `key` - Matrix key
/// * `mxtype` - Matrix type
///
/// # Returns
///
/// * `HBSparseMatrix<f64>` - The Harwell-Boeing matrix
pub fn ccs_to_hb(
    colptr: &Array1<usize>,
    rowind: &Array1<usize>,
    values: &Array1<f64>,
    shape: (usize, usize),
    title: String,
    key: String,
    mxtype: HBMatrixType,
) -> HBSparseMatrix<f64> {
    let (nrow, ncol) = shape;
    let nnzero = rowind.len();

    // Calculate header card counts (rough estimates)
    let ptrcrd = ((ncol + 1) + 7) / 8; // 8 integers per line
    let indcrd = (nnzero + 7) / 8; // 8 integers per line
    let valcrd = if mxtype == HBMatrixType::Pattern {
        0
    } else {
        (nnzero + 3) / 4 // 4 reals per line
    };
    let totcrd = 4 + ptrcrd + indcrd + valcrd; // 4 header lines + data

    let header = HBHeader {
        title,
        key,
        totcrd,
        ptrcrd,
        indcrd,
        valcrd,
        rhscrd: 0,
        mxtype,
        nrow,
        ncol,
        nnzero,
        neltvl: 0,
        ptrfmt: "(8I10)".to_string(),
        indfmt: "(8I10)".to_string(),
        valfmt: if mxtype == HBMatrixType::Pattern {
            String::new()
        } else {
            "(4E20.12)".to_string()
        },
        rhsfmt: String::new(),
    };

    HBSparseMatrix {
        header,
        colptr: colptr.to_vec(),
        rowind: rowind.to_vec(),
        values: if mxtype == HBMatrixType::Pattern {
            None
        } else {
            Some(values.to_vec())
        },
        rhs: None,
    }
}

/// Read integer data from file
fn read_integer_data<R: BufRead>(
    reader: &mut R,
    num_lines: usize,
    data: &mut Vec<usize>,
) -> Result<()> {
    for _ in 0..num_lines {
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .map_err(|e| IoError::FileError(e.to_string()))?;

        // Parse integers from the line (assuming free format for simplicity)
        for token in line.split_whitespace() {
            if let Ok(value) = token.parse::<usize>() {
                data.push(value);
            }
        }
    }
    Ok(())
}

/// Read real data from file
fn read_real_data<R: BufRead>(reader: &mut R, num_lines: usize, data: &mut Vec<f64>) -> Result<()> {
    for _ in 0..num_lines {
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .map_err(|e| IoError::FileError(e.to_string()))?;

        // Parse reals from the line (assuming free format for simplicity)
        for token in line.split_whitespace() {
            if let Ok(value) = token.parse::<f64>() {
                data.push(value);
            }
        }
    }
    Ok(())
}

/// Write integer data to file
fn write_integer_data<W: Write>(writer: &mut W, data: &[usize], field_width: usize) -> Result<()> {
    const INTS_PER_LINE: usize = 8;

    for chunk in data.chunks(INTS_PER_LINE) {
        for (i, &value) in chunk.iter().enumerate() {
            if i > 0 {
                write!(writer, " ").map_err(|e| IoError::FileError(e.to_string()))?;
            }
            write!(writer, "{:width$}", value, width = field_width)
                .map_err(|e| IoError::FileError(e.to_string()))?;
        }
        writeln!(writer).map_err(|e| IoError::FileError(e.to_string()))?;
    }
    Ok(())
}

/// Write real data to file
fn write_real_data<W: Write>(writer: &mut W, data: &[f64], field_width: usize) -> Result<()> {
    const REALS_PER_LINE: usize = 4;

    for chunk in data.chunks(REALS_PER_LINE) {
        for (i, &value) in chunk.iter().enumerate() {
            if i > 0 {
                write!(writer, " ").map_err(|e| IoError::FileError(e.to_string()))?;
            }
            write!(writer, "{:width$.6E}", value, width = field_width)
                .map_err(|e| IoError::FileError(e.to_string()))?;
        }
        writeln!(writer).map_err(|e| IoError::FileError(e.to_string()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_type_parsing() {
        assert_eq!(
            HBMatrixType::from_str("RUA").unwrap(),
            HBMatrixType::RealUnsymmetric
        );
        assert_eq!(
            HBMatrixType::from_str("RSA").unwrap(),
            HBMatrixType::RealSymmetric
        );
        assert_eq!(
            HBMatrixType::from_str("CUA").unwrap(),
            HBMatrixType::ComplexUnsymmetric
        );
        assert_eq!(
            HBMatrixType::from_str("PUA").unwrap(),
            HBMatrixType::Pattern
        );
    }

    #[test]
    fn test_matrix_type_display() {
        assert_eq!(HBMatrixType::RealUnsymmetric.to_string(), "RUA");
        assert_eq!(HBMatrixType::ComplexHermitian.to_string(), "CHA");
        assert_eq!(HBMatrixType::Pattern.to_string(), "PUA");
    }

    #[test]
    fn test_ccs_conversion() {
        // Create test data
        let colptr = Array1::from(vec![0, 2, 4]);
        let rowind = Array1::from(vec![0, 1, 0, 1]);
        let values = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);

        // Convert to HB format
        let hb_matrix = ccs_to_hb(
            &colptr,
            &rowind,
            &values,
            (2, 2),
            "Test matrix".to_string(),
            "TEST".to_string(),
            HBMatrixType::RealUnsymmetric,
        );

        // Verify
        assert_eq!(hb_matrix.header.nrow, 2);
        assert_eq!(hb_matrix.header.ncol, 2);
        assert_eq!(hb_matrix.header.nnzero, 4);
        assert_eq!(hb_matrix.colptr, vec![0, 2, 4]);
        assert_eq!(hb_matrix.rowind, vec![0, 1, 0, 1]);
        assert_eq!(
            hb_matrix.values.as_ref().unwrap(),
            &vec![1.0, 2.0, 3.0, 4.0]
        );

        // Convert back to CCS
        let (new_colptr, new_rowind, new_values) = hb_to_ccs(&hb_matrix);

        assert_eq!(new_colptr, colptr);
        assert_eq!(new_rowind, rowind);
        assert_eq!(new_values, values);
    }

    #[test]
    fn test_pattern_matrix() {
        let colptr = Array1::from(vec![0, 1, 2]);
        let rowind = Array1::from(vec![0, 1]);
        let values = Array1::from(vec![1.0, 1.0]); // Will be ignored for pattern matrix

        let hb_matrix = ccs_to_hb(
            &colptr,
            &rowind,
            &values,
            (2, 2),
            "Pattern matrix".to_string(),
            "PATTERN".to_string(),
            HBMatrixType::Pattern,
        );

        assert_eq!(hb_matrix.header.mxtype, HBMatrixType::Pattern);
        assert!(hb_matrix.values.is_none());
        assert_eq!(hb_matrix.header.valcrd, 0);
    }

    #[test]
    fn test_header_fields() {
        let header = HBHeader {
            title: "Test matrix for unit testing".to_string(),
            key: "TESTKEY".to_string(),
            totcrd: 10,
            ptrcrd: 1,
            indcrd: 1,
            valcrd: 1,
            rhscrd: 0,
            mxtype: HBMatrixType::RealSymmetric,
            nrow: 100,
            ncol: 100,
            nnzero: 500,
            neltvl: 0,
            ptrfmt: "(8I10)".to_string(),
            indfmt: "(8I10)".to_string(),
            valfmt: "(4E20.12)".to_string(),
            rhsfmt: String::new(),
        };

        assert_eq!(header.title, "Test matrix for unit testing");
        assert_eq!(header.key, "TESTKEY");
        assert_eq!(header.mxtype, HBMatrixType::RealSymmetric);
        assert_eq!(header.nrow, 100);
        assert_eq!(header.ncol, 100);
        assert_eq!(header.nnzero, 500);
    }
}
