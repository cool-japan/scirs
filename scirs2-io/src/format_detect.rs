//! Data format detection via magic bytes and content sniffing
//!
//! Detects file formats by examining:
//! 1. Magic bytes (file signature) at the beginning of the file
//! 2. Content-based heuristics for formats without fixed signatures (CSV, JSON)
//! 3. File extension as a fallback
//!
//! # Supported Formats
//!
//! | Format | Magic Bytes | Extension |
//! |--------|-------------|-----------|
//! | HDF5 | `\x89HDF\r\n\x1a\n` | `.h5`, `.hdf5` |
//! | NetCDF Classic | `CDF\x01` or `CDF\x02` | `.nc`, `.cdf` |
//! | Arrow IPC | `ARROW1` | `.arrow`, `.feather` |
//! | MATLAB v5 | `MATLAB 5.0` | `.mat` |
//! | WAV | `RIFF....WAVE` | `.wav` |
//! | NPY | `\x93NUMPY` | `.npy` |
//! | NPZ (ZIP) | `PK\x03\x04` | `.npz` |
//! | PNG | `\x89PNG` | `.png` |
//! | JPEG | `\xFF\xD8\xFF` | `.jpg`, `.jpeg` |
//! | BMP | `BM` | `.bmp` |
//! | TIFF | `II\x2A\x00` or `MM\x00\x2A` | `.tif`, `.tiff` |
//! | FITS | `SIMPLE` | `.fits`, `.fit` |
//! | CSV | content-based | `.csv`, `.tsv` |
//! | JSON | content-based | `.json` |
//! | Parquet | `PAR1` | `.parquet` |
//! | FASTA | content-based (`>`) | `.fasta`, `.fa` |
//! | Matrix Market | `%%MatrixMarket` | `.mtx` |
//! | ARFF | `@RELATION` | `.arff` |
//! | MessagePack | content-based | `.msgpack` |
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_io::format_detect::{detect_format, detect_format_from_bytes, DataFormat};
//!
//! // Detect from file
//! let format = detect_format("data.h5")?;
//! assert_eq!(format, DataFormat::Hdf5);
//!
//! // Detect from bytes
//! let hdf5_magic = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];
//! let format = detect_format_from_bytes(&hdf5_magic, None);
//! assert_eq!(format, DataFormat::Hdf5);
//! # Ok::<(), scirs2_io::error::IoError>(())
//! ```

use crate::error::{IoError, Result};
use std::path::Path;

/// Recognized scientific data formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataFormat {
    /// HDF5 (Hierarchical Data Format v5)
    Hdf5,
    /// NetCDF Classic (v1 or v2)
    NetCdf,
    /// Apache Arrow IPC file format
    ArrowIpc,
    /// MATLAB v5 .mat file
    Matlab,
    /// WAV audio file
    Wav,
    /// NumPy .npy binary array
    Npy,
    /// NumPy .npz compressed archive
    Npz,
    /// PNG image
    Png,
    /// JPEG image
    Jpeg,
    /// BMP image
    Bmp,
    /// TIFF image
    Tiff,
    /// FITS (Flexible Image Transport System)
    Fits,
    /// CSV (Comma/Tab-Separated Values)
    Csv,
    /// JSON (JavaScript Object Notation)
    Json,
    /// Apache Parquet columnar format
    Parquet,
    /// FASTA biological sequence format
    Fasta,
    /// Matrix Market exchange format
    MatrixMarket,
    /// ARFF (Attribute-Relation File Format)
    Arff,
    /// MessagePack binary format
    MessagePack,
    /// ZIP archive
    Zip,
    /// Fortran unformatted sequential file
    FortranUnformatted,
    /// SciRS2 custom binary format
    Scirs2,
    /// Unknown or unrecognized format
    Unknown,
}

impl DataFormat {
    /// Human-readable name for the format
    pub fn name(&self) -> &'static str {
        match self {
            Self::Hdf5 => "HDF5",
            Self::NetCdf => "NetCDF",
            Self::ArrowIpc => "Arrow IPC",
            Self::Matlab => "MATLAB v5",
            Self::Wav => "WAV",
            Self::Npy => "NumPy NPY",
            Self::Npz => "NumPy NPZ",
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
            Self::Bmp => "BMP",
            Self::Tiff => "TIFF",
            Self::Fits => "FITS",
            Self::Csv => "CSV",
            Self::Json => "JSON",
            Self::Parquet => "Parquet",
            Self::Fasta => "FASTA",
            Self::MatrixMarket => "Matrix Market",
            Self::Arff => "ARFF",
            Self::MessagePack => "MessagePack",
            Self::Zip => "ZIP",
            Self::FortranUnformatted => "Fortran Unformatted",
            Self::Scirs2 => "SciRS2",
            Self::Unknown => "Unknown",
        }
    }

    /// Common file extension(s) for this format
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            Self::Hdf5 => &["h5", "hdf5", "he5"],
            Self::NetCdf => &["nc", "cdf", "nc3"],
            Self::ArrowIpc => &["arrow", "feather", "ipc"],
            Self::Matlab => &["mat"],
            Self::Wav => &["wav"],
            Self::Npy => &["npy"],
            Self::Npz => &["npz"],
            Self::Png => &["png"],
            Self::Jpeg => &["jpg", "jpeg"],
            Self::Bmp => &["bmp"],
            Self::Tiff => &["tif", "tiff"],
            Self::Fits => &["fits", "fit", "fts"],
            Self::Csv => &["csv", "tsv", "txt"],
            Self::Json => &["json", "geojson"],
            Self::Parquet => &["parquet"],
            Self::Fasta => &["fasta", "fa", "fna", "faa"],
            Self::MatrixMarket => &["mtx"],
            Self::Arff => &["arff"],
            Self::MessagePack => &["msgpack", "mpk"],
            Self::Zip => &["zip"],
            Self::FortranUnformatted => &["unf"],
            Self::Scirs2 => &["scirs2"],
            Self::Unknown => &[],
        }
    }

    /// Whether this format is text-based (not binary)
    pub fn is_text(&self) -> bool {
        matches!(
            self,
            Self::Csv | Self::Json | Self::Fasta | Self::MatrixMarket | Self::Arff
        )
    }

    /// Whether this format supports streaming reads
    pub fn supports_streaming(&self) -> bool {
        matches!(
            self,
            Self::Csv | Self::Json | Self::ArrowIpc | Self::Fasta | Self::Wav | Self::MatrixMarket
        )
    }
}

impl std::fmt::Display for DataFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Result of format detection with confidence
#[derive(Debug, Clone)]
pub struct FormatDetection {
    /// Detected format
    pub format: DataFormat,
    /// Confidence level (0.0 = guess, 1.0 = certain)
    pub confidence: f64,
    /// How the format was detected
    pub method: DetectionMethod,
}

/// How the format was detected
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionMethod {
    /// Detected via magic bytes (highest confidence)
    MagicBytes,
    /// Detected via content analysis
    ContentSniffing,
    /// Detected via file extension (lowest confidence)
    Extension,
    /// Multiple methods agreed
    Combined,
}

/// Detect the format of a file at the given path.
///
/// Reads the first 64 bytes of the file for magic-byte detection,
/// then falls back to extension-based detection.
pub fn detect_format<P: AsRef<Path>>(path: P) -> Result<DataFormat> {
    let path = path.as_ref();

    if !path.exists() {
        return Err(IoError::FileNotFound(path.display().to_string()));
    }

    // Read first 256 bytes for sniffing
    let file = std::fs::File::open(path)
        .map_err(|e| IoError::FileError(format!("Cannot open '{}': {e}", path.display())))?;
    let mut reader = std::io::BufReader::new(file);
    let mut header = vec![0u8; 256];
    let bytes_read = std::io::Read::read(&mut reader, &mut header)
        .map_err(|e| IoError::FileError(format!("Cannot read '{}': {e}", path.display())))?;
    header.truncate(bytes_read);

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    Ok(detect_format_from_bytes(&header, ext.as_deref()))
}

/// Detect format with detailed confidence information.
pub fn detect_format_detailed<P: AsRef<Path>>(path: P) -> Result<FormatDetection> {
    let path = path.as_ref();

    if !path.exists() {
        return Err(IoError::FileNotFound(path.display().to_string()));
    }

    let file = std::fs::File::open(path)
        .map_err(|e| IoError::FileError(format!("Cannot open '{}': {e}", path.display())))?;
    let mut reader = std::io::BufReader::new(file);
    let mut header = vec![0u8; 256];
    let bytes_read = std::io::Read::read(&mut reader, &mut header)
        .map_err(|e| IoError::FileError(format!("Cannot read '{}': {e}", path.display())))?;
    header.truncate(bytes_read);

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    Ok(detect_format_detailed_from_bytes(&header, ext.as_deref()))
}

/// Detect format from raw bytes with optional extension hint.
///
/// Checks magic bytes first (highest priority), then content sniffing,
/// then extension as fallback.
pub fn detect_format_from_bytes(data: &[u8], extension: Option<&str>) -> DataFormat {
    // 1. Magic bytes detection
    if let Some(fmt) = detect_magic_bytes(data) {
        return fmt;
    }

    // 2. Content-based sniffing
    if let Some(fmt) = detect_content(data) {
        return fmt;
    }

    // 3. Extension-based fallback
    if let Some(ext) = extension {
        if let Some(fmt) = detect_extension(ext) {
            return fmt;
        }
    }

    DataFormat::Unknown
}

/// Detect format with detailed confidence from bytes.
pub fn detect_format_detailed_from_bytes(data: &[u8], extension: Option<&str>) -> FormatDetection {
    // Magic bytes: high confidence
    if let Some(fmt) = detect_magic_bytes(data) {
        let ext_agrees = extension
            .and_then(|ext| detect_extension(ext))
            .map_or(false, |ef| ef == fmt);

        return FormatDetection {
            format: fmt,
            confidence: if ext_agrees { 1.0 } else { 0.95 },
            method: if ext_agrees {
                DetectionMethod::Combined
            } else {
                DetectionMethod::MagicBytes
            },
        };
    }

    // Content sniffing: medium confidence
    if let Some(fmt) = detect_content(data) {
        let ext_agrees = extension
            .and_then(|ext| detect_extension(ext))
            .map_or(false, |ef| ef == fmt);

        return FormatDetection {
            format: fmt,
            confidence: if ext_agrees { 0.85 } else { 0.6 },
            method: if ext_agrees {
                DetectionMethod::Combined
            } else {
                DetectionMethod::ContentSniffing
            },
        };
    }

    // Extension only: low confidence
    if let Some(ext) = extension {
        if let Some(fmt) = detect_extension(ext) {
            return FormatDetection {
                format: fmt,
                confidence: 0.4,
                method: DetectionMethod::Extension,
            };
        }
    }

    FormatDetection {
        format: DataFormat::Unknown,
        confidence: 0.0,
        method: DetectionMethod::Extension,
    }
}

// =====================================================================
// Internal detection functions
// =====================================================================

/// Detect format from magic bytes
fn detect_magic_bytes(data: &[u8]) -> Option<DataFormat> {
    if data.len() < 4 {
        return None;
    }

    // HDF5: \x89HDF\r\n\x1a\n
    if data.len() >= 8
        && data[0] == 0x89
        && data[1] == 0x48
        && data[2] == 0x44
        && data[3] == 0x46
        && data[4] == 0x0d
        && data[5] == 0x0a
        && data[6] == 0x1a
        && data[7] == 0x0a
    {
        return Some(DataFormat::Hdf5);
    }

    // NetCDF: "CDF\x01" (classic) or "CDF\x02" (64-bit offset)
    if data.len() >= 4
        && data[0] == b'C'
        && data[1] == b'D'
        && data[2] == b'F'
        && (data[3] == 0x01 || data[3] == 0x02)
    {
        return Some(DataFormat::NetCdf);
    }

    // Arrow IPC: "ARROW1"
    if data.len() >= 6 && &data[..6] == b"ARROW1" {
        return Some(DataFormat::ArrowIpc);
    }

    // MATLAB v5: starts with "MATLAB 5.0"
    if data.len() >= 10 && &data[..10] == b"MATLAB 5.0" {
        return Some(DataFormat::Matlab);
    }

    // WAV: "RIFF" + 4 bytes + "WAVE"
    if data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WAVE" {
        return Some(DataFormat::Wav);
    }

    // NPY: \x93NUMPY
    if data.len() >= 6 && data[0] == 0x93 && &data[1..6] == b"NUMPY" {
        return Some(DataFormat::Npy);
    }

    // Parquet: "PAR1"
    if data.len() >= 4 && &data[..4] == b"PAR1" {
        return Some(DataFormat::Parquet);
    }

    // PNG: \x89PNG\r\n\x1a\n
    if data.len() >= 8 && data[0] == 0x89 && data[1] == b'P' && data[2] == b'N' && data[3] == b'G' {
        return Some(DataFormat::Png);
    }

    // JPEG: \xFF\xD8\xFF
    if data.len() >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF {
        return Some(DataFormat::Jpeg);
    }

    // BMP: "BM"
    if data.len() >= 2 && data[0] == b'B' && data[1] == b'M' {
        return Some(DataFormat::Bmp);
    }

    // TIFF: "II\x2A\x00" (little-endian) or "MM\x00\x2A" (big-endian)
    if data.len() >= 4
        && ((data[0] == b'I' && data[1] == b'I' && data[2] == 0x2A && data[3] == 0x00)
            || (data[0] == b'M' && data[1] == b'M' && data[2] == 0x00 && data[3] == 0x2A))
    {
        return Some(DataFormat::Tiff);
    }

    // FITS: "SIMPLE  ="
    if data.len() >= 9 && &data[..6] == b"SIMPLE" {
        return Some(DataFormat::Fits);
    }

    // ZIP/NPZ: "PK\x03\x04"
    if data.len() >= 4 && data[0] == b'P' && data[1] == b'K' && data[2] == 0x03 && data[3] == 0x04 {
        // Could be NPZ if extension says so; default to ZIP
        return Some(DataFormat::Zip);
    }

    // SciRS2 custom: "SCIRS2\x00\x01"
    if data.len() >= 8 && &data[..6] == b"SCIRS2" {
        return Some(DataFormat::Scirs2);
    }

    None
}

/// Detect format from content analysis (for text-based formats)
fn detect_content(data: &[u8]) -> Option<DataFormat> {
    // Skip BOM if present
    let text_data = if data.len() >= 3 && data[0] == 0xEF && data[1] == 0xBB && data[2] == 0xBF {
        &data[3..]
    } else {
        data
    };

    // Try to interpret as UTF-8
    let text = std::str::from_utf8(text_data).ok()?;
    let trimmed = text.trim();

    if trimmed.is_empty() {
        return None;
    }

    // Matrix Market: starts with "%%MatrixMarket"
    if trimmed.starts_with("%%MatrixMarket") {
        return Some(DataFormat::MatrixMarket);
    }

    // ARFF: starts with "@RELATION" (case-insensitive)
    let upper = trimmed.to_uppercase();
    if upper.starts_with("@RELATION") || upper.starts_with("% ") && upper.contains("@RELATION") {
        return Some(DataFormat::Arff);
    }

    // JSON: starts with { or [
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        return Some(DataFormat::Json);
    }

    // FASTA: starts with > followed by sequence header
    if trimmed.starts_with('>') {
        let lines: Vec<&str> = trimmed.lines().take(5).collect();
        if lines.len() >= 2 {
            let second_line = lines[1].trim();
            if second_line
                .chars()
                .all(|c| "ACGTUNRYWSMKHBVD.-*acgtunrywsmkhbvd".contains(c))
            {
                return Some(DataFormat::Fasta);
            }
        }
    }

    // CSV: check for consistent delimited structure
    if is_likely_csv(trimmed) {
        return Some(DataFormat::Csv);
    }

    None
}

/// Heuristic check for CSV format
fn is_likely_csv(text: &str) -> bool {
    let lines: Vec<&str> = text.lines().take(10).collect();
    if lines.len() < 2 {
        return false;
    }

    // Check for consistent delimiter usage
    for delim in [',', '\t', ';', '|'] {
        let counts: Vec<usize> = lines
            .iter()
            .map(|line| line.matches(delim).count())
            .collect();

        if counts.is_empty() {
            continue;
        }

        let first = counts[0];
        if first == 0 {
            continue;
        }

        // Most lines should have the same count
        let matching = counts.iter().filter(|&&c| c == first).count();
        if matching >= counts.len() * 3 / 4 {
            return true;
        }
    }

    false
}

/// Detect format from file extension
fn detect_extension(ext: &str) -> Option<DataFormat> {
    let lower = ext.to_lowercase();
    // Remove leading dot if present
    let ext_clean = lower.trim_start_matches('.');

    match ext_clean {
        "h5" | "hdf5" | "he5" | "hdf" => Some(DataFormat::Hdf5),
        "nc" | "cdf" | "nc3" | "nc4" => Some(DataFormat::NetCdf),
        "arrow" | "feather" | "ipc" => Some(DataFormat::ArrowIpc),
        "mat" => Some(DataFormat::Matlab),
        "wav" | "wave" => Some(DataFormat::Wav),
        "npy" => Some(DataFormat::Npy),
        "npz" => Some(DataFormat::Npz),
        "png" => Some(DataFormat::Png),
        "jpg" | "jpeg" => Some(DataFormat::Jpeg),
        "bmp" => Some(DataFormat::Bmp),
        "tif" | "tiff" => Some(DataFormat::Tiff),
        "fits" | "fit" | "fts" => Some(DataFormat::Fits),
        "csv" | "tsv" => Some(DataFormat::Csv),
        "json" | "geojson" => Some(DataFormat::Json),
        "parquet" => Some(DataFormat::Parquet),
        "fasta" | "fa" | "fna" | "faa" => Some(DataFormat::Fasta),
        "mtx" => Some(DataFormat::MatrixMarket),
        "arff" => Some(DataFormat::Arff),
        "msgpack" | "mpk" => Some(DataFormat::MessagePack),
        "zip" => Some(DataFormat::Zip),
        "scirs2" => Some(DataFormat::Scirs2),
        _ => None,
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdf5_magic() {
        let magic = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];
        assert_eq!(detect_format_from_bytes(&magic, None), DataFormat::Hdf5);
    }

    #[test]
    fn test_netcdf_magic() {
        let magic = [b'C', b'D', b'F', 0x01];
        assert_eq!(detect_format_from_bytes(&magic, None), DataFormat::NetCdf);

        let magic2 = [b'C', b'D', b'F', 0x02];
        assert_eq!(detect_format_from_bytes(&magic2, None), DataFormat::NetCdf);
    }

    #[test]
    fn test_arrow_magic() {
        let magic = b"ARROW1\x00\x00";
        assert_eq!(detect_format_from_bytes(magic, None), DataFormat::ArrowIpc);
    }

    #[test]
    fn test_matlab_magic() {
        let magic = b"MATLAB 5.0 MAT-file, Platform: WIN64";
        assert_eq!(detect_format_from_bytes(magic, None), DataFormat::Matlab);
    }

    #[test]
    fn test_wav_magic() {
        let mut magic = Vec::new();
        magic.extend_from_slice(b"RIFF");
        magic.extend_from_slice(&1000u32.to_le_bytes());
        magic.extend_from_slice(b"WAVE");
        assert_eq!(detect_format_from_bytes(&magic, None), DataFormat::Wav);
    }

    #[test]
    fn test_npy_magic() {
        let magic = [0x93, b'N', b'U', b'M', b'P', b'Y'];
        assert_eq!(detect_format_from_bytes(&magic, None), DataFormat::Npy);
    }

    #[test]
    fn test_png_magic() {
        let magic = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        assert_eq!(detect_format_from_bytes(&magic, None), DataFormat::Png);
    }

    #[test]
    fn test_jpeg_magic() {
        let magic = [0xFF, 0xD8, 0xFF, 0xE0];
        assert_eq!(detect_format_from_bytes(&magic, None), DataFormat::Jpeg);
    }

    #[test]
    fn test_bmp_magic() {
        let magic = [b'B', b'M', 0x00, 0x00];
        assert_eq!(detect_format_from_bytes(&magic, None), DataFormat::Bmp);
    }

    #[test]
    fn test_tiff_magic_le() {
        let magic = [b'I', b'I', 0x2A, 0x00];
        assert_eq!(detect_format_from_bytes(&magic, None), DataFormat::Tiff);
    }

    #[test]
    fn test_tiff_magic_be() {
        let magic = [b'M', b'M', 0x00, 0x2A];
        assert_eq!(detect_format_from_bytes(&magic, None), DataFormat::Tiff);
    }

    #[test]
    fn test_parquet_magic() {
        let magic = b"PAR1";
        assert_eq!(detect_format_from_bytes(magic, None), DataFormat::Parquet);
    }

    #[test]
    fn test_zip_magic() {
        let magic = [b'P', b'K', 0x03, 0x04];
        assert_eq!(detect_format_from_bytes(&magic, None), DataFormat::Zip);
    }

    #[test]
    fn test_fits_magic() {
        let magic = b"SIMPLE  =                    T";
        assert_eq!(detect_format_from_bytes(magic, None), DataFormat::Fits);
    }

    #[test]
    fn test_json_content() {
        let data = b"{\"key\": \"value\"}";
        assert_eq!(detect_format_from_bytes(data, None), DataFormat::Json);

        let data2 = b"[1, 2, 3]";
        assert_eq!(detect_format_from_bytes(data2, None), DataFormat::Json);
    }

    #[test]
    fn test_matrix_market_content() {
        let data = b"%%MatrixMarket matrix coordinate real general\n3 3 4\n1 1 1.0\n";
        assert_eq!(
            detect_format_from_bytes(data, None),
            DataFormat::MatrixMarket
        );
    }

    #[test]
    fn test_arff_content() {
        let data = b"@RELATION test\n@ATTRIBUTE x NUMERIC\n@DATA\n1.0\n";
        assert_eq!(detect_format_from_bytes(data, None), DataFormat::Arff);
    }

    #[test]
    fn test_csv_content() {
        let data = b"a,b,c\n1,2,3\n4,5,6\n7,8,9\n";
        assert_eq!(detect_format_from_bytes(data, None), DataFormat::Csv);
    }

    #[test]
    fn test_tsv_content() {
        let data = b"a\tb\tc\n1\t2\t3\n4\t5\t6\n";
        assert_eq!(detect_format_from_bytes(data, None), DataFormat::Csv);
    }

    #[test]
    fn test_fasta_content() {
        let data = b">seq1\nACGTACGT\n>seq2\nGGGCCCAAA\n";
        assert_eq!(detect_format_from_bytes(data, None), DataFormat::Fasta);
    }

    #[test]
    fn test_extension_fallback() {
        let empty: &[u8] = &[];
        assert_eq!(
            detect_format_from_bytes(empty, Some("h5")),
            DataFormat::Hdf5
        );
        assert_eq!(
            detect_format_from_bytes(empty, Some("csv")),
            DataFormat::Csv
        );
        assert_eq!(
            detect_format_from_bytes(empty, Some("json")),
            DataFormat::Json
        );
        assert_eq!(
            detect_format_from_bytes(empty, Some("mat")),
            DataFormat::Matlab
        );
        assert_eq!(
            detect_format_from_bytes(empty, Some("parquet")),
            DataFormat::Parquet
        );
    }

    #[test]
    fn test_unknown_format() {
        let data = [0x00, 0x01, 0x02, 0x03];
        assert_eq!(detect_format_from_bytes(&data, None), DataFormat::Unknown);
    }

    #[test]
    fn test_format_name() {
        assert_eq!(DataFormat::Hdf5.name(), "HDF5");
        assert_eq!(DataFormat::Csv.name(), "CSV");
        assert_eq!(DataFormat::Unknown.name(), "Unknown");
    }

    #[test]
    fn test_format_extensions() {
        assert!(DataFormat::Hdf5.extensions().contains(&"h5"));
        assert!(DataFormat::Csv.extensions().contains(&"csv"));
        assert!(DataFormat::Unknown.extensions().is_empty());
    }

    #[test]
    fn test_is_text() {
        assert!(DataFormat::Csv.is_text());
        assert!(DataFormat::Json.is_text());
        assert!(!DataFormat::Hdf5.is_text());
        assert!(!DataFormat::Png.is_text());
    }

    #[test]
    fn test_supports_streaming() {
        assert!(DataFormat::Csv.supports_streaming());
        assert!(DataFormat::ArrowIpc.supports_streaming());
        assert!(!DataFormat::Hdf5.supports_streaming());
    }

    #[test]
    fn test_detailed_detection_magic() {
        let magic = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];
        let det = detect_format_detailed_from_bytes(&magic, Some("h5"));
        assert_eq!(det.format, DataFormat::Hdf5);
        assert!(det.confidence > 0.9);
        assert_eq!(det.method, DetectionMethod::Combined);
    }

    #[test]
    fn test_detailed_detection_content_only() {
        let data = b"{\"key\": 42}";
        let det = detect_format_detailed_from_bytes(data, None);
        assert_eq!(det.format, DataFormat::Json);
        assert_eq!(det.method, DetectionMethod::ContentSniffing);
        assert!(det.confidence > 0.5);
    }

    #[test]
    fn test_detailed_detection_extension_only() {
        let data: &[u8] = &[];
        let det = detect_format_detailed_from_bytes(data, Some("parquet"));
        assert_eq!(det.format, DataFormat::Parquet);
        assert_eq!(det.method, DetectionMethod::Extension);
        assert!(det.confidence < 0.5);
    }

    #[test]
    fn test_format_display() {
        assert_eq!(format!("{}", DataFormat::Hdf5), "HDF5");
        assert_eq!(format!("{}", DataFormat::ArrowIpc), "Arrow IPC");
    }

    #[test]
    fn test_empty_data_with_no_ext() {
        let data: &[u8] = &[];
        assert_eq!(detect_format_from_bytes(data, None), DataFormat::Unknown);
    }

    #[test]
    fn test_too_short_data() {
        let data = [0x89];
        assert_eq!(detect_format_from_bytes(&data, None), DataFormat::Unknown);
    }

    #[test]
    fn test_extension_with_dot() {
        assert_eq!(detect_extension(".h5"), Some(DataFormat::Hdf5));
        assert_eq!(detect_extension("h5"), Some(DataFormat::Hdf5));
    }

    #[test]
    fn test_extension_case_insensitive() {
        assert_eq!(detect_extension("H5"), Some(DataFormat::Hdf5));
        assert_eq!(detect_extension("CSV"), Some(DataFormat::Csv));
        assert_eq!(detect_extension("Json"), Some(DataFormat::Json));
    }

    #[test]
    fn test_file_not_found() {
        let result = detect_format("/definitely/nonexistent/path.h5");
        assert!(result.is_err());
    }

    #[test]
    fn test_scirs2_magic() {
        let mut magic = Vec::new();
        magic.extend_from_slice(b"SCIRS2");
        magic.push(0x00);
        magic.push(0x01);
        assert_eq!(detect_format_from_bytes(&magic, None), DataFormat::Scirs2);
    }
}
