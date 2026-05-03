//! Support for various data formats (Parquet, Arrow, HDF5)
//!
//! This module provides integration with scirs2-io for reading and writing
//! datasets in modern columnar formats like Parquet and Arrow, as well as
//! scientific formats like HDF5, with memory-efficient streaming support.
//!
//! ## Parquet conventions
//!
//! When writing a `Dataset` to Parquet, each feature column is stored as
//! `feature_0`, `feature_1`, … (or by the names in `featurenames` when present).
//! The optional target is stored in a column named `__target__`.
//! On read, any column named `__target__` is loaded as the target vector;
//! the remaining columns become the feature matrix.
//!
//! ## HDF5 conventions
//!
//! When writing, the feature matrix is stored under `<dataset_name>` and
//! the optional target (if present) is stored under `<dataset_name>_target`.
//! On read, the `<dataset_name>` dataset provides the feature matrix; a
//! companion `<dataset_name>_target` dataset (if present) becomes the target.

#[cfg(feature = "formats")]
use crate::error::{DatasetsError, Result};
#[cfg(feature = "formats")]
use crate::utils::Dataset;
#[cfg(feature = "formats")]
use scirs2_core::ndarray::{Array1, Array2};
#[cfg(feature = "formats")]
use std::path::Path;

/// Format type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatType {
    /// Apache Parquet columnar format
    Parquet,
    /// Apache Arrow in-memory format
    Arrow,
    /// HDF5 hierarchical format
    Hdf5,
    /// CSV format (for completeness)
    Csv,
}

impl FormatType {
    /// Detect format from file extension
    pub fn from_extension(path: &str) -> Option<Self> {
        let lower = path.to_lowercase();
        if lower.ends_with(".parquet") || lower.ends_with(".pq") {
            Some(FormatType::Parquet)
        } else if lower.ends_with(".arrow") {
            Some(FormatType::Arrow)
        } else if lower.ends_with(".h5") || lower.ends_with(".hdf5") {
            Some(FormatType::Hdf5)
        } else if lower.ends_with(".csv") {
            Some(FormatType::Csv)
        } else {
            None
        }
    }

    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            FormatType::Parquet => "parquet",
            FormatType::Arrow => "arrow",
            FormatType::Hdf5 => "h5",
            FormatType::Csv => "csv",
        }
    }
}

/// Configuration for format conversion
#[derive(Debug, Clone)]
pub struct FormatConfig {
    /// Chunk size for streaming operations
    pub chunk_size: usize,
    /// Compression codec
    pub compression: Option<CompressionCodec>,
    /// Whether to use memory mapping when possible
    pub use_mmap: bool,
    /// Buffer size for I/O operations
    pub buffer_size: usize,
}

impl Default for FormatConfig {
    fn default() -> Self {
        Self {
            chunk_size: 10_000,
            compression: Some(CompressionCodec::Snappy),
            use_mmap: true,
            buffer_size: 8 * 1024 * 1024, // 8 MB
        }
    }
}

/// Compression codec options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionCodec {
    /// No compression
    None,
    /// Snappy compression
    Snappy,
    /// GZIP compression
    Gzip,
    /// LZ4 compression
    Lz4,
    /// ZSTD compression
    Zstd,
}

impl CompressionCodec {
    /// Get compression level (0-9 where applicable)
    pub fn level(&self) -> Option<i32> {
        match self {
            CompressionCodec::None | CompressionCodec::Snappy | CompressionCodec::Lz4 => None,
            CompressionCodec::Gzip => Some(6), // Default GZIP level
            CompressionCodec::Zstd => Some(3), // Default ZSTD level
        }
    }
}

// ============================================================================
// Parquet Support (when formats feature is enabled)
// ============================================================================

// Target column name stored inside Parquet files.
#[cfg(feature = "formats")]
const PARQUET_TARGET_COLUMN: &str = "__target__";

/// Build column names for a Dataset: use featurenames when available, otherwise
/// synthesise `feature_0`, `feature_1`, …
#[cfg(feature = "formats")]
fn feature_column_names(dataset: &Dataset) -> Vec<String> {
    let n = dataset.n_features();
    match &dataset.featurenames {
        Some(names) if names.len() == n => names.clone(),
        _ => (0..n).map(|i| format!("feature_{i}")).collect(),
    }
}

/// Convert a `ParquetData` (from scirs2-io) back to a `Dataset`.
///
/// Columns whose name is `__target__` become the target vector; all others
/// (in schema order) populate the feature matrix.
#[cfg(feature = "formats")]
fn parquet_data_to_dataset(pdata: &scirs2_io::parquet::ParquetData) -> Result<Dataset> {
    // Use schema().column_names() to preserve the column order from the Arrow schema,
    // rather than pdata.column_names() which is backed by a HashMap (unordered).
    let all_columns = pdata.schema().column_names();
    let n_rows = pdata.num_rows();

    // Separate feature columns from the target column (preserving schema order).
    let feat_names: Vec<String> = all_columns
        .iter()
        .filter(|n| n.as_str() != PARQUET_TARGET_COLUMN)
        .cloned()
        .collect();

    if feat_names.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "Parquet file contains no feature columns (only '__target__' found)".to_string(),
        ));
    }

    let n_features = feat_names.len();

    // Build feature matrix row-by-row from each column's values.
    let mut flat: Vec<f64> = Vec::with_capacity(n_rows * n_features);
    for col_name in &feat_names {
        let col = pdata.get_column_f64(col_name).map_err(|e| {
            DatasetsError::InvalidFormat(format!(
                "Failed to read feature column '{}': {}",
                col_name, e
            ))
        })?;
        flat.extend(col.iter());
    }

    // flat is column-major (feature0[row0..rowN], feature1[row0..rowN], …)
    // Array2::from_shape_vec with shape (n_features, n_rows) then transpose gives (n_rows, n_features).
    let column_major = Array2::from_shape_vec((n_features, n_rows), flat).map_err(|e| {
        DatasetsError::InvalidFormat(format!("Failed to shape feature matrix: {e}"))
    })?;
    let data = column_major.t().to_owned();

    // Load optional target column.
    let target: Option<Array1<f64>> = if all_columns
        .iter()
        .any(|n| n.as_str() == PARQUET_TARGET_COLUMN)
    {
        let col = pdata.get_column_f64(PARQUET_TARGET_COLUMN).map_err(|e| {
            DatasetsError::InvalidFormat(format!("Failed to read target column: {e}"))
        })?;
        Some(Array1::from_vec(col.to_vec()))
    } else {
        None
    };

    let mut ds = Dataset::new(data, target);
    ds.featurenames = Some(feat_names);
    Ok(ds)
}

/// Write a `Dataset` to a Parquet file, building a multi-column RecordBatch.
///
/// Each feature column is named from `featurenames` or synthesised as
/// `feature_N`.  The optional target column is appended as `__target__`.
#[cfg(feature = "formats")]
fn write_dataset_parquet<P: AsRef<Path>>(dataset: &Dataset, path: P) -> Result<()> {
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use scirs2_io::parquet::{ParquetWriteOptions, ParquetWriter as IoParquetWriter};
    use std::sync::Arc;

    let col_names = feature_column_names(dataset);
    let n_rows = dataset.n_samples();
    let n_feats = dataset.n_features();

    // Build schema fields.
    let mut fields: Vec<Field> = col_names
        .iter()
        .map(|name| Field::new(name.as_str(), DataType::Float64, false))
        .collect();
    let has_target = dataset.target.is_some();
    if has_target {
        fields.push(Field::new(PARQUET_TARGET_COLUMN, DataType::Float64, false));
    }
    let schema = Arc::new(Schema::new(fields));

    // Build Arrow column arrays.
    let mut arrays: Vec<Arc<dyn arrow::array::Array>> = Vec::with_capacity(n_feats + 1);
    for col_idx in 0..n_feats {
        let col_data: Vec<f64> = (0..n_rows)
            .map(|row| dataset.data[[row, col_idx]])
            .collect();
        arrays.push(Arc::new(Float64Array::from(col_data)));
    }
    if let Some(target) = &dataset.target {
        let tgt_data: Vec<f64> = target.to_vec();
        arrays.push(Arc::new(Float64Array::from(tgt_data)));
    }

    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)
        .map_err(|e| DatasetsError::InvalidFormat(format!("Failed to build RecordBatch: {e}")))?;

    // Write using scirs2-io's ParquetWriter directly.
    let options = ParquetWriteOptions::default();
    let mut writer = IoParquetWriter::from_path(path, schema, options)
        .map_err(|e| DatasetsError::InvalidFormat(format!("Parquet writer creation error: {e}")))?;

    writer
        .write_batch(&batch)
        .map_err(|e| DatasetsError::InvalidFormat(format!("Parquet write error: {e}")))?;

    writer
        .close()
        .map_err(|e| DatasetsError::InvalidFormat(format!("Parquet close error: {e}")))
}

#[cfg(feature = "formats")]
/// Parquet reader for datasets
pub struct ParquetReader {
    config: FormatConfig,
}

#[cfg(feature = "formats")]
impl ParquetReader {
    /// Create a new Parquet reader
    pub fn new() -> Self {
        Self {
            config: FormatConfig::default(),
        }
    }

    /// Create a Parquet reader with custom configuration
    pub fn with_config(config: FormatConfig) -> Self {
        Self { config }
    }

    /// Read a Parquet file into a Dataset.
    ///
    /// Columns named `__target__` are treated as the target vector; all other
    /// columns (which must be `Float64`) form the feature matrix.
    pub fn read<P: AsRef<Path>>(&self, path: P) -> Result<Dataset> {
        let pdata = scirs2_io::parquet::read_parquet(path)
            .map_err(|e| DatasetsError::InvalidFormat(format!("Parquet read error: {e}")))?;
        let _ = &self.config; // config reserved for future chunked reads
        parquet_data_to_dataset(&pdata)
    }
}

#[cfg(feature = "formats")]
impl Default for ParquetReader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "formats")]
/// Parquet writer for datasets
pub struct ParquetWriter {
    config: FormatConfig,
}

#[cfg(feature = "formats")]
impl ParquetWriter {
    /// Create a new Parquet writer
    pub fn new() -> Self {
        Self {
            config: FormatConfig::default(),
        }
    }

    /// Create a Parquet writer with custom configuration
    pub fn with_config(config: FormatConfig) -> Self {
        Self { config }
    }

    /// Write a Dataset to a Parquet file.
    ///
    /// Feature columns are named from `featurenames` (or synthesised as
    /// `feature_N`). The optional target column is stored as `__target__`.
    pub fn write<P: AsRef<Path>>(&self, dataset: &Dataset, path: P) -> Result<()> {
        let _ = &self.config; // config reserved for future compression options
        write_dataset_parquet(dataset, path)
    }
}

#[cfg(feature = "formats")]
impl Default for ParquetWriter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// HDF5 Support
// ============================================================================

/// Convert from scirs2-io HDF5 error to DatasetsError.
#[cfg(feature = "formats")]
fn hdf5_err(msg: impl std::fmt::Display) -> DatasetsError {
    DatasetsError::InvalidFormat(format!("HDF5 error: {msg}"))
}

/// Read a `Dataset` from an HDF5 group structure produced by `write_hdf5_dataset`.
///
/// The feature matrix is read from the dataset named `dataset_name`; a
/// companion dataset `{dataset_name}_target` (if present) becomes the target.
#[cfg(feature = "formats")]
fn read_dataset_hdf5<P: AsRef<Path>>(path: P, dataset_name: &str) -> Result<Dataset> {
    use scirs2_io::hdf5::read_hdf5;

    let root = read_hdf5(path).map_err(hdf5_err)?;

    // Retrieve the feature matrix dataset.
    let ds = root.datasets.get(dataset_name).ok_or_else(|| {
        DatasetsError::InvalidFormat(format!("Dataset '{}' not found in HDF5 file", dataset_name))
    })?;

    let shape = &ds.shape;
    if shape.len() != 2 {
        return Err(DatasetsError::InvalidFormat(format!(
            "Expected 2-D dataset for '{}', got {}-D",
            dataset_name,
            shape.len()
        )));
    }
    let n_rows = shape[0];
    let n_cols = shape[1];

    let float_data = ds.as_float_vec().ok_or_else(|| {
        DatasetsError::InvalidFormat(format!(
            "Dataset '{}' contains non-numeric data",
            dataset_name
        ))
    })?;

    let data = Array2::from_shape_vec((n_rows, n_cols), float_data).map_err(|e| {
        DatasetsError::InvalidFormat(format!("Failed to shape feature matrix: {e}"))
    })?;

    // Optionally load the companion target dataset.
    let target_name = format!("{}_target", dataset_name);
    let target: Option<Array1<f64>> = if let Some(tds) = root.datasets.get(&target_name) {
        let tvec = tds.as_float_vec().ok_or_else(|| {
            DatasetsError::InvalidFormat(format!(
                "Target dataset '{}' contains non-numeric data",
                target_name
            ))
        })?;
        Some(Array1::from_vec(tvec))
    } else {
        None
    };

    Ok(Dataset::new(data, target))
}

/// Write a `Dataset` to an HDF5 file.
///
/// The feature matrix is stored as a 2-D dataset named `dataset_name`; if
/// the dataset has a target vector it is stored as `{dataset_name}_target`.
#[cfg(feature = "formats")]
fn write_dataset_hdf5<P: AsRef<Path>>(
    dataset: &Dataset,
    path: P,
    dataset_name: &str,
) -> Result<()> {
    use scirs2_core::ndarray::IxDyn;
    use scirs2_io::hdf5::write_hdf5;
    use std::collections::HashMap;

    let mut map: HashMap<String, scirs2_core::ndarray::ArrayD<f64>> = HashMap::new();

    // Store feature matrix as a flat ArrayD with 2-D shape.
    let n_rows = dataset.n_samples();
    let n_cols = dataset.n_features();
    let flat: Vec<f64> = dataset.data.iter().cloned().collect();
    let arr_dyn = scirs2_core::ndarray::ArrayD::from_shape_vec(IxDyn(&[n_rows, n_cols]), flat)
        .map_err(|e| {
            DatasetsError::InvalidFormat(format!("Failed to convert data to ArrayD: {e}"))
        })?;
    map.insert(dataset_name.to_string(), arr_dyn);

    // Store target vector if present.
    if let Some(target) = &dataset.target {
        let tvec: Vec<f64> = target.to_vec();
        let tlen = tvec.len();
        let tarr =
            scirs2_core::ndarray::ArrayD::from_shape_vec(IxDyn(&[tlen]), tvec).map_err(|e| {
                DatasetsError::InvalidFormat(format!("Failed to convert target to ArrayD: {e}"))
            })?;
        map.insert(format!("{}_target", dataset_name), tarr);
    }

    write_hdf5(path, map).map_err(hdf5_err)
}

#[cfg(feature = "formats")]
/// HDF5 reader for datasets
pub struct Hdf5Reader {
    config: FormatConfig,
}

#[cfg(feature = "formats")]
impl Hdf5Reader {
    /// Create a new HDF5 reader
    pub fn new() -> Self {
        Self {
            config: FormatConfig::default(),
        }
    }

    /// Create an HDF5 reader with custom configuration
    pub fn with_config(config: FormatConfig) -> Self {
        Self { config }
    }

    /// Read an HDF5 file into a Dataset.
    ///
    /// `dataset_name` is the name of the HDF5 dataset containing the 2-D
    /// feature matrix.  A companion dataset `{dataset_name}_target` (if
    /// present) is loaded as the target vector.
    pub fn read<P: AsRef<Path>>(&self, path: P, dataset_name: &str) -> Result<Dataset> {
        let _ = &self.config;
        read_dataset_hdf5(path, dataset_name)
    }
}

#[cfg(feature = "formats")]
impl Default for Hdf5Reader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "formats")]
/// HDF5 writer for datasets
pub struct Hdf5Writer {
    config: FormatConfig,
}

#[cfg(feature = "formats")]
impl Hdf5Writer {
    /// Create a new HDF5 writer
    pub fn new() -> Self {
        Self {
            config: FormatConfig::default(),
        }
    }

    /// Create an HDF5 writer with custom configuration
    pub fn with_config(config: FormatConfig) -> Self {
        Self { config }
    }

    /// Write a Dataset to an HDF5 file.
    ///
    /// The feature matrix is stored under `dataset_name`; an optional target
    /// is stored under `{dataset_name}_target`.
    pub fn write<P: AsRef<Path>>(
        &self,
        dataset: &Dataset,
        path: P,
        dataset_name: &str,
    ) -> Result<()> {
        let _ = &self.config;
        write_dataset_hdf5(dataset, path, dataset_name)
    }
}

#[cfg(feature = "formats")]
impl Default for Hdf5Writer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Format Conversion
// ============================================================================

#[cfg(feature = "formats")]
/// Convert between different data formats
pub struct FormatConverter {
    config: FormatConfig,
}

#[cfg(feature = "formats")]
impl FormatConverter {
    /// Create a new format converter
    pub fn new() -> Self {
        Self {
            config: FormatConfig::default(),
        }
    }

    /// Convert a dataset from one format to another
    pub fn convert<P1: AsRef<Path>, P2: AsRef<Path>>(
        &self,
        input_path: P1,
        input_format: FormatType,
        output_path: P2,
        output_format: FormatType,
    ) -> Result<()> {
        // Read in input format
        let dataset = match input_format {
            FormatType::Parquet => ParquetReader::new().read(input_path)?,
            FormatType::Hdf5 => Hdf5Reader::new().read(input_path, "data")?,
            FormatType::Csv => {
                return Err(DatasetsError::InvalidFormat(
                    "CSV reading via format converter not yet implemented".to_string(),
                ))
            }
            FormatType::Arrow => {
                return Err(DatasetsError::InvalidFormat(
                    "Arrow format not yet supported".to_string(),
                ))
            }
        };

        // Write in output format
        match output_format {
            FormatType::Parquet => ParquetWriter::new().write(&dataset, output_path)?,
            FormatType::Hdf5 => Hdf5Writer::new().write(&dataset, output_path, "data")?,
            FormatType::Csv => {
                return Err(DatasetsError::InvalidFormat(
                    "CSV writing via format converter not yet implemented".to_string(),
                ))
            }
            FormatType::Arrow => {
                return Err(DatasetsError::InvalidFormat(
                    "Arrow format not yet supported".to_string(),
                ))
            }
        }

        Ok(())
    }

    /// Auto-detect format and read
    pub fn read_auto<P: AsRef<Path>>(&self, path: P) -> Result<Dataset> {
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| DatasetsError::InvalidFormat("Invalid path".to_string()))?;

        let format = FormatType::from_extension(path_str)
            .ok_or_else(|| DatasetsError::InvalidFormat("Could not detect format".to_string()))?;

        match format {
            FormatType::Parquet => ParquetReader::new().read(path),
            FormatType::Hdf5 => Hdf5Reader::new().read(path, "data"),
            _ => Err(DatasetsError::InvalidFormat(format!(
                "Unsupported format: {:?}",
                format
            ))),
        }
    }
}

#[cfg(feature = "formats")]
impl Default for FormatConverter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Read a Parquet file
#[cfg(feature = "formats")]
pub fn read_parquet<P: AsRef<Path>>(path: P) -> Result<Dataset> {
    ParquetReader::new().read(path)
}

/// Write a Parquet file
#[cfg(feature = "formats")]
pub fn write_parquet<P: AsRef<Path>>(dataset: &Dataset, path: P) -> Result<()> {
    ParquetWriter::new().write(dataset, path)
}

/// Read an HDF5 file
#[cfg(feature = "formats")]
pub fn read_hdf5<P: AsRef<Path>>(path: P, dataset_name: &str) -> Result<Dataset> {
    Hdf5Reader::new().read(path, dataset_name)
}

/// Write an HDF5 file
#[cfg(feature = "formats")]
pub fn write_hdf5<P: AsRef<Path>>(dataset: &Dataset, path: P, dataset_name: &str) -> Result<()> {
    Hdf5Writer::new().write(dataset, path, dataset_name)
}

/// Auto-detect format and read
#[cfg(feature = "formats")]
pub fn read_auto<P: AsRef<Path>>(path: P) -> Result<Dataset> {
    FormatConverter::new().read_auto(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection() {
        assert_eq!(
            FormatType::from_extension("data.parquet"),
            Some(FormatType::Parquet)
        );
        assert_eq!(
            FormatType::from_extension("data.h5"),
            Some(FormatType::Hdf5)
        );
        assert_eq!(
            FormatType::from_extension("data.csv"),
            Some(FormatType::Csv)
        );
        assert_eq!(FormatType::from_extension("data.txt"), None);
    }

    #[test]
    fn test_format_extension() {
        assert_eq!(FormatType::Parquet.extension(), "parquet");
        assert_eq!(FormatType::Hdf5.extension(), "h5");
        assert_eq!(FormatType::Csv.extension(), "csv");
    }

    #[test]
    fn test_compression_codec() {
        assert_eq!(CompressionCodec::None.level(), None);
        assert_eq!(CompressionCodec::Snappy.level(), None);
        assert_eq!(CompressionCodec::Gzip.level(), Some(6));
        assert_eq!(CompressionCodec::Zstd.level(), Some(3));
    }

    #[test]
    fn test_format_config() {
        let config = FormatConfig::default();
        assert_eq!(config.chunk_size, 10_000);
        assert_eq!(config.compression, Some(CompressionCodec::Snappy));
        assert!(config.use_mmap);
    }

    // -----------------------------------------------------------------------
    // Parquet round-trip tests
    // -----------------------------------------------------------------------

    /// Write a Dataset to Parquet and read it back; verify shape and values.
    #[cfg(feature = "formats")]
    #[test]
    fn test_parquet_roundtrip_no_target() {
        use scirs2_core::ndarray::Array2;
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("shape");
        let ds = Dataset::new(data.clone(), None);

        let mut tmp = std::env::temp_dir();
        tmp.push("scirs2_test_parquet_roundtrip_no_target.parquet");

        write_parquet(&ds, &tmp).expect("parquet write");
        let recovered = read_parquet(&tmp).expect("parquet read");

        assert_eq!(recovered.n_samples(), 4, "n_samples mismatch");
        assert_eq!(recovered.n_features(), 3, "n_features mismatch");
        assert!(recovered.target.is_none(), "unexpected target");

        // Verify values element-wise (within f64 precision).
        for row in 0..4 {
            for col in 0..3 {
                let expected = data[[row, col]];
                let actual = recovered.data[[row, col]];
                assert!(
                    (expected - actual).abs() < 1e-10,
                    "mismatch at [{row},{col}]: expected {expected}, got {actual}"
                );
            }
        }

        let _ = std::fs::remove_file(&tmp);
    }

    /// Write a Dataset with target to Parquet and read it back.
    #[cfg(feature = "formats")]
    #[test]
    fn test_parquet_roundtrip_with_target() {
        use scirs2_core::ndarray::{Array1, Array2};
        let data =
            Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("shape");
        let target = Some(Array1::from_vec(vec![0.0, 1.0, 0.0]));
        let ds = Dataset::new(data.clone(), target.clone());

        let mut tmp = std::env::temp_dir();
        tmp.push("scirs2_test_parquet_roundtrip_with_target.parquet");

        write_parquet(&ds, &tmp).expect("parquet write");
        let recovered = read_parquet(&tmp).expect("parquet read");

        assert_eq!(recovered.n_samples(), 3);
        assert_eq!(recovered.n_features(), 2);
        assert!(
            recovered.target.is_some(),
            "target missing after round-trip"
        );

        let rtarget = recovered.target.as_ref().expect("target");
        assert_eq!(rtarget.len(), 3);
        for (i, (&expected, &actual)) in target
            .as_ref()
            .expect("t")
            .iter()
            .zip(rtarget.iter())
            .enumerate()
        {
            assert!(
                (expected - actual).abs() < 1e-10,
                "target mismatch at [{i}]: expected {expected}, got {actual}"
            );
        }

        let _ = std::fs::remove_file(&tmp);
    }

    /// Feature names survive the Parquet round-trip.
    #[cfg(feature = "formats")]
    #[test]
    fn test_parquet_roundtrip_feature_names() {
        use scirs2_core::ndarray::Array2;
        let data = Array2::from_shape_vec((2, 2), vec![10.0, 20.0, 30.0, 40.0]).expect("shape");
        let mut ds = Dataset::new(data, None);
        ds.featurenames = Some(vec!["alpha".to_string(), "beta".to_string()]);

        let mut tmp = std::env::temp_dir();
        tmp.push("scirs2_test_parquet_feature_names.parquet");

        write_parquet(&ds, &tmp).expect("parquet write");
        let recovered = read_parquet(&tmp).expect("parquet read");

        let names = recovered.featurenames.as_ref().expect("featurenames");
        assert_eq!(names, &["alpha", "beta"]);

        let _ = std::fs::remove_file(&tmp);
    }

    // -----------------------------------------------------------------------
    // HDF5 round-trip tests
    // -----------------------------------------------------------------------

    /// Write a Dataset to HDF5 and read it back.
    #[cfg(feature = "formats")]
    #[test]
    fn test_hdf5_roundtrip_no_target() {
        use scirs2_core::ndarray::Array2;
        let data = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("shape");
        let ds = Dataset::new(data.clone(), None);

        let mut tmp = std::env::temp_dir();
        tmp.push("scirs2_test_hdf5_roundtrip_no_target.h5");

        write_hdf5(&ds, &tmp, "mydata").expect("hdf5 write");
        let recovered = read_hdf5(&tmp, "mydata").expect("hdf5 read");

        assert_eq!(recovered.n_samples(), 3, "n_samples mismatch");
        assert_eq!(recovered.n_features(), 4, "n_features mismatch");
        assert!(recovered.target.is_none());

        for row in 0..3 {
            for col in 0..4 {
                let expected = data[[row, col]];
                let actual = recovered.data[[row, col]];
                assert!(
                    (expected - actual).abs() < 1e-10,
                    "mismatch [{row},{col}]: {expected} != {actual}"
                );
            }
        }

        let _ = std::fs::remove_file(&tmp);
        // The no-hdf5-feature path also creates a sidecar JSON.
        let sidecar = format!("{}.json", tmp.to_string_lossy());
        let _ = std::fs::remove_file(&sidecar);
    }

    /// Write a Dataset with target to HDF5 and read it back.
    #[cfg(feature = "formats")]
    #[test]
    fn test_hdf5_roundtrip_with_target() {
        use scirs2_core::ndarray::{Array1, Array2};
        let data =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("shape");
        let target = Some(Array1::from_vec(vec![1.0, 0.0]));
        let ds = Dataset::new(data.clone(), target.clone());

        let mut tmp = std::env::temp_dir();
        tmp.push("scirs2_test_hdf5_roundtrip_with_target.h5");

        write_hdf5(&ds, &tmp, "experiment").expect("hdf5 write");
        let recovered = read_hdf5(&tmp, "experiment").expect("hdf5 read");

        assert_eq!(recovered.n_samples(), 2);
        assert_eq!(recovered.n_features(), 3);
        assert!(recovered.target.is_some());

        let rtarget = recovered.target.as_ref().expect("target");
        assert_eq!(rtarget.len(), 2);
        for (i, (&e, &a)) in target
            .as_ref()
            .expect("t")
            .iter()
            .zip(rtarget.iter())
            .enumerate()
        {
            assert!((e - a).abs() < 1e-10, "target mismatch [{i}]: {e} != {a}");
        }

        let _ = std::fs::remove_file(&tmp);
        let sidecar = format!("{}.json", tmp.to_string_lossy());
        let _ = std::fs::remove_file(&sidecar);
    }
}
