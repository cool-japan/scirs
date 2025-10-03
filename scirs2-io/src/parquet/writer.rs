//! Parquet file writing

use crate::error::{IoError, Result};
use crate::parquet::conversion::{ndarray_to_arrow, ToArrowArray};
use crate::parquet::options::ParquetWriteOptions;
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use scirs2_core::ndarray::{ArrayBase, Data as NdData, Dimension};
use std::fs::File;
use std::path::Path;

/// Parquet file writer
pub struct ParquetWriter {
    writer: ArrowWriter<File>,
    options: ParquetWriteOptions,
}

impl ParquetWriter {
    /// Create a new Parquet writer
    pub fn from_path<P: AsRef<Path>>(
        path: P,
        schema: arrow::datatypes::SchemaRef,
        options: ParquetWriteOptions,
    ) -> Result<Self> {
        let file = File::create(path.as_ref()).map_err(|e| {
            IoError::FileError(format!(
                "Failed to create file '{}': {}",
                path.as_ref().display(),
                e
            ))
        })?;

        let props = options.to_writer_properties();
        let writer = ArrowWriter::try_new(file, schema, Some(props)).map_err(|e| {
            IoError::ParquetError(format!("Failed to create Parquet writer: {}", e))
        })?;

        Ok(Self { writer, options })
    }

    /// Write a RecordBatch to the file
    pub fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.writer
            .write(batch)
            .map_err(|e| IoError::ParquetError(format!("Failed to write batch: {}", e)))
    }

    /// Finalize and close the writer
    pub fn close(mut self) -> Result<()> {
        self.writer
            .close()
            .map(|_| ())
            .map_err(|e| IoError::ParquetError(format!("Failed to close writer: {}", e)))
    }

    /// Get a reference to the write options
    pub fn options(&self) -> &ParquetWriteOptions {
        &self.options
    }
}

/// Write ndarray data to a Parquet file
///
/// # Arguments
///
/// * `path` - Path to the output Parquet file
/// * `data` - ndarray data to write
/// * `options` - Write options (compression, row group size, etc.)
///
/// # Examples
///
/// ```rust,no_run
/// use scirs2_io::parquet::{write_parquet, ParquetWriteOptions, CompressionCodec};
/// use scirs2_core::ndarray::Array1;
///
/// let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let options = ParquetWriteOptions::with_compression(CompressionCodec::Zstd);
/// write_parquet("output.parquet", &data, options)?;
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
pub fn write_parquet<P, S, D, T>(
    path: P,
    data: &ArrayBase<S, D>,
    options: ParquetWriteOptions,
) -> Result<()>
where
    P: AsRef<Path>,
    S: NdData<Elem = T>,
    D: Dimension,
    T: ToArrowArray + Clone,
{
    // Convert ndarray to Arrow RecordBatch
    let batch = ndarray_to_arrow(data, "value")?;
    let schema = batch.schema();

    // Create writer
    let mut writer = ParquetWriter::from_path(path, schema, options)?;

    // Write batch
    writer.write_batch(&batch)?;

    // Close and finalize
    writer.close()?;

    Ok(())
}

/// Write ndarray data to a Parquet file with custom column name
///
/// # Arguments
///
/// * `path` - Path to the output Parquet file
/// * `data` - ndarray data to write
/// * `column_name` - Name for the column
/// * `options` - Write options
///
/// # Examples
///
/// ```rust,no_run
/// use scirs2_io::parquet::{write_parquet_with_name, ParquetWriteOptions};
/// use scirs2_core::ndarray::Array1;
///
/// let temperatures = Array1::from_vec(vec![20.5, 21.3, 19.8, 22.1]);
/// write_parquet_with_name(
///     "temperatures.parquet",
///     &temperatures,
///     "temperature_celsius",
///     Default::default()
/// )?;
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
pub fn write_parquet_with_name<P, S, D, T>(
    path: P,
    data: &ArrayBase<S, D>,
    column_name: &str,
    options: ParquetWriteOptions,
) -> Result<()>
where
    P: AsRef<Path>,
    S: NdData<Elem = T>,
    D: Dimension,
    T: ToArrowArray + Clone,
{
    // Convert ndarray to Arrow RecordBatch
    let batch = ndarray_to_arrow(data, column_name)?;
    let schema = batch.schema();

    // Create writer
    let mut writer = ParquetWriter::from_path(path, schema, options)?;

    // Write batch
    writer.write_batch(&batch)?;

    // Close and finalize
    writer.close()?;

    Ok(())
}

/// Write multiple ndarray columns to a Parquet file
///
/// # Arguments
///
/// * `path` - Path to the output Parquet file
/// * `batches` - Vector of RecordBatches to write
/// * `options` - Write options
///
/// # Note
///
/// This is a low-level function for writing multiple batches.
/// For simple single-array writes, use `write_parquet` instead.
pub fn write_parquet_batches<P: AsRef<Path>>(
    path: P,
    batches: Vec<RecordBatch>,
    options: ParquetWriteOptions,
) -> Result<()> {
    if batches.is_empty() {
        return Err(IoError::ParquetError(
            "Cannot write empty batches".to_string(),
        ));
    }

    let schema = batches[0].schema();

    // Validate all batches have same schema
    for batch in &batches[1..] {
        if batch.schema() != schema {
            return Err(IoError::ParquetError(
                "All batches must have the same schema".to_string(),
            ));
        }
    }

    // Create writer
    let mut writer = ParquetWriter::from_path(path, schema, options)?;

    // Write all batches
    for batch in &batches {
        writer.write_batch(batch)?;
    }

    // Close and finalize
    writer.close()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::options::CompressionCodec;
    use crate::parquet::reader::read_parquet;
    use scirs2_core::ndarray::Array1;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_write_parquet_f64() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.parquet");

        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        write_parquet(&path, &data, Default::default()).unwrap();

        assert!(path.exists());
        assert!(fs::metadata(&path).unwrap().len() > 0);
    }

    #[test]
    fn test_write_with_compression() {
        let dir = tempdir().unwrap();
        let data = Array1::from_vec((0..100).map(|x| x as f64).collect::<Vec<_>>());

        let codecs = [
            CompressionCodec::Uncompressed,
            CompressionCodec::Snappy,
            CompressionCodec::Gzip,
        ];

        for codec in codecs {
            let path = dir.path().join(format!("test_{:?}.parquet", codec));
            let options = ParquetWriteOptions::with_compression(codec);
            write_parquet(&path, &data, options).unwrap();

            assert!(path.exists());
            assert!(fs::metadata(&path).unwrap().len() > 0);
        }
    }

    #[test]
    fn test_write_with_custom_name() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("named.parquet");

        let data = Array1::from_vec(vec![10.5, 20.3, 30.1]);
        write_parquet_with_name(&path, &data, "measurements", Default::default()).unwrap();

        let loaded = read_parquet(&path).unwrap();
        assert_eq!(loaded.column_names()[0], "measurements");
    }

    #[test]
    fn test_write_i32() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("integers.parquet");

        let data = Array1::from_vec(vec![1i32, 2, 3, 4, 5]);
        write_parquet(&path, &data, Default::default()).unwrap();

        assert!(path.exists());
    }

    #[test]
    fn test_write_options_builder() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("configured.parquet");

        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let options = ParquetWriteOptions::with_compression(CompressionCodec::Zstd)
            .with_row_group_size(1000)
            .with_dictionary(true);

        write_parquet(&path, &data, options).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("roundtrip.parquet");

        let original = Array1::from_vec(vec![1.5, 2.7, 3.9, 4.2, 5.1]);
        write_parquet(&path, &original, Default::default()).unwrap();

        let loaded = read_parquet(&path).unwrap();
        let recovered = loaded.get_column_f64("value").unwrap();

        assert_eq!(recovered.len(), original.len());
        for (a, b) in recovered.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_empty_batches_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.parquet");

        let result = write_parquet_batches(&path, vec![], Default::default());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Cannot write empty batches"));
    }
}
