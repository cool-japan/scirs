//! Parquet write options and configuration

use serde::{Deserialize, Serialize};

/// Compression codec for Parquet files
///
/// Note: Only `Uncompressed` is available by default. Other codecs (`Snappy`, `Gzip`,
/// `Lz4`, `Zstd`, `Brotli`, `Lz4Raw`) require enabling the corresponding
/// feature flags on the `parquet` dependency (`snap`, `flate2-zlib-rs`, `lz4`,
/// `zstd`, `brotli`). Attempting to write with an unavailable codec will panic
/// at runtime with "Disabled feature at compile time: <codec>".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CompressionCodec {
    /// No compression (always available, no feature flags required)
    #[default]
    Uncompressed,
    /// Snappy compression (fast, moderate compression) — requires `snap` parquet feature
    Snappy,
    /// Gzip compression (slower, better compression) — requires `flate2-zlib-rs` or `flate2-rust_backened` parquet feature
    Gzip,
    /// LZ4 compression (very fast, moderate compression) — requires `lz4` parquet feature
    Lz4,
    /// ZSTD compression (good balance of speed and compression) — requires `zstd` parquet feature
    Zstd,
    /// Brotli compression (slower, best compression) — requires `brotli` parquet feature
    Brotli,
    /// LZ4 raw format — requires `lz4` parquet feature
    Lz4Raw,
}

impl CompressionCodec {
    /// Convert to parquet compression type
    pub fn to_parquet_compression(self) -> parquet::basic::Compression {
        match self {
            Self::Uncompressed => parquet::basic::Compression::UNCOMPRESSED,
            Self::Snappy => parquet::basic::Compression::SNAPPY,
            Self::Gzip => parquet::basic::Compression::GZIP(Default::default()),
            Self::Lz4 => parquet::basic::Compression::LZ4,
            Self::Zstd => parquet::basic::Compression::ZSTD(Default::default()),
            Self::Brotli => parquet::basic::Compression::BROTLI(Default::default()),
            Self::Lz4Raw => parquet::basic::Compression::LZ4_RAW,
        }
    }
}

/// Parquet file version
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ParquetVersion {
    /// Parquet version 1.0
    V1,
    /// Parquet version 2.0 (supports more encoding types)
    #[default]
    V2,
}

impl ParquetVersion {
    /// Convert to parquet writer version
    pub fn to_parquet_version(self) -> parquet::file::properties::WriterVersion {
        match self {
            Self::V1 => parquet::file::properties::WriterVersion::PARQUET_1_0,
            Self::V2 => parquet::file::properties::WriterVersion::PARQUET_2_0,
        }
    }
}

/// Options for writing Parquet files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParquetWriteOptions {
    /// Compression codec to use
    pub compression: CompressionCodec,

    /// Row group size (number of rows per row group)
    /// Larger values = better compression, more memory usage
    pub row_group_size: usize,

    /// Data page size in bytes
    /// Typical value: 1MB
    pub data_page_size: usize,

    /// Enable dictionary encoding for string/binary columns
    /// Improves compression for columns with repeated values
    pub enable_dictionary: bool,

    /// Enable statistics collection for each column
    /// Used for predicate pushdown and query optimization
    pub enable_statistics: bool,

    /// Parquet file format version
    pub version: ParquetVersion,

    /// Writer batch size (for buffering)
    pub write_batch_size: usize,
}

impl Default for ParquetWriteOptions {
    fn default() -> Self {
        Self {
            compression: CompressionCodec::default(),
            row_group_size: 1024 * 1024, // 1M rows
            data_page_size: 1024 * 1024, // 1MB
            enable_dictionary: true,
            enable_statistics: true,
            version: ParquetVersion::default(),
            write_batch_size: 1024,
        }
    }
}

impl ParquetWriteOptions {
    /// Create new options with specified compression
    pub fn with_compression(compression: CompressionCodec) -> Self {
        Self {
            compression,
            ..Default::default()
        }
    }

    /// Set row group size
    pub fn with_row_group_size(mut self, size: usize) -> Self {
        self.row_group_size = size;
        self
    }

    /// Set data page size
    pub fn with_data_page_size(mut self, size: usize) -> Self {
        self.data_page_size = size;
        self
    }

    /// Enable or disable dictionary encoding
    pub fn with_dictionary(mut self, enable: bool) -> Self {
        self.enable_dictionary = enable;
        self
    }

    /// Enable or disable statistics
    pub fn with_statistics(mut self, enable: bool) -> Self {
        self.enable_statistics = enable;
        self
    }

    /// Set Parquet version
    pub fn with_version(mut self, version: ParquetVersion) -> Self {
        self.version = version;
        self
    }

    /// Convert to parquet WriterProperties
    pub fn to_writer_properties(&self) -> parquet::file::properties::WriterProperties {
        parquet::file::properties::WriterProperties::builder()
            .set_compression(self.compression.to_parquet_compression())
            .set_max_row_group_row_count(Some(self.row_group_size))
            .set_data_page_size_limit(self.data_page_size)
            .set_dictionary_enabled(self.enable_dictionary)
            .set_statistics_enabled(if self.enable_statistics {
                parquet::file::properties::EnabledStatistics::Page
            } else {
                parquet::file::properties::EnabledStatistics::None
            })
            .set_writer_version(self.version.to_parquet_version())
            .set_write_batch_size(self.write_batch_size)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_codec_conversion() {
        // Only Uncompressed is always available without additional parquet feature flags.
        // Other codecs (Snappy, Gzip, Lz4, Brotli, Zstd) require optional features.
        let codec = CompressionCodec::Uncompressed;
        let parquet_compression = codec.to_parquet_compression();
        assert!(matches!(
            parquet_compression,
            parquet::basic::Compression::UNCOMPRESSED
        ));
    }

    #[test]
    fn test_default_options() {
        let options = ParquetWriteOptions::default();
        assert_eq!(options.compression, CompressionCodec::Uncompressed);
        assert_eq!(options.version, ParquetVersion::V2);
        assert!(options.enable_dictionary);
        assert!(options.enable_statistics);
    }

    #[test]
    fn test_builder_pattern() {
        let options = ParquetWriteOptions::with_compression(CompressionCodec::Uncompressed)
            .with_row_group_size(50000)
            .with_dictionary(false);

        assert_eq!(options.compression, CompressionCodec::Uncompressed);
        assert_eq!(options.row_group_size, 50000);
        assert!(!options.enable_dictionary);
    }
}
