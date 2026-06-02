# scirs2-io

[![crates.io](https://img.shields.io/crates/v/scirs2-io.svg)](https://crates.io/crates/scirs2-io)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-io)](https://docs.rs/scirs2-io)

**Scientific data input/output for the SciRS2 scientific computing library (v0.5.0).**

`scirs2-io` provides comprehensive, high-performance file I/O for scientific and numerical workloads. It covers everything from classic scientific formats (MATLAB, NetCDF, HDF5, WAV) through modern columnar and streaming formats (Parquet, Arrow IPC, NDJSON, ORC) to cloud storage, ETL pipelines, data catalogs, and lineage tracking — all as pure Rust with no C/Fortran dependencies.

## Features (v0.5.0)

### Classic Scientific Formats
- **MATLAB (.mat)**: Full `.mat` v4/v5 read/write with all data types, structures, and cell arrays
- **WAV**: Professional-grade WAV audio read/write
- **ARFF**: Complete Weka Attribute-Relation File Format support
- **NetCDF**: NetCDF3 Classic and NetCDF4/HDF5 with unlimited dimensions and chunking
- **HDF5-lite**: Pure-Rust hierarchical data format reader; group hierarchies, attributes, datasets
- **Matrix Market / Harwell-Boeing**: Sparse and dense matrix exchange formats

### Modern Columnar and Binary Formats
- **Parquet-lite**: Pure-Rust Parquet reader for columnar analytical data
- **Feather (Arrow IPC)**: Memory-mapped Arrow columnar format, zero-copy read/write
- **ORC**: ORC columnar format reader
- **Binary format utilities**: Portable binary encoding helpers

### Serialization Formats
- **CBOR**: Concise Binary Object Representation (RFC 7049) serialization and deserialization
- **BSON**: Binary JSON as used by MongoDB ecosystems
- **Avro**: Schema-based binary serialization with schema evolution support
- **Protobuf-lite**: Pure-Rust protocol buffer encoding and decoding (no C codegen dependency)
- **MessagePack**: Compact, fast binary serialization
- **JSON / NDJSON**: Standard JSON and Newline-Delimited JSON (streaming-friendly)

### Streaming and Lazy Evaluation
- **Streaming CSV**: Lazy, chunk-by-chunk CSV processing; handles files larger than RAM
- **Streaming JSON**: Incremental JSON parser for large document streams
- **NDJSON streaming**: Line-by-line newline-delimited JSON processing
- **Arrow IPC streaming**: Framed streaming Arrow record batches
- **Backpressure-aware pipelines**: Push-pull pipeline with configurable buffer sizes

### Compression
- **LZ4**: High-speed lossless compression
- **Zstd**: Zstandard compression with configurable levels
- **Brotli**: General-purpose compression (HTTP-friendly)
- **Snappy**: Google Snappy for fast block compression
- **GZIP / BZIP2**: Classic deflate-based and BWT-based compression
- **Parallel compression**: Chunk-parallel encode/decode with up to 2.5x throughput improvement

### Data Catalog, Lineage, and Governance
- **Data catalog**: Register, tag, and discover datasets by name and schema
- **Lineage tracking**: Record transformations and data provenance for auditability
- **Schema registry**: Store and evolve data schemas; validate records on read/write
- **Versioning support**: Immutable dataset versioning with diff and rollback

### ETL and Query
- **ETL pipeline framework**: Declarative source → transform → sink pipelines with parallel stages
- **Query interface**: SQL-like predicate pushdown and projection for tabular formats
- **Universal reader**: Automatic format detection from magic bytes and file extension
- **Format detection**: Detect dozens of formats without specifying them explicitly

### Cloud and Distributed I/O
- **Cloud storage connectors**: Framework for AWS S3, Google Cloud Storage, and Azure Blob Storage
- **Distributed I/O**: Partitioned parallel read/write for cluster workloads

### Validation and Integrity
- **Checksums**: CRC32, SHA-256, BLAKE3 integrity verification
- **Schema-based validation**: JSON Schema-compatible validation engine
- **Format validators**: Format-specific structural validators

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2-io = "0.5.0"
```

To enable optional feature groups:

```toml
[dependencies]
scirs2-io = { version = "0.5.0", features = ["async", "compression"] }
```

### Reading a CSV file

```rust
use scirs2_io::csv::{read_csv, CsvReaderConfig};

let config = CsvReaderConfig {
    has_header: true,
    delimiter: ',',
    ..Default::default()
};
let (headers, data) = read_csv("dataset.csv", Some(config))?;
println!("Columns: {:?}", headers);
println!("Shape: {:?}", data.shape());
```

### Streaming a large NDJSON file

```rust
use scirs2_io::ndjson_streaming::NdjsonReader;

let reader = NdjsonReader::open("events.ndjson")?;
for record in reader {
    let obj = record?;
    // process each JSON object without loading the whole file
}
```

### CBOR serialization round-trip

```rust
use scirs2_io::formats::cbor::{encode_cbor, decode_cbor};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct Measurement { sensor_id: u32, value: f64 }

let m = Measurement { sensor_id: 42, value: 3.14 };
let bytes = encode_cbor(&m)?;
let decoded: Measurement = decode_cbor(&bytes)?;
```

### Streaming CSV with lazy evaluation

```rust
use scirs2_io::streaming_csv::StreamingCsvReader;

let mut reader = StreamingCsvReader::open("large.csv")?;
reader.set_chunk_size(65_536);

while let Some(chunk) = reader.next_chunk()? {
    // each chunk is a small Vec<Vec<String>>; process without buffering the file
    println!("chunk rows: {}", chunk.len());
}
```

### Parallel Zstd compression

```rust
use scirs2_io::compression::advanced::{compress_parallel, CompressionAlgorithm, ParallelConfig};

let data = std::fs::read("large_array.bin")?;
let cfg = ParallelConfig { num_threads: 8, chunk_size: 1 << 20, ..Default::default() };
let compressed = compress_parallel(&data, CompressionAlgorithm::Zstd, 6, cfg)?;
println!("compressed to {}%", 100 * compressed.len() / data.len());
```

## API Overview

| Module | Purpose |
|--------|---------|
| `matlab` | `.mat` file read/write |
| `wavfile` | WAV audio read/write |
| `netcdf` | NetCDF3/4 |
| `hdf5_lite` | Pure-Rust HDF5-lite |
| `csv` | CSV with type inference |
| `image` | PNG, JPEG, BMP, TIFF |
| `matrix_market` | Matrix Market / Harwell-Boeing |
| `formats::cbor` | CBOR encode/decode |
| `formats::bson` | BSON encode/decode |
| `formats::avro` | Avro schema-based serialization |
| `formats::protobuf` | Protobuf-lite encode/decode |
| `formats::msgpack` | MessagePack encode/decode |
| `formats::orc` | ORC reader |
| `formats::feather` | Feather (Arrow IPC) |
| `parquet_lite` | Parquet-lite reader |
| `streaming_csv` | Lazy streaming CSV |
| `streaming_json` | Streaming JSON parser |
| `ndjson_streaming` | NDJSON line-by-line streaming |
| `arrow_ipc` / `arrow_streaming` | Arrow IPC framed streaming |
| `binary_format` | Binary encoding utilities |
| `compression` / `compression_utils` | LZ4, Zstd, Brotli, Snappy, GZIP, BZIP2 |
| `universal_reader` | Auto-detect format and open |
| `format_detect` | Magic bytes / extension detection |
| `query` | SQL-like predicate and projection |
| `catalog` | Data catalog |
| `lineage` | Provenance and lineage tracking |
| `schema` | Schema registry |
| `versioning` | Dataset versioning |
| `etl` | ETL pipeline framework |
| `cloud` | Cloud storage connectors |
| `distributed` | Distributed / partitioned I/O |
| `pipeline` | Typed streaming pipeline (sources, transforms, sinks) |
| `validation` | Checksums, schema validation |
| `serialize` | Array and sparse-matrix serialization |
| `streaming` | Low-level streaming utilities |

## Feature Flags

| Feature | Description |
|---------|-------------|
| `default` | CSV, compression, validation |
| `async` | Async I/O via tokio |
| `reqwest` | HTTP/HTTPS network I/O |
| `compression` | LZ4, Zstd, Brotli, Snappy (enabled by default) |
| `hdf5` | HDF5 system library binding (optional; hdf5-lite is always available) |
| `all` | All features |

## Documentation

Full API documentation is available at [docs.rs/scirs2-io](https://docs.rs/scirs2-io).

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
