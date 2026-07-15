# scirs2-io

[![crates.io](https://img.shields.io/crates/v/scirs2-io.svg)](https://crates.io/crates/scirs2-io)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-io)](https://docs.rs/scirs2-io)
[![Status](https://img.shields.io/badge/status-partial-yellow)]()

**Scientific data input/output for the SciRS2 scientific computing library (v0.6.1).**

`scirs2-io` provides comprehensive, high-performance file I/O for scientific and numerical workloads. It covers everything from classic scientific formats (MATLAB, NetCDF, HDF5, WAV, NumPy, Fortran unformatted, IDL) through domain-specific formats (FITS/VOTable, FASTA/FASTQ/SAM/BAM/VCF, GeoTIFF/Shapefile/GeoJSON/KML), modern columnar and lakehouse formats (Parquet, Arrow IPC, Zarr, Delta Lake, Iceberg), to cloud storage, ETL pipelines, data catalogs, and lineage tracking — all as pure Rust with no required C/Fortran dependencies (a few optional interop paths, such as the system `hdf5` library binding, are feature-gated and off by default).

## Features (v0.6.1)

### Classic Scientific Formats
- **MATLAB (.mat)**: `.mat` v4/v5 read/write with structures and cell arrays
- **WAV**: Professional-grade WAV audio read/write
- **ARFF**: Complete Weka Attribute-Relation File Format support
- **NetCDF**: Hand-rolled, pure-Rust NetCDF3 Classic and NetCDF4/HDF5 reader/writer with unlimited dimensions and chunking (`netcdf/`, 1300+ lines; does **not** wrap the external `netcdf3` crate — that dependency is currently unused dead weight in `Cargo.toml`)
- **HDF5-lite**: Pure-Rust, read-only hierarchical data format reader (v0/v1 superblocks, B-tree group navigation, contiguous/chunked datasets, attributes); the optional `hdf5` feature additionally binds the system C `hdf5` library for write support and advanced layouts
- **NumPy (.npy/.npz)**: NumPy binary array format and ZIP-of-arrays archives
- **Fortran unformatted files**: Read/write support for Fortran sequential unformatted records, common in physics/weather/engineering codes
- **IDL save files (.sav)**: Reader/writer for Interactive Data Language save files used in astronomy and remote sensing
- **Matrix Market / Harwell-Boeing**: Sparse and dense matrix exchange formats
- **Sparse matrix formats**: Unified COO/CSR/CSC support with conversions between representations

### Domain-Specific Scientific Formats
- **Astronomical**: FITS (Flexible Image Transport System) image/table read-write, VOTable (Virtual Observatory Table) column access
- **Bioinformatics**: FASTA and FASTQ sequence read/write (with quality scores), SAM/BAM alignment format, VCF variant call format
- **Geospatial**: GeoTIFF (georeferenced raster bands/windows), Shapefile, GeoJSON, KML/KMZ read/write

### Modern Columnar, Lakehouse, and Binary Formats
- **Parquet** (`parquet` feature): real Apache Arrow/Parquet-ecosystem reader/writer (via the `arrow` and `parquet` crates) with schema inference, predicate pushdown, row-group statistics, and Pandas/Polars/PyArrow interoperability
- **Parquet-lite**: always-available, pure-Rust simplified columnar format that borrows Parquet's structural ideas (magic bytes, row groups, footer) without the `parquet`/`arrow` dependency — not binary-compatible with third-party `.parquet` files
- **Feather (Arrow IPC)**: Memory-mapped Arrow columnar format, zero-copy read/write
- **ORC-style columnar formats**: two self-contained pure-Rust formats (`formats::orc`, `formats::orc_lite`) supporting RLE v2 integer encoding (direct/delta/variable-length), dictionary string encoding, and bit-packed boolean RLE — inspired by Apache ORC's on-disk design but using their own magic bytes/framing, not byte-compatible with third-party `.orc` files
- **Zarr v2/v3**: Chunked, compressed, N-dimensional array store compatible with Zarr-Python
- **TileDB-inspired arrays**: Pure-Rust dense/sparse multidimensional array storage with tile-based I/O (row-major/column-major/Hilbert ordering), not a TileDB file-format client
- **Lance-inspired columnar format**: Pure-Rust, in-process reader/writer for a Lance-like ML-dataset columnar layout
- **Delta Lake**: JSON transaction-log (`_delta_log/`) reader/writer with append commits, version conflict detection, checkpoint/replay, and time travel
- **Apache Iceberg** (simplified): in-memory table abstraction with schema, snapshot-based versioning, and JSON metadata serialization
- **Binary format utilities**: Portable typed-array and columnar binary encoding helpers

### Serialization Formats
- **CBOR**: Concise Binary Object Representation (RFC 7049), own `CborValue` data model with incremental encoder/decoder plus `encode_cbor`/`decode_cbor` convenience functions (not a `serde`-generic `encode(&T)` API)
- **BSON**: Binary JSON as used by MongoDB ecosystems (all floating point stored as BSON `Double`; `f32` values upcast to `f64`, matching the BSON type system)
- **Avro**: Schema-based binary serialization with schema evolution support
- **Protobuf**: Two-tier pure-Rust implementation — low-level varint primitives (`protobuf_lite`) plus a higher-level `ProtoMessage`/`ProtoValue` encoder/decoder (`formats::protobuf`); no `protoc` code-generation dependency
- **MessagePack**: Compact, fast binary serialization (via `rmp-serde`)
- **JSON / NDJSON**: Standard JSON and Newline-Delimited JSON (streaming-friendly)

### Streaming and Lazy Evaluation
- **Streaming CSV**: Lazy, chunk-by-chunk CSV processing via `CsvStreamReader`; handles files larger than RAM
- **Streaming JSON**: Incremental JSON parser for large document streams
- **NDJSON streaming**: Line-by-line newline-delimited JSON processing
- **Arrow IPC streaming**: Framed streaming Arrow record batches; unexpected/unknown message tags now raise a `FormatError` instead of being silently skipped
- **Backpressure-aware pipelines**: Push-pull pipeline with configurable buffer sizes
- **Windowed aggregation**: Tumbling, sliding, and session windows over event-time streams
- **Watermarks**: Watermark-based late-data handling for windowed operators
- **Checkpointing**: Periodic operator-state snapshots for restartable streaming jobs
- **Exactly-once sinks**: Write-ahead log (disk or in-memory) plus idempotency-key deduplication

### Compression
- **LZ4 / Zstd / Brotli / Snappy / GZIP / BZIP2**: Pure-Rust codecs (via the `oxiarc-*` family) — no zip/flate2/zstd/bzip2/lz4/miniz_oxide C dependencies
- **Parallel compression**: `compress_data_parallel`/`decompress_data_parallel`/`compress_file_parallel`/`decompress_file_parallel` in the `compression` module split input into chunks and compress them concurrently via Rayon (`ParallelCompressionConfig`, `ParallelCompressionStats`)
- **Adaptive compression**: Runtime Shannon-entropy profiling auto-selects LZ4/Zstd/Brotli; `auto_compress`/`auto_decompress` embed a 1-byte algorithm tag
- **GPU-assisted compression**: backend detection/capability probing for CUDA, Metal, and OpenCL (`gpu` module); the OpenCL compression header now reports the adapter's real compute-unit count instead of a hardcoded placeholder

### Data Catalog, Lineage, and Governance
- **Data catalog**: Register, tag, and discover datasets by name and schema
- **Lineage tracking**: Record transformations and data provenance for auditability
- **Schema module**: `Schema`/`SchemaField`/`SchemaBuilder` definitions, validation, and type inference against data
- **Protobuf schema registry**: separate, versioned `SchemaRegistry` with compatibility checks for Protobuf-shaped schemas (pure Rust, no `prost`)
- **Versioning support**: Immutable dataset versioning with diff and rollback
- **Metadata management**: Unified metadata read/write across supported formats
- **TOML configuration utilities**: Typed config wrapper, deep-merge, flattening, and validation on top of the `toml` crate

### ETL, Query, and Analytics
- **ETL pipeline framework**: Declarative source → transform → sink pipelines with parallel stages
- **Query interface**: SQL-like predicate pushdown and projection for tabular formats
- **DataFusion-compatible table provider**: standalone abstraction mirroring the Apache DataFusion `TableProvider` trait without depending on the `datafusion`/`arrow` crates (two parallel implementations exist: `table_provider` and `datafusion_provider`)
- **Vectorized expression evaluation**: Row-at-a-time and SIMD-friendly filter/project evaluation over record batches
- **Join algorithms**: Hash join, sort-merge join, and nested-loop join across record batches
- **Approximate analytics**: Bloom filter column indexes, HyperLogLog++ cardinality estimation, t-digest quantiles, count-min sketch frequency estimation
- **Columnar encodings**: Dictionary encoding, run-length encoding, delta/zigzag/varint encoding, FSST string compression
- **Universal reader**: Automatic format detection from magic bytes and file extension
- **Format detection**: Detect dozens of formats without specifying them explicitly

### Streaming Protocols and Messaging
- **Real network clients** (`realtime` feature, built on real crates): WebSocket (`tokio-tungstenite`), Server-Sent Events (`eventsource-client`), gRPC (`tonic`), and MQTT (`rumqttc`) client support
- **In-memory protocol simulations** (for testing/offline development, no network involved): a Kafka-protocol in-memory broker (`protocols::kafka`), an Arrow Flight in-memory simulation (`protocols::arrow_flight`), and an in-process MQTT-style topic broker with full wildcard matching (`mqtt_broker`)

### Machine Learning / Tensor Interop
- **SafeTensors**: Read/write of the HuggingFace SafeTensors binary tensor format
- **ONNX**: Minimal hand-rolled protobuf parser/writer for ONNX `ModelProto` (nodes, opsets, inputs/outputs)
- **TFRecord**: Reader for TensorFlow's length-prefixed, CRC32C-checked record format
- **Mini-batch sampling**: Shuffled/stratified batch sampling with train/validation/test splitting

### Database Connectivity
- **SQLite**: Real, pure-Rust implementation via `oxisql-sqlite-compat` (Limbo engine) behind the `sqlite` feature — query, execute, bulk array insert, schema introspection
- **PostgreSQL / MySQL**: Feature flags exist (`postgres`, `mysql`) but the connection types are explicit stubs that return "implementation pending" errors — no `sqlx` or other real driver is wired in yet
- **MongoDB / Redis**: Feature flags exist for API-surface compatibility only; there is no backing crate or implementation

### Cloud and Distributed I/O
- **Object-store abstraction**: A unified `ObjectStore` trait with fully working `LocalObjectStore` and `MemoryObjectStore` backends
- **S3 / GCS / Azure backends**: Configuration types, presigned-URL builders, and simulated multipart/resumable-upload state machines are real and tested; however `put`/`get`/`delete`/`list`/`head` return a "real HTTP implementation not yet complete" error even when the `aws-sdk-s3`, `google-cloud-storage`, or `azure-storage-blobs` feature is enabled — no live network call is made yet
- **Azure SAS tokens**: Token structure and validation logic are real, but the signature is produced by a deterministic placeholder (`mock_sign`), not HMAC-SHA256 — do not use for production Azure authentication
- **Distributed I/O**: Partitioned parallel read/write for cluster workloads

### Performance and Low-Level I/O
- **Memory-mapped files**: `mmap`-based access for arrays larger than RAM
- **Zero-copy I/O**: Allocation-minimizing read/write paths for large datasets
- **SIMD-accelerated I/O**: Vectorized encode/decode paths for supported hardware
- **I/O thread pool**: Configurable, work-stealing thread pool tuned for I/O-bound vs. CPU-bound tasks
- **Out-of-core processing**: Disk-backed algorithms for terabyte-scale datasets that don't fit in memory
- **Experimental adaptive I/O tuning**: heuristic runtime tuning modules named "neural-adaptive" and "quantum-inspired" (`neural_adaptive_io`, `quantum_inspired_io`) implement real weight-update/Q-learning/parameter-search logic as optimization metaphors — they do not involve actual neural-network training frameworks or quantum hardware/simulators

### Validation and Integrity
- **Checksums**: CRC32, SHA-256, BLAKE3 integrity verification
- **Schema-based validation**: JSON Schema-compatible validation engine
- **Format validators**: Format-specific structural validators

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2-io = "0.6.1"
```

To enable optional feature groups:

```toml
[dependencies]
scirs2-io = { version = "0.6.1", features = ["async", "parquet"] }
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
use std::fs::File;
use std::io::BufReader;

let file = File::open("events.ndjson")?;
let mut reader = NdjsonReader::new(BufReader::new(file));
while let Some(record) = reader.next_record()? {
    // process each JSON object without loading the whole file
    let _ = record;
}
```

### CBOR round-trip

```rust
use scirs2_io::formats::cbor::{decode_cbor, encode_cbor, CborValue};

let value = CborValue::Map(vec![
    (CborValue::Text("sensor_id".into()), CborValue::Unsigned(42)),
    (CborValue::Text("value".into()), CborValue::Float(3.14)),
]);
let bytes = encode_cbor(&value);
let (decoded, _bytes_consumed) = decode_cbor(&bytes)?;
println!("{decoded:?}");
```

### Streaming CSV in fixed-size chunks

```rust
use scirs2_io::streaming_csv::CsvStreamReader;

let mut reader = CsvStreamReader::new("large.csv", b',', true)?;
loop {
    let chunk = reader.read_chunk(65_536)?;
    if chunk.is_empty() {
        break;
    }
    println!("chunk rows: {}", chunk.len());
}
```

### Parallel Zstd compression

```rust
use scirs2_io::compression::{compress_data_parallel, CompressionAlgorithm, ParallelCompressionConfig};

let data = std::fs::read("large_array.bin")?;
let cfg = ParallelCompressionConfig {
    num_threads: 8,
    chunk_size: 1 << 20,
    ..Default::default()
};
let (compressed, stats) = compress_data_parallel(&data, CompressionAlgorithm::Zstd, Some(6), cfg)?;
println!(
    "compressed to {}% of original using {} threads",
    100 * compressed.len() / data.len(),
    stats.threads_used
);
```

## API Overview

| Module | Purpose |
|--------|---------|
| `matlab` | `.mat` file read/write |
| `wavfile` | WAV audio read/write |
| `netcdf` | NetCDF3/4 (hand-rolled, does not use the `netcdf3` crate) |
| `hdf5_lite` | Pure-Rust HDF5 reader (read-only) |
| `hdf5` | System-library HDF5 binding (optional feature) |
| `npy` | NumPy `.npy`/`.npz` read/write |
| `fortran` | Fortran unformatted file read/write |
| `idl` | IDL `.sav` save file read/write |
| `csv` | CSV with type inference |
| `image` | PNG, JPEG, BMP, TIFF |
| `matrix_market` / `harwell_boeing` | Matrix Market / Harwell-Boeing |
| `sparse` | COO/CSR/CSC sparse matrix formats |
| `formats::astronomical` | FITS, VOTable |
| `formats::bioinformatics` | FASTA, FASTQ, SAM/BAM, VCF |
| `formats::geospatial` | GeoTIFF, Shapefile, GeoJSON, KML/KMZ |
| `formats::cbor` | CBOR encode/decode (`CborValue` model) |
| `formats::bson` | BSON encode/decode |
| `formats::avro` | Avro schema-based serialization |
| `formats::protobuf` / `protobuf_lite` | Protobuf message encoding / varint primitives |
| `msgpack` / `formats::msgpack` | MessagePack encode/decode |
| `formats::orc` / `formats::orc_lite` | ORC-inspired columnar formats |
| `formats::feather` | Feather (Arrow IPC) |
| `parquet` | Real Apache Arrow/Parquet reader/writer (feature-gated) |
| `parquet_lite` | Simplified Parquet-inspired reader (always available) |
| `zarr` | Zarr v2/v3 chunked arrays |
| `tiledb` | TileDB-inspired multidimensional arrays |
| `lance` | Lance-inspired ML columnar format |
| `delta` | Delta Lake transaction log reader/writer |
| `iceberg` | Simplified Apache Iceberg table abstraction |
| `streaming_csv` | Lazy streaming CSV (`CsvStreamReader`) |
| `streaming_json` | Streaming JSON parser |
| `ndjson_streaming` | NDJSON line-by-line streaming |
| `arrow_ipc` / `arrow_streaming` | Arrow IPC framed streaming |
| `streaming::windows` / `watermark` / `checkpoint` | Windowed aggregation, watermarks, checkpointing |
| `exactly_once` | Exactly-once delivery for streaming sinks |
| `binary_format` / `binary_formats` | Binary encoding utilities |
| `compression` / `compression_utils` | LZ4, Zstd, Brotli, Snappy, GZIP, BZIP2, parallel compression |
| `adaptive_compression` | Entropy-based automatic codec selection |
| `gpu` | GPU backend detection and GPU-assisted compression |
| `universal_reader` | Auto-detect format and open |
| `format_detect` | Magic bytes / extension detection |
| `query` | SQL-like predicate and projection |
| `joins` | Hash / sort-merge / nested-loop joins |
| `vectorized_eval` | Vectorized filter/project evaluation |
| `table_provider` / `datafusion_provider` | DataFusion-compatible table provider interface |
| `analytics` | Bloom filter, HyperLogLog++, t-digest, count-min sketch |
| `columnar` | Dictionary, RLE, delta/zigzag, FSST encoding |
| `catalog` | Data catalog |
| `lineage` | Provenance and lineage tracking |
| `schema` | Schema definitions, validation, and inference |
| `schema_registry` | Versioned Protobuf schema registry |
| `versioning` | Dataset versioning |
| `metadata` | Cross-format metadata management |
| `toml_ext` | Typed TOML configuration utilities |
| `etl` | ETL pipeline framework |
| `cloud` / `network` | Cloud storage connectors (S3/GCS/Azure — see Features) |
| `s3_multipart` | S3 multipart upload state machine |
| `distributed` | Distributed / partitioned I/O |
| `database` | SQLite (real), PostgreSQL/MySQL (stub) |
| `realtime` | Real WebSocket/SSE/gRPC/MQTT network clients |
| `protocols::kafka` / `protocols::arrow_flight` | In-memory protocol simulations |
| `mqtt_broker` | In-process MQTT-style pub/sub broker |
| `tensors` | SafeTensors, ONNX, TFRecord |
| `minibatch` | Mini-batch sampler |
| `pipeline` | Typed streaming pipeline (sources, transforms, sinks) |
| `validation` | Checksums, schema validation |
| `serialize` | Array and sparse-matrix serialization |
| `streaming` | Low-level streaming utilities |
| `mmap` / `zero_copy` / `simd_io` / `thread_pool` / `out_of_core` | Low-level performance I/O |

## Feature Flags

| Feature | Description |
|---------|-------------|
| `default` | `csv`, `compression`, `validation`, `image_io` |
| `csv` | CSV read/write support |
| `compression` | LZ4, Zstd, Brotli, Snappy pure-Rust codecs |
| `validation` | Checksums and schema validation |
| `image_io` | PNG/JPEG/BMP/TIFF via the `image` crate, plus EXIF metadata |
| `exif` | Alias enabling `image_io` for EXIF metadata access |
| `hdf5` | Enables the system-library HDF5 binding alongside the always-available `hdf5_lite` reader |
| `hdf5-legacy` | Keeps the C-linked `hdf5` crate available for write paths pending pure-Rust `oxih5` write support |
| `parquet` | Real Apache Arrow/Parquet reader/writer (`dep:parquet`, `dep:arrow`) |
| `async` | Async I/O via `tokio` |
| `reqwest` | HTTP/HTTPS network I/O |
| `gpu` | GPU backend detection and GPU-assisted compression |
| `sqlite` / `sqlite-stable` | Pure-Rust SQLite via `oxisql-sqlite-compat` (real) |
| `postgres` / `mysql` | Stub connection types only — no real driver wired in |
| `mongodb` / `redis` | Placeholder feature names — no backing crate or implementation |
| `database` | `sqlite` + `postgres` + `mysql` |
| `database-full` | Alias for `database` |
| `aws-sdk-s3` / `google-cloud-storage` / `azure-storage-blobs` | Gate cloud-vendor code paths, but live HTTP calls are not yet implemented (see Features) |
| `websocket` | Real WebSocket client via `tokio-tungstenite` (implies `async`) |
| `mqtt` | Real MQTT client via `rumqttc` (implies `async`) |
| `sse` | Real Server-Sent Events client via `eventsource-client` (implies `async`, `reqwest`) |
| `grpc` | Real gRPC client via `tonic` (implies `async`) |
| `realtime` | `websocket` + `mqtt` + `sse` + `grpc` |
| `all` | All of the above |

## Testing

`grep -rn "todo!()\|unimplemented!()" src/` returns 0 matches. Freshly measured 2026-07-15:

```bash
cargo nextest run -p scirs2-io               # default features: 1294 tests run, 1294 passed, 0 skipped
cargo nextest run -p scirs2-io --all-features # all-features:     1405 tests run, 1405 passed, 0 skipped
```

Both runs: 0 failed. See [TODO.md](./TODO.md) for the feature-by-feature maturity caveats (several
feature-gated areas — `postgres`/`mysql`/`mongodb`/`redis`, live cloud HTTP, Azure SAS signing —
are explicit, off-by-default stubs, not silent gaps in the tested surface above).

## Documentation

Full API documentation is available at [docs.rs/scirs2-io](https://docs.rs/scirs2-io).

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
