# scirs2-io TODO

## Status: v0.6.2 (2026-07-22)

**0.6.2:** MAT v7.3 (HDF5-based) support — `EnhancedMatFile`, `V73MatFile`, `PartialIoSupport` — is now available in default builds; all 45 feature gates across `matlab::enhanced`/`matlab::v73_enhanced` that previously required the `hdf5` Cargo feature (and a system `libhdf5`) are gone now that the backend is pure-Rust `oxih5`. The in-tree `hdf5_lite` module (2697 lines) was deleted outright; `oxih5` now serves every HDF5 path in the crate, widening format coverage (superblock v2/v3, fractal heaps, extensible arrays, virtual datasets, szip) beyond what `hdf5_lite` supported. Fixed three real bugs: `create_dataset_with_compression` was silently dropping every value it was handed (critical silent-data-loss bug), writing a `SparseLogical` matrix to MAT v7.3 stored all zeros, and the HDF5 v2-dataspace header was mis-parsed as three bytes instead of four. Added a new `tests/hdf5_conformance.rs` integration suite. See `CHANGELOG.md` `[0.6.2]` for full detail.

Zero `todo!()`/`unimplemented!()` stubs in `src/`. Freshly measured test counts (2026-07-15, predates the 0.6.2 changes above): `cargo nextest run -p scirs2-io` → 1294 tests run, 1294 passed, 0 skipped, 0 failed; `--all-features` → 1405 tests run, 1405 passed, 0 skipped, 0 failed. The OpenCL GPU-compression header now reports the adapter's real compute-unit count instead of a hardcoded placeholder (`gpu/compression.rs:305-310`, covered by `test_compress_opencl_header_uses_real_compute_units`). `src/hdf5/` and `src/advanced_coordinator/` were split into per-file module directories (all files under 2000 lines) via SplitRS, re-exports preserved. `zarr` v2/v3 confirmed publicly exported and non-stub. This pass re-verified every "v0.4.0 Roadmap" item below against `src/`: all are real, but several are scoped more narrowly than their one-line description implies (in-memory-only Kafka/Arrow Flight/MQTT-broker protocol simulations with no live network I/O; "-inspired" TileDB/Lance formats and a "simplified" Iceberg that are not byte-compatible with the reference ecosystem tools; cloud S3/GCS/Azure backends that build real config/state-machine/URL-signing logic but never make a live HTTP call, even with their feature flags enabled). Given the breadth of explicit off-by-default stub features (`postgres`/`mysql`/`mongodb`/`redis`, live cloud HTTP, Azure SAS signing), the status badge is **Partial** rather than Stable, even though the always-on default surface is solid and 100% green. See `README.md` for the precise, per-feature caveats.

## v0.3.3 Completed

### Classic Scientific Formats
- [x] MATLAB `.mat` v4/v5 read/write with all data types, structures, cell arrays
- [x] WAV audio read/write
- [x] ARFF (Attribute-Relation File Format) read/write
- [x] NetCDF3 and NetCDF4/HDF5 with unlimited dimensions and chunking
- [x] HDF5-lite pure-Rust hierarchical data reader
- [x] Matrix Market and Harwell-Boeing sparse matrix formats

### Modern Columnar and Binary Formats
- [x] Parquet-lite: pure-Rust Parquet reader
- [x] Feather (Arrow IPC): memory-mapped columnar format
- [x] ORC format reader
- [x] Binary format encoding utilities

### Serialization Formats
- [x] CBOR (RFC 7049) serialization and deserialization
- [x] BSON (Binary JSON) encode/decode
- [x] Avro schema-based serialization with schema evolution
- [x] Protobuf-lite: pure-Rust protobuf encoding/decoding
- [x] MessagePack encode/decode
- [x] NDJSON (Newline-Delimited JSON) streaming reader

### Streaming and Lazy Evaluation
- [x] Streaming CSV with lazy chunk evaluation
- [x] Streaming JSON incremental parser
- [x] NDJSON line-by-line streaming
- [x] Arrow IPC framed streaming
- [x] Backpressure-aware pipeline (sources, transforms, sinks)
- [x] Typed transform pipeline

### Compression
- [x] LZ4 high-speed compression
- [x] Zstd compression with configurable levels
- [x] Brotli general-purpose compression
- [x] Snappy block compression
- [x] GZIP / BZIP2 deflate-based compression
- [x] Parallel chunk compression (up to 2.5x throughput)

### Data Catalog, Lineage, Governance
- [x] Data catalog: register, tag, discover datasets
- [x] Lineage tracking: record transformations and provenance
- [x] Schema registry: store, evolve, and validate schemas
- [x] Dataset versioning with diff and rollback

### ETL and Query
- [x] ETL pipeline framework: source -> transform -> sink with parallel stages
- [x] SQL-like query interface: predicate pushdown and projection
- [x] Universal reader: auto-detect format from magic bytes/extension
- [x] Format detection for dozens of formats

### Cloud and Distributed
- [x] Cloud storage connector framework (AWS S3, GCS, Azure Blob)
- [x] Distributed / partitioned parallel read/write

### Validation and Integrity
- [x] CRC32, SHA-256, BLAKE3 checksum verification
- [x] JSON Schema-compatible schema validation engine
- [x] Format-specific structural validators

## v0.4.0 Roadmap

### New Formats
- [x] Zarr v2/v3 format: chunked, compressed, N-dimensional arrays; compatible with Zarr-Python — Implemented in v0.4.0 (`zarr/` module); re-verified 2026-07-15, `pub mod zarr;` confirmed non-stub
- [x] TileDB integration: dense and sparse multi-dimensional arrays for analytics — Implemented in v0.4.0 (`tiledb/` module, own doc comment says "TileDB-inspired"); pure-Rust array storage inspired by TileDB's design, not a client for real TileDB arrays
- [x] Lance format: modern columnar format for ML datasets — Implemented in v0.4.0 (`lance/` module, own doc comment says "Lance-inspired ... pure Rust, in-process"); not binary-compatible with the reference Lance format
- [x] Delta Lake log-based table format reader — Implemented in v0.4.0 (`delta/` module: `log.rs`/`table.rs`/`types.rs`); real JSON `_delta_log/` transaction log with commit/checkpoint/replay/time-travel
- [x] Iceberg table format support — implemented in v0.4.2 (`iceberg.rs`, own doc comment says "simplified pure-Rust implementation"); in-memory table abstraction with snapshot versioning, not a full Iceberg catalog/REST client

### Transport Protocols
- [x] Apache Arrow Flight protocol: high-throughput gRPC-based data transfer — Implemented in v0.4.0 (`protocols/arrow_flight.rs`, own doc comment says "pure-Rust in-memory simulation"); does not open a real gRPC/network connection
- [x] Apache Kafka consumer/producer for streaming scientific data — Implemented in v0.4.0 (`protocols/kafka.rs`, own doc comment says "pure-Rust in-memory broker simulation"); does not connect to a real Kafka broker over the network
- [x] MQTT topic-based streaming for IoT/sensor data ingestion — Implemented in v0.4.0 (`mqtt_broker/` module, own doc comment says "No network, no external crates — everything runs in-process"); real network MQTT client connectivity is separately provided by the `mqtt` feature (`rumqttc`, wired up in `realtime.rs`)

### Compression and Encoding
- [x] Columnar-aware compression: dictionary encoding, RLE, delta encoding per column — Implemented in v0.4.0 (`columnar/dictionary.rs`, `columnar/rle.rs`, `columnar/delta.rs`)
- [x] Bloom filter indexes for Parquet-like predicate pushdown — Implemented in v0.4.0 (`analytics/bloom_index.rs`)
- [x] FSST (Fast Static Symbol Table) string compression — Implemented in v0.4.0 (`columnar/fsst.rs`)
- [x] Adaptive compression: auto-select algorithm based on data entropy — Implemented in v0.4.2 (`adaptive_compression/mod.rs`); OxiARC-backed LZ4/Zstd/Brotli with Shannon entropy selection; `auto_compress`/`auto_decompress` with 1-byte tag

### Cloud and Distributed
- [x] Native AWS S3 multipart upload with parallel chunk upload — Implemented in v0.4.2 (`s3_multipart.rs`); feature-gated stub with full state machine simulation; real HTTP requires `aws-sdk-s3` feature
- [x] Native GCS resumable uploads — Implemented in v0.4.2 (`cloud/gcs.rs`); simulation-mode state machine with offset validation, abort/finalize, assembled_data; 8 tests
- [x] Azure Blob SAS-token authentication support — Implemented in v0.4.2 (`cloud/azure_sas.rs`); SasPermissions, SasResource, generate_sas_token, build_sas_url, parse_sas_token, is_sas_valid; 8 tests
- [x] Object-store abstraction layer unified across providers — Implemented in v0.4.2 (`cloud/mod.rs`); `ObjectStore` trait + `LocalObjectStore` + `MemoryObjectStore` + S3/GCS/Azure stubs; `parse_store_url` + `from_url` factory; GCS and Azure stubs available, feature-gated

### Query and Analytics
- [x] DataFusion-compatible table provider interface — implemented in v0.4.2 (`datafusion_provider.rs`)
- [x] Vectorized expression evaluation for filter and project — implemented in v0.4.2 (`datafusion_provider.rs`)
- [x] Approximate aggregations: HyperLogLog, t-digest, count-min sketch — Implemented in v0.4.0 (`analytics/hyperloglog.rs`, `analytics/tdigest.rs`, `analytics/count_min.rs`)
- [x] Join algorithms for cross-format dataset merge — implemented in v0.4.2 (`joins.rs`)

### Streaming Enhancements
- [x] Exactly-once delivery semantics for streaming pipeline sinks — Implemented in v0.4.2 (`exactly_once.rs`); WriteAheadLog (disk + in-memory) + ExactlyOnceSink with idempotency-key deduplication; 10 tests
- [x] Windowed aggregation (tumbling, sliding, session windows) — Implemented in v0.4.0 (`streaming/windows.rs`)
- [x] Watermark-based late-data handling — Implemented in v0.4.0 (`streaming/watermark.rs`)
- [x] Checkpointing and restart for long-running streaming jobs — Implemented in v0.4.0 (`streaming/checkpoint.rs`)

### Machine Learning Integration
- [x] Tensor serialization (safetensors-compatible read/write) — implemented in v0.4.2 (`tensors/safetensors.rs`)
- [x] ONNX model proto read/write — implemented in v0.4.2 (`tensors/onnx_proto.rs`)
- [x] TFRecord reader for TensorFlow data pipelines — implemented in v0.4.2 (`tensors/tfrecord.rs`)
- [x] Efficient mini-batch sampler with shuffle and stratified splitting — implemented in v0.4.2 (`minibatch.rs`)

## Known Issues

- **[Corrected 2026-07-22]** ~~Large HDF5 files with deeply nested groups may be slow on the pure-Rust hdf5-lite reader; the system-library `hdf5` feature should be preferred for those workloads.~~ Obsolete as of 0.6.2: the `hdf5_lite` module was deleted outright (2697 lines), superseded by the pure-Rust `oxih5` backend for every HDF5 path in the crate; the `hdf5` Cargo feature is now a no-op alias (HDF5 support is unconditional), not a system-library switch — there is no `libhdf5` alternative to fall back to any more. Performance of `oxih5` on deeply-nested-group files has not been independently benchmarked.
- **[Corrected 2026-07-15]** ~~The ORC reader does not yet support all column encodings (RLE v2, dictionary, DIRECT_V2); unsupported columns fall back to raw bytes.~~ Verified stale: both `formats::orc` and `formats::orc_lite` implement RLE v2 integer encoding (direct/delta/variable-length modes), dictionary string encoding, and bit-packed boolean RLE (`orc.rs::decode_i64`/`decode_dict_strings`/`decode_bool_rle`, `orc_lite.rs::IntRleV2`). Remaining caveat: neither format reads/writes third-party `.orc` files produced by Hive/Spark/etc. — both use their own magic bytes and framing (`ORCEXT\0\0` / `ORCLITE\0`) rather than the real Apache ORC Protobuf postscript/footer, so they are ORC-inspired pure-Rust formats, not Apache-ORC-file-compatible readers.
- **[Corrected 2026-07-15]** ~~Arrow IPC streaming does not yet validate all IPC message types; unknown message types are silently skipped.~~ Verified stale: `arrow_ipc::read_message`/`read_batches` and `arrow_streaming::read_message` now return `IoError::FormatError` ("Unexpected/unexpected message type ...") on any unrecognized message tag instead of skipping it silently.
- **[Refined 2026-07-15]** Cloud connector framework (`cloud`, `network::cloud`, `s3_multipart`) provides the interface, config/URL-signing types, and simulated multipart/resumable-upload state machines only. Verified: even with `aws-sdk-s3` / `google-cloud-storage` / `azure-storage-blobs` enabled, `ObjectStore::put/get/delete/list/head` on the S3/GCS/Azure backends unconditionally return a "real HTTP implementation not yet complete" error (`cloud/mod.rs`) — no live network call is made under any current feature combination, so simply "activating `reqwest`" is not sufficient to get real cloud I/O. Separately, Azure SAS token generation (`cloud/azure_sas.rs`) signs with a deterministic placeholder (`mock_sign`, explicitly documented as "NOT cryptographically secure"), not HMAC-SHA256 — do not use for production Azure authentication.
- BSON serialization of f32 arrays upcasts to f64 to conform with the BSON type system (confirmed: `BsonValue` has no dedicated `Float32` variant, only `Double(f64)`).
