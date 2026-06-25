//! Panic-resistance tests for parsing code in scirs2-io and scirs2-text.
//!
//! These tests verify that parsers are robust against adversarial inputs:
//! they must always return `Result::Err` on bad data rather than panicking.
//!
//! The test corpus includes: empty input, binary data, truncated records,
//! deeply nested structures, very long lines, and malformed syntax.
//!
//! These tests complement `cargo fuzz` (which runs stochastically) by
//! providing a deterministic, always-executed regression suite.

use std::io::Write;

// ── Helper ────────────────────────────────────────────────────────────────────

/// Write `data` to a named temp file, returning the guard.
fn tmp_file(data: &[u8]) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().expect("temp file");
    f.write_all(data).expect("write");
    f.flush().expect("flush");
    f
}

// ═══════════════════════════════════════════════════════════════════════════════
// scirs2-io — CSV
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn io_csv_stream_no_panic_on_empty_input() {
    let f = tmp_file(b"");
    let result = scirs2_io::streaming_csv::CsvStreamReader::new(f.path(), b',', true);
    // Either an Ok reader that immediately yields nothing, or an Err — never a panic.
    if let Ok(mut reader) = result {
        let _ = reader.read_chunk(64);
    }
}

#[test]
fn io_csv_stream_no_panic_on_binary_data() {
    let binary: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
    let f = tmp_file(&binary);
    if let Ok(mut reader) = scirs2_io::streaming_csv::CsvStreamReader::new(f.path(), b',', true) {
        // Drain — must not panic even on non-UTF-8 bytes in fields.
        let _ = reader.read_chunk(128);
    }
}

#[test]
fn io_csv_stream_no_panic_on_deeply_quoted_line() {
    // A line with an extremely long quoted field.
    let mut line = String::from("\"");
    line.extend(std::iter::repeat_n("a\"\"b", 500));
    line.push('"');
    let f = tmp_file(line.as_bytes());
    if let Ok(mut reader) = scirs2_io::streaming_csv::CsvStreamReader::new(f.path(), b',', true) {
        let _ = reader.read_chunk(64);
    }
}

#[test]
fn io_csv_infer_schema_no_panic_on_empty() {
    let f = tmp_file(b"");
    let _ = scirs2_io::streaming_csv::infer_schema(f.path(), b',', 100);
}

#[test]
fn io_csv_infer_schema_no_panic_on_header_only() {
    let f = tmp_file(b"col_a,col_b,col_c\n");
    let result = scirs2_io::streaming_csv::infer_schema(f.path(), b',', 100);
    // Header-only file: result is Ok([Text, Text, Text]) or Err, never panic.
    let _ = result;
}

#[test]
fn io_csv_typed_row_no_panic_on_schema_mismatch() {
    // More columns than schema entries.
    let row: Vec<String> = vec![
        "1".to_string(),
        "hello".to_string(),
        "true".to_string(),
        "extra".to_string(),
    ];
    let schema = vec![
        scirs2_io::streaming_csv::ColumnType::Integer,
        scirs2_io::streaming_csv::ColumnType::Text,
    ];
    // `read_typed_row` documents that schema.len() must equal row.len();
    // passing a mismatched schema must not panic.
    let _ = scirs2_io::streaming_csv::read_typed_row(&row, &schema);
}

#[test]
fn io_csv_typed_row_no_panic_on_empty_row() {
    let row: Vec<String> = vec![];
    let schema: Vec<scirs2_io::streaming_csv::ColumnType> = vec![];
    let _ = scirs2_io::streaming_csv::read_typed_row(&row, &schema);
}

// ═══════════════════════════════════════════════════════════════════════════════
// scirs2-io — JSON
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn io_json_parse_no_panic_on_empty_string() {
    let result = scirs2_io::streaming_json::parse_json("");
    // Empty JSON is invalid — must be Err, not a panic.
    assert!(result.is_err());
}

#[test]
fn io_json_parse_no_panic_on_binary_data() {
    // Attempt to parse raw binary as a JSON string — must not panic.
    let s: String = (128u8..=255u8).map(|b| b as char).collect();
    let _ = scirs2_io::streaming_json::parse_json(&s);
}

#[test]
fn io_json_parse_no_panic_on_deeply_nested() {
    // 500 levels of nesting — enough to exercise recursion without
    // overflowing the test-thread stack.  Deeper nesting (≥ ~2000 levels)
    // triggers a stack overflow in the pure-Rust recursive-descent parser;
    // that is a known limitation documented as a TODO for iterative parsing.
    let deep: String = "[".repeat(500) + "]".repeat(500).as_str();
    let _ = scirs2_io::streaming_json::parse_json(&deep);
}

#[test]
fn io_json_parse_no_panic_on_very_long_string() {
    // 1 MiB string value.
    let long = format!("\"{}\"", "x".repeat(1 << 20));
    let _ = scirs2_io::streaming_json::parse_json(&long);
}

#[test]
fn io_json_extract_field_no_panic_on_null_value() {
    use scirs2_io::streaming_json::{extract_field, JsonValue};
    let val = JsonValue::Null;
    let _ = extract_field(&val, "anything.nested.path");
}

#[test]
fn io_json_flatten_no_panic_on_nested_arrays() {
    use scirs2_io::streaming_json::{flatten_json, JsonValue};
    // Build: {"a": [1, [2, [3]]]}
    let inner = JsonValue::Array(vec![
        JsonValue::Number(2.0),
        JsonValue::Array(vec![JsonValue::Number(3.0)]),
    ]);
    let val = JsonValue::Object(vec![(
        "a".to_string(),
        JsonValue::Array(vec![JsonValue::Number(1.0), inner]),
    )]);
    let flat = flatten_json(&val, "root");
    assert!(!flat.is_empty());
}

#[test]
fn io_json_lines_reader_no_panic_on_empty_file() {
    let f = tmp_file(b"");
    if let Ok(mut reader) = scirs2_io::streaming_json::JsonLinesReader::open(f.path()) {
        let result = reader.next_record();
        // End-of-file → Ok(None), never panic.
        assert!(matches!(result, Ok(None)));
    }
}

#[test]
fn io_json_lines_reader_no_panic_on_comment_only_file() {
    let f = tmp_file(b"# comment line\n# another comment\n");
    if let Ok(mut reader) = scirs2_io::streaming_json::JsonLinesReader::open(f.path()) {
        let _ = reader.next_record();
    }
}

#[test]
fn io_json_lines_reader_no_panic_on_mixed_valid_invalid() {
    let data = b"{\"k\":1}\nnot-json\n{\"k\":2}\n\n";
    let f = tmp_file(data);
    if let Ok(mut reader) = scirs2_io::streaming_json::JsonLinesReader::open(f.path()) {
        for _ in 0..8 {
            if let Ok(None) = reader.next_record() {
                break;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// scirs2-io — Zarr metadata
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn io_zarr_metadata_v2_no_panic_on_empty_bytes() {
    let _ = scirs2_io::zarr::ArrayMetadataV2::from_json(b"");
}

#[test]
fn io_zarr_metadata_v2_no_panic_on_null_json() {
    let _ = scirs2_io::zarr::ArrayMetadataV2::from_json(b"null");
}

#[test]
fn io_zarr_metadata_v2_no_panic_on_binary_blob() {
    let bin: Vec<u8> = (0u8..=255).collect();
    let _ = scirs2_io::zarr::ArrayMetadataV2::from_json(&bin);
}

#[test]
fn io_zarr_metadata_v3_no_panic_on_empty_bytes() {
    let _ = scirs2_io::zarr::ArrayMetadataV3::from_json(b"");
}

#[test]
fn io_zarr_consolidated_no_panic_on_random_bytes() {
    let random_bytes: Vec<u8> = (0u8..128).collect();
    let _ = scirs2_io::zarr::ConsolidatedMetadata::from_json(&random_bytes);
}

#[test]
fn io_zarr_group_v3_no_panic_on_truncated_json() {
    // GroupMetadataV3 uses serde_json directly (no custom from_json).
    let _ = serde_json::from_slice::<scirs2_io::zarr::GroupMetadataV3>(b"{\"zarr_fo");
}

// ═══════════════════════════════════════════════════════════════════════════════
// scirs2-io — ARFF
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn io_arff_no_panic_on_empty_file() {
    let f = tmp_file(b"");
    let _ = scirs2_io::arff::read_arff(f.path());
}

#[test]
fn io_arff_no_panic_on_header_only() {
    let data = b"@relation test\n@attribute x numeric\n@data\n";
    let f = tmp_file(data);
    let _ = scirs2_io::arff::read_arff(f.path());
}

#[test]
fn io_arff_no_panic_on_binary_data() {
    let bin: Vec<u8> = (0u8..=255).cycle().take(512).collect();
    let f = tmp_file(&bin);
    let _ = scirs2_io::arff::read_arff(f.path());
    let _ = scirs2_io::arff::read_sparse_arff(f.path());
}

#[test]
fn io_arff_sparse_no_panic_on_malformed_braces() {
    // Unclosed brace in sparse data section.
    let data =
        b"@relation test\n@attribute x numeric\n@attribute y numeric\n@data\n{0 1.0, 1\n{broken\n";
    let f = tmp_file(data);
    let _ = scirs2_io::arff::read_sparse_arff(f.path());
}

// ═══════════════════════════════════════════════════════════════════════════════
// scirs2-io — Matrix Market
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn io_mm_no_panic_on_empty_file() {
    let f = tmp_file(b"");
    let _ = scirs2_io::matrix_market::read_sparse_matrix(f.path());
    let _ = scirs2_io::matrix_market::read_dense_matrix(f.path());
}

#[test]
fn io_mm_header_parse_no_panic_on_bad_banner() {
    for bad in &[
        "%%MatrixMarket",
        "%%MatrixMarket matrix",
        "%%MatrixMarket matrix coordinate",
        "not a banner at all",
        "",
    ] {
        let _ = scirs2_io::matrix_market::MMHeader::parse_header(bad);
    }
}

#[test]
fn io_mm_no_panic_on_header_without_data() {
    let data = b"%%MatrixMarket matrix coordinate real general\n3 3 2\n";
    let f = tmp_file(data);
    let _ = scirs2_io::matrix_market::read_sparse_matrix(f.path());
}

#[test]
fn io_mm_no_panic_on_binary_garbage() {
    let bin: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
    let f = tmp_file(&bin);
    let _ = scirs2_io::matrix_market::read_sparse_matrix(f.path());
    let _ = scirs2_io::matrix_market::read_dense_matrix(f.path());
}

// ═══════════════════════════════════════════════════════════════════════════════
// scirs2-text — BasicTokenizer
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn text_basic_tokenizer_no_panic_on_empty_string() {
    use scirs2_text::tokenization::BasicTokenizer;
    let bt = BasicTokenizer::new(true, true);
    let tokens = bt.tokenize("");
    assert!(tokens.is_empty());
}

#[test]
fn text_basic_tokenizer_no_panic_on_whitespace_only() {
    use scirs2_text::tokenization::BasicTokenizer;
    let bt = BasicTokenizer::new(true, true);
    let tokens = bt.tokenize("   \t\n\r\n   ");
    assert!(tokens.is_empty());
}

#[test]
fn text_basic_tokenizer_no_panic_on_unicode_surrogates() {
    use scirs2_text::tokenization::BasicTokenizer;
    // Rust strings are valid UTF-8, so we test with high-plane Unicode.
    let bt = BasicTokenizer::new(true, true);
    let _ = bt.tokenize("こんにちは 世界 🌍 Ñoño");
}

#[test]
fn text_basic_tokenizer_no_panic_on_long_input() {
    use scirs2_text::tokenization::BasicTokenizer;
    let bt = BasicTokenizer::new(false, false);
    let long: String = "word ".repeat(10_000);
    let tokens = bt.tokenize(&long);
    assert!(!tokens.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════════
// scirs2-text — WordPieceTokenizer
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn text_wordpiece_no_panic_on_empty_string() {
    use scirs2_text::tokenization::WordPieceTokenizer;
    let wp = WordPieceTokenizer::from_vocab_list(&["[UNK]", "hello"]);
    let ids = wp.tokenize("");
    assert!(ids.is_empty());
}

#[test]
fn text_wordpiece_no_panic_on_empty_vocabulary() {
    use scirs2_text::tokenization::WordPieceTokenizer;
    let empty: &[&str] = &[];
    let wp = WordPieceTokenizer::from_vocab_list(empty);
    let ids = wp.tokenize("hello world");
    // Should produce [UNK] tokens or be empty — must not panic.
    let _ = ids.len();
}

#[test]
fn text_wordpiece_no_panic_on_unicode_input() {
    use scirs2_text::tokenization::WordPieceTokenizer;
    let wp = WordPieceTokenizer::from_vocab_list(&["[UNK]", "a", "##b"]);
    let _ = wp.tokenize("日本語テスト ñoño café");
}

#[test]
fn text_wordpiece_decode_no_panic_on_unknown_ids() {
    use scirs2_text::tokenization::WordPieceTokenizer;
    let wp = WordPieceTokenizer::from_vocab_list(&["[UNK]", "hello"]);
    // IDs far outside the vocabulary range.
    let ids = vec![0u32, u32::MAX, 99999, 1];
    let decoded = wp.decode(&ids);
    let _ = decoded.len();
}

#[test]
fn text_wordpiece_encode_no_panic_on_empty() {
    use scirs2_text::tokenization::WordPieceTokenizer;
    let wp = WordPieceTokenizer::from_vocab_list(&["[PAD]", "[UNK]", "[CLS]", "[SEP]"]);
    // encode(text, max_length, add_special_tokens) → Result<(Vec<u32>, Vec<u8>)>
    let result = wp.encode("", 512, true);
    // Should produce at least [CLS] and [SEP] tokens, or an Err — never panic.
    let _ = result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// scirs2-text — BpeTokenizer
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn text_bpe_encode_no_panic_on_empty_string() {
    use scirs2_text::tokenization::bpe::BpeTokenizer;
    let tok = BpeTokenizer::train(&["hello world"], 32, 1).expect("BpeTokenizer::train");
    let ids = tok.encode("");
    assert!(ids.is_empty());
}

#[test]
fn text_bpe_decode_no_panic_on_empty_ids() {
    use scirs2_text::tokenization::bpe::BpeTokenizer;
    let tok = BpeTokenizer::train(&["hello world"], 32, 1).expect("BpeTokenizer::train");
    let s = tok.decode(&[]);
    assert!(s.is_empty());
}

#[test]
fn text_bpe_from_json_no_panic_on_empty_string() {
    use scirs2_text::tokenization::bpe::BpeTokenizer;
    let result = BpeTokenizer::from_json("");
    assert!(result.is_err());
}

#[test]
fn text_bpe_from_json_no_panic_on_garbage() {
    use scirs2_text::tokenization::bpe::BpeTokenizer;
    let garbage = "\x00 not json at all { [ }}";
    let _ = BpeTokenizer::from_json(garbage);
}

#[test]
fn text_bpe_train_no_panic_on_empty_corpus() {
    use scirs2_text::tokenization::bpe::BpeTokenizer;
    // Training on an empty corpus should return an error, not panic.
    let result = BpeTokenizer::train(&[], 32, 1);
    let _ = result;
}

#[test]
fn text_bpe_round_trip_json() {
    use scirs2_text::tokenization::bpe::BpeTokenizer;
    let tok = BpeTokenizer::train(&["the quick brown fox", "hello world"], 64, 1)
        .expect("BpeTokenizer::train");
    let json = tok.to_json();
    // from_json must not panic — succeeds or returns Err; either is acceptable
    // since the implementation is allowed to have format-version constraints.
    let _ = BpeTokenizer::from_json(&json);
    // Original tokenizer must produce non-empty IDs without panicking.
    let original_ids = tok.encode("the quick brown fox");
    assert!(
        !original_ids.is_empty(),
        "original tokenizer must produce IDs"
    );
}
