//! Apache Iceberg table format support (simplified pure-Rust implementation).
//!
//! Provides an in-memory Apache Iceberg table abstraction with:
//! - Schema definition (fields, types, nullability)
//! - Snapshot-based versioning for time travel
//! - JSON serialization of table metadata
//! - Append-only writes with commit semantics

use std::collections::HashMap;
use std::path::Path;

use serde_json::{json, Value};
use uuid::Uuid;

// ──────────────────────────────────────────────────────────────────────────────
// Error type
// ──────────────────────────────────────────────────────────────────────────────

/// Errors that can arise when working with Iceberg tables.
#[derive(Debug, thiserror::Error)]
pub enum IcebergError {
    /// An underlying I/O error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Table metadata is malformed or missing required fields.
    #[error("Invalid metadata: {0}")]
    InvalidMetadata(String),
    /// The requested snapshot does not exist in the table history.
    #[error("Snapshot not found: {0}")]
    SnapshotNotFound(i64),
    /// The supplied data does not match the declared schema.
    #[error("Schema mismatch: {0}")]
    SchemaMismatch(String),
    /// JSON serialization / deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

// ──────────────────────────────────────────────────────────────────────────────
// Type system
// ──────────────────────────────────────────────────────────────────────────────

/// Iceberg column type.
#[derive(Debug, Clone, PartialEq)]
pub enum IcebergType {
    /// Boolean value.
    Boolean,
    /// 32-bit signed integer.
    Integer,
    /// 64-bit signed integer.
    Long,
    /// 32-bit IEEE 754 floating-point number.
    Float,
    /// 64-bit IEEE 754 floating-point number.
    Double,
    /// UTF-8 string.
    String,
    /// Opaque byte array.
    Bytes,
    /// Calendar date (days since epoch).
    Date,
    /// Microsecond-precision timestamp.
    Timestamp,
    /// Variable-length list whose element type is `Box<IcebergType>`.
    List(Box<IcebergType>),
    /// Nested struct of named fields.
    Struct(Vec<IcebergField>),
}

impl IcebergType {
    /// Return a JSON-compatible type string for metadata serialization.
    fn type_string(&self) -> Value {
        match self {
            IcebergType::Boolean => json!("boolean"),
            IcebergType::Integer => json!("int"),
            IcebergType::Long => json!("long"),
            IcebergType::Float => json!("float"),
            IcebergType::Double => json!("double"),
            IcebergType::String => json!("string"),
            IcebergType::Bytes => json!("binary"),
            IcebergType::Date => json!("date"),
            IcebergType::Timestamp => json!("timestamp"),
            IcebergType::List(elem) => json!({
                "type": "list",
                "element-type": elem.type_string(),
            }),
            IcebergType::Struct(fields) => {
                let serialized: Vec<Value> = fields.iter().map(|f| f.to_json()).collect();
                json!({ "type": "struct", "fields": serialized })
            }
        }
    }

    /// Parse an `IcebergType` from a JSON value produced by `type_string`.
    fn from_json(v: &Value) -> Result<Self, IcebergError> {
        match v {
            Value::String(s) => match s.as_str() {
                "boolean" => Ok(IcebergType::Boolean),
                "int" => Ok(IcebergType::Integer),
                "long" => Ok(IcebergType::Long),
                "float" => Ok(IcebergType::Float),
                "double" => Ok(IcebergType::Double),
                "string" => Ok(IcebergType::String),
                "binary" => Ok(IcebergType::Bytes),
                "date" => Ok(IcebergType::Date),
                "timestamp" => Ok(IcebergType::Timestamp),
                other => Err(IcebergError::InvalidMetadata(format!(
                    "Unknown scalar type: {other}"
                ))),
            },
            Value::Object(map) => {
                let type_tag = map
                    .get("type")
                    .and_then(|t| t.as_str())
                    .ok_or_else(|| IcebergError::InvalidMetadata("Missing 'type' key".into()))?;
                match type_tag {
                    "list" => {
                        let elem_json = map.get("element-type").ok_or_else(|| {
                            IcebergError::InvalidMetadata("List missing element-type".into())
                        })?;
                        Ok(IcebergType::List(Box::new(IcebergType::from_json(
                            elem_json,
                        )?)))
                    }
                    "struct" => {
                        let fields_json =
                            map.get("fields")
                                .and_then(|f| f.as_array())
                                .ok_or_else(|| {
                                    IcebergError::InvalidMetadata(
                                        "Struct missing fields array".into(),
                                    )
                                })?;
                        let fields = fields_json
                            .iter()
                            .map(IcebergField::from_json)
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok(IcebergType::Struct(fields))
                    }
                    other => Err(IcebergError::InvalidMetadata(format!(
                        "Unknown complex type: {other}"
                    ))),
                }
            }
            _ => Err(IcebergError::InvalidMetadata(format!(
                "Unexpected JSON for type: {v}"
            ))),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Schema
// ──────────────────────────────────────────────────────────────────────────────

/// A single named field in an Iceberg schema or nested struct.
#[derive(Debug, Clone, PartialEq)]
pub struct IcebergField {
    /// Stable, monotonically increasing column identifier.
    pub id: i32,
    /// Column name (case-sensitive).
    pub name: std::string::String,
    /// Declared column type.
    pub field_type: IcebergType,
    /// When `true` the column must not contain null values.
    pub required: bool,
    /// Optional human-readable description.
    pub doc: Option<std::string::String>,
}

impl IcebergField {
    fn to_json(&self) -> Value {
        let mut map = serde_json::Map::new();
        map.insert("id".into(), json!(self.id));
        map.insert("name".into(), json!(self.name));
        map.insert("type".into(), self.field_type.type_string());
        map.insert("required".into(), json!(self.required));
        if let Some(doc) = &self.doc {
            map.insert("doc".into(), json!(doc));
        }
        Value::Object(map)
    }

    fn from_json(v: &Value) -> Result<Self, IcebergError> {
        let id = v["id"]
            .as_i64()
            .ok_or_else(|| IcebergError::InvalidMetadata("Field missing 'id'".into()))?
            as i32;
        let name = v["name"]
            .as_str()
            .ok_or_else(|| IcebergError::InvalidMetadata("Field missing 'name'".into()))?
            .to_string();
        let field_type = IcebergType::from_json(&v["type"])?;
        let required = v["required"].as_bool().unwrap_or(false);
        let doc = v["doc"].as_str().map(|s| s.to_string());
        Ok(IcebergField {
            id,
            name,
            field_type,
            required,
            doc,
        })
    }
}

/// Iceberg table schema, comprising an ordered list of typed fields.
#[derive(Debug, Clone)]
pub struct IcebergSchema {
    /// Monotonically increasing schema identifier.
    pub schema_id: i32,
    /// Ordered list of column descriptors.
    pub fields: Vec<IcebergField>,
}

impl IcebergSchema {
    fn to_json(&self) -> Value {
        json!({
            "schema-id": self.schema_id,
            "type": "struct",
            "fields": self.fields.iter().map(|f| f.to_json()).collect::<Vec<_>>(),
        })
    }

    fn from_json(v: &Value) -> Result<Self, IcebergError> {
        let schema_id = v["schema-id"].as_i64().unwrap_or(0) as i32;
        let fields = v["fields"]
            .as_array()
            .ok_or_else(|| IcebergError::InvalidMetadata("Schema missing 'fields'".into()))?
            .iter()
            .map(IcebergField::from_json)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(IcebergSchema { schema_id, fields })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Snapshots
// ──────────────────────────────────────────────────────────────────────────────

/// An Iceberg snapshot captures the table state at a specific point in time.
///
/// In the full Iceberg spec a snapshot references a manifest list file which
/// in turn references manifest files describing individual data files. This
/// implementation keeps the row data in-memory instead, but preserves all
/// metadata fields for compatibility with the JSON format.
#[derive(Debug, Clone)]
pub struct IcebergSnapshot {
    /// Globally unique 64-bit snapshot identifier.
    pub snapshot_id: i64,
    /// Snapshot ID of the immediate predecessor, if any.
    pub parent_snapshot_id: Option<i64>,
    /// Monotonically increasing sequence number for ordering.
    pub sequence_number: i64,
    /// Unix time in milliseconds when this snapshot was committed.
    pub timestamp_ms: i64,
    /// Path to the manifest list file (in-memory: descriptive placeholder).
    pub manifest_list: std::string::String,
    /// Arbitrary key-value pairs summarising the commit (e.g. added-files count).
    pub summary: HashMap<std::string::String, std::string::String>,
}

impl IcebergSnapshot {
    fn to_json(&self) -> Value {
        let mut summary_map = serde_json::Map::new();
        for (k, v) in &self.summary {
            summary_map.insert(k.clone(), json!(v));
        }
        let mut obj = serde_json::Map::new();
        obj.insert("snapshot-id".into(), json!(self.snapshot_id));
        if let Some(parent) = self.parent_snapshot_id {
            obj.insert("parent-snapshot-id".into(), json!(parent));
        }
        obj.insert("sequence-number".into(), json!(self.sequence_number));
        obj.insert("timestamp-ms".into(), json!(self.timestamp_ms));
        obj.insert("manifest-list".into(), json!(self.manifest_list));
        obj.insert("summary".into(), Value::Object(summary_map));
        Value::Object(obj)
    }

    fn from_json(v: &Value) -> Result<Self, IcebergError> {
        let snapshot_id = v["snapshot-id"].as_i64().ok_or_else(|| {
            IcebergError::InvalidMetadata("Snapshot missing 'snapshot-id'".into())
        })?;
        let parent_snapshot_id = v["parent-snapshot-id"].as_i64();
        let sequence_number = v["sequence-number"].as_i64().unwrap_or(0);
        let timestamp_ms = v["timestamp-ms"].as_i64().unwrap_or(0);
        let manifest_list = v["manifest-list"].as_str().unwrap_or("").to_string();
        let mut summary = HashMap::new();
        if let Some(s) = v["summary"].as_object() {
            for (k, val) in s {
                if let Some(vs) = val.as_str() {
                    summary.insert(k.clone(), vs.to_string());
                }
            }
        }
        Ok(IcebergSnapshot {
            snapshot_id,
            parent_snapshot_id,
            sequence_number,
            timestamp_ms,
            manifest_list,
            summary,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Table metadata
// ──────────────────────────────────────────────────────────────────────────────

/// Iceberg table metadata, analogous to the `metadata.json` file on disk.
#[derive(Debug, Clone)]
pub struct IcebergTableMetadata {
    /// Iceberg format version (1 or 2).
    pub format_version: u32,
    /// Globally unique table identifier (UUID v4).
    pub table_uuid: std::string::String,
    /// Canonical location of the table data (e.g. `s3://bucket/prefix/`).
    pub location: std::string::String,
    /// Highest sequence number used so far.
    pub last_sequence_number: i64,
    /// Unix time in milliseconds of the last metadata update.
    pub last_updated_ms: i64,
    /// Highest column ID assigned so far.
    pub last_column_id: i32,
    /// Current table schema.
    pub schema: IcebergSchema,
    /// Ordered list of all snapshots.
    pub snapshots: Vec<IcebergSnapshot>,
    /// ID of the snapshot currently considered "current", if any.
    pub current_snapshot_id: Option<i64>,
}

impl IcebergTableMetadata {
    fn to_json(&self) -> Value {
        let snapshots: Vec<Value> = self.snapshots.iter().map(|s| s.to_json()).collect();
        let mut obj = serde_json::Map::new();
        obj.insert("format-version".into(), json!(self.format_version));
        obj.insert("table-uuid".into(), json!(self.table_uuid));
        obj.insert("location".into(), json!(self.location));
        obj.insert(
            "last-sequence-number".into(),
            json!(self.last_sequence_number),
        );
        obj.insert("last-updated-ms".into(), json!(self.last_updated_ms));
        obj.insert("last-column-id".into(), json!(self.last_column_id));
        obj.insert("schema".into(), self.schema.to_json());
        obj.insert("snapshots".into(), json!(snapshots));
        if let Some(id) = self.current_snapshot_id {
            obj.insert("current-snapshot-id".into(), json!(id));
        }
        Value::Object(obj)
    }

    fn from_json(v: &Value) -> Result<Self, IcebergError> {
        let format_version = v["format-version"].as_u64().unwrap_or(2) as u32;
        let table_uuid = v["table-uuid"].as_str().unwrap_or("").to_string();
        let location = v["location"].as_str().unwrap_or("").to_string();
        let last_sequence_number = v["last-sequence-number"].as_i64().unwrap_or(0);
        let last_updated_ms = v["last-updated-ms"].as_i64().unwrap_or(0);
        let last_column_id = v["last-column-id"].as_i64().unwrap_or(0) as i32;
        let schema = IcebergSchema::from_json(&v["schema"])?;
        let snapshots = v["snapshots"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(IcebergSnapshot::from_json)
                    .collect::<Result<Vec<_>, _>>()
            })
            .unwrap_or(Ok(Vec::new()))?;
        let current_snapshot_id = v["current-snapshot-id"].as_i64();
        Ok(IcebergTableMetadata {
            format_version,
            table_uuid,
            location,
            last_sequence_number,
            last_updated_ms,
            last_column_id,
            schema,
            snapshots,
            current_snapshot_id,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Value type
// ──────────────────────────────────────────────────────────────────────────────

/// A single typed Iceberg cell value.
#[derive(Debug, Clone, PartialEq)]
pub enum IcebergValue {
    /// SQL NULL.
    Null,
    /// Boolean.
    Boolean(bool),
    /// 32-bit signed integer.
    Integer(i32),
    /// 64-bit signed integer.
    Long(i64),
    /// 32-bit floating-point number.
    Float(f32),
    /// 64-bit floating-point number.
    Double(f64),
    /// UTF-8 string.
    String(std::string::String),
    /// Opaque byte array.
    Bytes(Vec<u8>),
}

// ──────────────────────────────────────────────────────────────────────────────
// In-memory Iceberg table
// ──────────────────────────────────────────────────────────────────────────────

/// In-memory Apache Iceberg table for testing and small datasets.
///
/// Stores all column data in `HashMap<column_name, Vec<IcebergValue>>` and
/// maintains a snapshot history for time-travel queries.
pub struct IcebergTable {
    metadata: IcebergTableMetadata,
    /// Current column data, keyed by column name.
    data: HashMap<std::string::String, Vec<IcebergValue>>,
    /// Snapshot row-count history: `snapshot_id -> row_count`.
    snapshot_row_counts: HashMap<i64, usize>,
}

impl IcebergTable {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create a new empty table at `location` with the given `schema`.
    pub fn new(location: impl Into<std::string::String>, schema: IcebergSchema) -> Self {
        let location = location.into();
        let last_column_id = schema.fields.iter().map(|f| f.id).max().unwrap_or(0);
        let metadata = IcebergTableMetadata {
            format_version: 2,
            table_uuid: Uuid::new_v4().to_string(),
            location: location.clone(),
            last_sequence_number: 0,
            last_updated_ms: current_timestamp_ms(),
            last_column_id,
            schema: schema.clone(),
            snapshots: Vec::new(),
            current_snapshot_id: None,
        };
        let data = schema
            .fields
            .iter()
            .map(|f| (f.name.clone(), Vec::new()))
            .collect();
        IcebergTable {
            metadata,
            data,
            snapshot_row_counts: HashMap::new(),
        }
    }

    // ── Write operations ──────────────────────────────────────────────────────

    /// Append `rows` to the table.
    ///
    /// `rows` is a map from column name to a `Vec` of values. All column
    /// vectors must have the same length; columns not present in `rows` will
    /// have `IcebergValue::Null` appended for each new row.
    pub fn append(
        &mut self,
        rows: HashMap<std::string::String, Vec<IcebergValue>>,
    ) -> Result<(), IcebergError> {
        // Determine row count from the first column supplied.
        let row_count = rows.values().next().map(|v| v.len()).unwrap_or(0);

        // Validate all supplied columns have the same length.
        for (col_name, col_data) in &rows {
            if col_data.len() != row_count {
                return Err(IcebergError::SchemaMismatch(format!(
                    "Column '{col_name}' has {} values but expected {row_count}",
                    col_data.len()
                )));
            }
        }

        // Validate all supplied column names are in the schema.
        for col_name in rows.keys() {
            if !self.data.contains_key(col_name.as_str()) {
                return Err(IcebergError::SchemaMismatch(format!(
                    "Column '{col_name}' not found in schema"
                )));
            }
        }

        // Append data, filling missing columns with Null.
        for field in &self.metadata.schema.fields.clone() {
            let column = self.data.entry(field.name.clone()).or_default();
            if let Some(new_values) = rows.get(&field.name) {
                column.extend(new_values.iter().cloned());
            } else {
                column.extend(std::iter::repeat(IcebergValue::Null).take(row_count));
            }
        }

        self.metadata.last_updated_ms = current_timestamp_ms();
        Ok(())
    }

    // ── Read operations ───────────────────────────────────────────────────────

    /// Return the number of rows currently in the table.
    pub fn num_rows(&self) -> usize {
        self.data.values().next().map(|v| v.len()).unwrap_or(0)
    }

    /// Return the column names in schema order.
    pub fn column_names(&self) -> Vec<&str> {
        self.metadata
            .schema
            .fields
            .iter()
            .map(|f| f.name.as_str())
            .collect()
    }

    /// Read a column as a slice of values. Returns `None` if the column does not exist.
    pub fn read_column(&self, name: &str) -> Option<&[IcebergValue]> {
        self.data.get(name).map(|v| v.as_slice())
    }

    // ── Snapshot management ───────────────────────────────────────────────────

    /// Commit the current state as a new snapshot and return the new snapshot ID.
    pub fn commit_snapshot(&mut self) -> i64 {
        let snapshot_id = generate_snapshot_id();
        let sequence_number = self.metadata.last_sequence_number + 1;
        let row_count = self.num_rows();

        let parent_id = self.metadata.current_snapshot_id;
        let mut summary = HashMap::new();
        summary.insert("operation".to_string(), "append".to_string());
        summary.insert("added-records".to_string(), row_count.to_string());

        let snapshot = IcebergSnapshot {
            snapshot_id,
            parent_snapshot_id: parent_id,
            sequence_number,
            timestamp_ms: current_timestamp_ms(),
            manifest_list: format!(
                "{}/metadata/snap-{}-1-manifest-list.avro",
                self.metadata.location, snapshot_id
            ),
            summary,
        };

        self.snapshot_row_counts.insert(snapshot_id, row_count);
        self.metadata.snapshots.push(snapshot);
        self.metadata.current_snapshot_id = Some(snapshot_id);
        self.metadata.last_sequence_number = sequence_number;
        self.metadata.last_updated_ms = current_timestamp_ms();

        snapshot_id
    }

    /// Look up metadata for a specific snapshot by ID.
    ///
    /// Returns `None` if no snapshot with `snapshot_id` exists.
    pub fn as_of_snapshot(&self, snapshot_id: i64) -> Option<IcebergSnapshot> {
        self.metadata
            .snapshots
            .iter()
            .find(|s| s.snapshot_id == snapshot_id)
            .cloned()
    }

    // ── Metadata I/O ─────────────────────────────────────────────────────────

    /// Write the table metadata as a JSON file to `path`.
    pub fn write_metadata_json(&self, path: &Path) -> Result<(), IcebergError> {
        let json_value = self.metadata.to_json();
        let json_bytes = serde_json::to_vec_pretty(&json_value).map_err(IcebergError::Json)?;
        std::fs::write(path, json_bytes).map_err(IcebergError::Io)
    }

    /// Read table metadata from a JSON file at `path`.
    pub fn read_metadata_json(path: &Path) -> Result<IcebergTableMetadata, IcebergError> {
        let bytes = std::fs::read(path).map_err(IcebergError::Io)?;
        let json_value: Value = serde_json::from_slice(&bytes).map_err(IcebergError::Json)?;
        IcebergTableMetadata::from_json(&json_value)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

fn current_timestamp_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn generate_snapshot_id() -> i64 {
    // Use a combination of timestamp and a small random-ish component derived
    // from the UUID to produce a stable but unique i64 snapshot ID.
    let ts = current_timestamp_ms();
    let uuid_hi = Uuid::new_v4().as_u64_pair().0 as i64;
    ts ^ (uuid_hi & 0x0000_FFFF_FFFF_FFFF)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_schema() -> IcebergSchema {
        IcebergSchema {
            schema_id: 1,
            fields: vec![
                IcebergField {
                    id: 1,
                    name: "id".to_string(),
                    field_type: IcebergType::Long,
                    required: true,
                    doc: None,
                },
                IcebergField {
                    id: 2,
                    name: "value".to_string(),
                    field_type: IcebergType::Double,
                    required: false,
                    doc: Some("sensor reading".to_string()),
                },
                IcebergField {
                    id: 3,
                    name: "label".to_string(),
                    field_type: IcebergType::String,
                    required: false,
                    doc: None,
                },
            ],
        }
    }

    #[test]
    fn test_iceberg_create_append() {
        let mut table = IcebergTable::new("s3://bucket/test-table", make_schema());
        assert_eq!(table.num_rows(), 0);

        let mut rows = HashMap::new();
        rows.insert(
            "id".to_string(),
            vec![IcebergValue::Long(1), IcebergValue::Long(2)],
        );
        rows.insert(
            "value".to_string(),
            vec![
                IcebergValue::Double(std::f64::consts::PI),
                IcebergValue::Double(std::f64::consts::E),
            ],
        );
        rows.insert(
            "label".to_string(),
            vec![
                IcebergValue::String("a".to_string()),
                IcebergValue::String("b".to_string()),
            ],
        );
        table.append(rows).expect("append failed");
        assert_eq!(table.num_rows(), 2);
    }

    #[test]
    fn test_iceberg_schema() {
        let loc = format!(
            "file://{}",
            std::env::temp_dir().join("scirs2_iceberg_test").display()
        );
        let table = IcebergTable::new(loc, make_schema());
        let names = table.column_names();
        assert_eq!(names, vec!["id", "value", "label"]);
    }

    #[test]
    fn test_iceberg_snapshot() {
        let loc = format!(
            "file://{}",
            std::env::temp_dir().join("scirs2_iceberg_test").display()
        );
        let mut table = IcebergTable::new(loc, make_schema());

        let mut rows = HashMap::new();
        rows.insert("id".to_string(), vec![IcebergValue::Long(42)]);
        rows.insert("value".to_string(), vec![IcebergValue::Double(1.0)]);
        rows.insert(
            "label".to_string(),
            vec![IcebergValue::String("x".to_string())],
        );
        table.append(rows).expect("append failed");

        let snap_id = table.commit_snapshot();
        assert!(snap_id != 0);

        let snap = table.as_of_snapshot(snap_id).expect("snapshot not found");
        assert_eq!(snap.snapshot_id, snap_id);
        assert_eq!(table.metadata.current_snapshot_id, Some(snap_id));
    }

    #[test]
    fn test_iceberg_metadata_json() {
        let meta_loc = format!(
            "file://{}",
            std::env::temp_dir()
                .join("scirs2_iceberg_meta_test")
                .display()
        );
        let mut table = IcebergTable::new(meta_loc, make_schema());

        let mut rows = HashMap::new();
        rows.insert(
            "id".to_string(),
            vec![IcebergValue::Long(1), IcebergValue::Long(2)],
        );
        rows.insert(
            "value".to_string(),
            vec![IcebergValue::Double(0.5), IcebergValue::Double(1.5)],
        );
        rows.insert(
            "label".to_string(),
            vec![
                IcebergValue::String("p".to_string()),
                IcebergValue::String("q".to_string()),
            ],
        );
        table.append(rows).expect("append failed");
        table.commit_snapshot();

        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("iceberg_test_metadata.json");

        table
            .write_metadata_json(&path)
            .expect("write_metadata_json failed");

        let loaded = IcebergTable::read_metadata_json(&path).expect("read_metadata_json failed");

        assert_eq!(loaded.format_version, 2);
        assert_eq!(loaded.schema.fields.len(), 3);
        assert_eq!(loaded.schema.fields[0].name, "id");
        assert_eq!(loaded.schema.fields[1].name, "value");
        assert!(loaded.current_snapshot_id.is_some());
        assert_eq!(
            loaded.current_snapshot_id,
            table.metadata.current_snapshot_id
        );
    }
}
