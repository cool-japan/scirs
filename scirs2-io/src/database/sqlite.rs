//! SQLite database implementation — Pure Rust via `oxisql-sqlite-compat` (Limbo engine).
//!
//! # Architecture: sync/async bridge
//!
//! The `DatabaseConnection` trait used in scirs2-io is **synchronous**, while
//! `oxisql-sqlite-compat` (and the underlying Limbo engine) is fully async (tokio).
//!
//! The bridge function `run_sync` resolves this impedance mismatch:
//! - If a tokio runtime is already active (`Handle::try_current()` succeeds) it
//!   uses `block_in_place` + `Handle::block_on` so that the current thread can
//!   block without parking the entire executor.
//! - Otherwise it spins up a fresh single-threaded `Runtime` for the call.
//!
//! # Limbo 0.0.22 caveats
//!
//! - **ROLLBACK**: not implemented by Limbo; `SqliteTransaction::rollback()`
//!   returns an error.  This file does not call `rollback()` — `commit()` is
//!   the only transaction completion path used here.
//! - **Affected-row count**: retrieved via `SELECT changes()` internally by
//!   `oxisql-sqlite-compat`, adding one round-trip per DML.

use crate::database::{
    ColumnDef, DataType, DatabaseConfig, DatabaseConnection, Index, QueryBuilder, QueryType,
    ResultSet, TableSchema,
};
use crate::error::{IoError, Result};
use oxisql_core::{Connection as OxiConnection, Row as OxiRow, ToSqlValue, Transaction, Value};
use oxisql_sqlite_compat::SqliteConnection;
use scirs2_core::ndarray::ArrayView2;
use std::future::Future;
use std::sync::Mutex;

// ── sync/async bridge ──────────────────────────────────────────────────────────

/// Run an async future to completion on the current thread, blocking it.
///
/// Works both inside an existing tokio multi-threaded runtime (via
/// `block_in_place`) and outside any runtime (via a freshly created
/// single-threaded `Runtime`).
fn run_sync<F, T, E>(fut: F) -> std::result::Result<T, E>
where
    F: Future<Output = std::result::Result<T, E>>,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            // We are inside an existing async runtime. Use block_in_place so
            // that the current tokio worker thread is temporarily allowed to
            // perform blocking work without stalling the scheduler.
            tokio::task::block_in_place(|| handle.block_on(fut))
        }
        Err(_) => {
            // No runtime active — create a minimal one for this call.
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|_| {
                    // This path is only reachable if tokio runtime creation
                    // fails (extremely rare). We can't propagate IoError here
                    // because the error type is generic E, so we have to panic.
                    // In practice this should never happen.
                    panic!("scirs2-io sqlite: failed to create tokio runtime for sync bridge")
                })
                .expect("tokio runtime creation cannot fail in practice");
            rt.block_on(fut)
        }
    }
}

/// Convert an [`oxisql_core::OxiSqlError`] into our [`IoError`].
fn oxi_err(e: oxisql_core::OxiSqlError) -> IoError {
    IoError::DatabaseError(e.to_string())
}

/// Convert a single [`oxisql_core::Value`] into a [`serde_json::Value`].
fn oxi_value_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::json!(*b),
        Value::I64(n) => serde_json::json!(*n),
        Value::F64(f) => serde_json::json!(*f),
        Value::Text(s) => serde_json::json!(s),
        Value::Blob(b) => serde_json::json!(crate::encoding_utils::base64_encode(b)),
        Value::Timestamp(ts) => serde_json::json!(*ts),
        Value::Date(d) => serde_json::json!(*d),
        Value::Time(t) => serde_json::json!(*t),
        Value::Uuid(u) => serde_json::json!(format!("{v}")),
        Value::Json(j) => serde_json::from_str(j).unwrap_or_else(|_| serde_json::json!(j)),
        Value::Decimal(d) => serde_json::json!(d),
        Value::Array(arr) => serde_json::Value::Array(arr.iter().map(oxi_value_to_json).collect()),
        // A typed array carries the same element values as a plain array plus a
        // nominal element type; the type annotation has no JSON representation,
        // so serialize the elements as a JSON array.
        Value::TypedArray { values, .. } => {
            serde_json::Value::Array(values.iter().map(oxi_value_to_json).collect())
        }
    }
}

/// Convert a [`serde_json::Value`] parameter into a boxed [`ToSqlValue`].
fn json_param_to_oxi(p: &serde_json::Value) -> Value {
    match p {
        serde_json::Value::String(s) => Value::Text(s.clone()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::I64(i)
            } else {
                Value::F64(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => Value::Text(p.to_string()),
    }
}

// ── SQLiteConnection ──────────────────────────────────────────────────────────

/// SQLite connection wrapper backed by `oxisql-sqlite-compat` (Pure Rust / Limbo).
pub struct SQLiteConnection {
    config: DatabaseConfig,
    /// The underlying async Limbo connection, wrapped in `Mutex` so that the
    /// synchronous `DatabaseConnection` methods can access it from any thread.
    connection: Option<Mutex<SqliteConnection>>,
}

impl SQLiteConnection {
    /// Open or create the SQLite database at the path given in `config.database`.
    ///
    /// Use `":memory:"` for an ephemeral in-memory database.
    pub fn new(config: &DatabaseConfig) -> Result<Self> {
        let conn = run_sync(SqliteConnection::open(&config.database)).map_err(oxi_err)?;

        // Enable foreign key constraints (best-effort; ignore error on failure).
        let _ = run_sync(conn.execute("PRAGMA foreign_keys = ON", &[]));

        Ok(Self {
            config: config.clone(),
            connection: Some(Mutex::new(conn)),
        })
    }

    /// Obtain a lock on the inner connection, returning an error if the
    /// connection was never initialised.
    fn with_conn<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&SqliteConnection) -> Result<T>,
    {
        let guard = self
            .connection
            .as_ref()
            .ok_or_else(|| IoError::DatabaseError("SQLite connection not initialised".to_string()))?
            .lock()
            .map_err(|_| IoError::DatabaseError("SQLite connection mutex poisoned".to_string()))?;
        f(&guard)
    }
}

impl DatabaseConnection for SQLiteConnection {
    fn query(&self, query: &QueryBuilder) -> Result<ResultSet> {
        let sql = query.build_sql();
        let params = &query.values;
        self.execute_sql(&sql, params)
    }

    fn execute_sql(&self, sql: &str, params: &[serde_json::Value]) -> Result<ResultSet> {
        self.with_conn(|conn| {
            // Convert serde_json params → oxisql Values, then collect refs.
            let oxi_params: Vec<Value> = params.iter().map(json_param_to_oxi).collect();
            let param_refs: Vec<&dyn ToSqlValue> =
                oxi_params.iter().map(|v| v as &dyn ToSqlValue).collect();

            let rows = run_sync(conn.query(sql, &param_refs)).map_err(oxi_err)?;

            // Derive column names from the first row, or an empty vec if no rows.
            let column_names: Vec<String> = rows
                .first()
                .map(|r| r.columns().to_vec())
                .unwrap_or_default();

            let mut result = ResultSet::new(column_names.clone());

            for row in &rows {
                let col_count = row.column_count();
                let mut row_data = Vec::with_capacity(col_count);
                for i in 0..col_count {
                    let v = row
                        .get_by_index(i)
                        .map(oxi_value_to_json)
                        .unwrap_or(serde_json::Value::Null);
                    row_data.push(v);
                }
                result.add_row(row_data);
            }

            Ok(result)
        })
    }

    fn insert_array(&self, table: &str, data: ArrayView2<f64>, columns: &[&str]) -> Result<usize> {
        if columns.len() != data.ncols() {
            return Err(IoError::ValidationError(
                "Number of columns doesn't match data dimensions".to_string(),
            ));
        }

        self.with_conn(|conn| {
            let placeholders: Vec<String> =
                (1..=columns.len()).map(|i| format!("${}", i)).collect();
            let insert_sql = format!(
                "INSERT INTO {} ({}) VALUES ({})",
                table,
                columns.join(", "),
                placeholders.join(", ")
            );

            // Open a transaction for the bulk insert.
            let mut txn = run_sync(conn.transaction()).map_err(oxi_err)?;

            for row in data.rows() {
                let row_vals: Vec<Value> = row.iter().map(|&f| Value::F64(f)).collect();
                let row_refs: Vec<&dyn ToSqlValue> =
                    row_vals.iter().map(|v| v as &dyn ToSqlValue).collect();
                run_sync(txn.execute(&insert_sql, &row_refs)).map_err(oxi_err)?;
            }

            run_sync(txn.commit()).map_err(oxi_err)?;
            // Any failed insert returns early above, so all rows were inserted.
            Ok(data.nrows())
        })
    }

    fn create_table(&self, table: &str, schema: &TableSchema) -> Result<()> {
        self.with_conn(|conn| {
            let column_defs: Vec<String> = schema
                .columns
                .iter()
                .map(|col| {
                    let sqlite_type = match col.data_type {
                        DataType::Integer | DataType::BigInt => "INTEGER",
                        DataType::Float | DataType::Double => "REAL",
                        DataType::Decimal(_, _) => "REAL",
                        DataType::Varchar(_) | DataType::Text => "TEXT",
                        DataType::Boolean => "INTEGER",
                        DataType::Date | DataType::Timestamp => "TEXT",
                        DataType::Json => "TEXT",
                        DataType::Binary => "BLOB",
                    };
                    let nullable = if col.nullable { "" } else { " NOT NULL" };
                    format!("{} {}{}", col.name, sqlite_type, nullable)
                })
                .collect();

            let mut create_sql = format!("CREATE TABLE {} (", table);
            create_sql.push_str(&column_defs.join(", "));

            if let Some(ref pk_cols) = schema.primary_key {
                create_sql.push_str(&format!(", PRIMARY KEY ({})", pk_cols.join(", ")));
            }
            create_sql.push(')');

            run_sync(conn.execute(&create_sql, &[])).map_err(oxi_err)?;
            Ok(())
        })
    }

    fn table_exists(&self, table: &str) -> Result<bool> {
        self.with_conn(|conn| {
            let table_val = Value::Text(table.to_string());
            let params: &[&dyn ToSqlValue] = &[&table_val];
            let rows = run_sync(conn.query(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=$1",
                params,
            ))
            .map_err(oxi_err)?;

            let count = rows
                .first()
                .and_then(|r| r.get_by_index(0))
                .and_then(|v| {
                    if let Value::I64(n) = v {
                        Some(*n)
                    } else {
                        None
                    }
                })
                .unwrap_or(0);

            Ok(count > 0)
        })
    }

    fn get_schema(&self, table: &str) -> Result<TableSchema> {
        self.with_conn(|conn| {
            // PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
            let sql = format!("PRAGMA table_info({})", table);
            let rows = run_sync(conn.query(&sql, &[])).map_err(oxi_err)?;

            let mut columns = Vec::new();
            let mut primary_key: Vec<String> = Vec::new();

            for row in &rows {
                let name = row
                    .get_by_index(1)
                    .and_then(|v| {
                        if let Value::Text(s) = v {
                            Some(s.clone())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default();

                let type_str = row
                    .get_by_index(2)
                    .and_then(|v| {
                        if let Value::Text(s) = v {
                            Some(s.clone())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default();

                let notnull = row
                    .get_by_index(3)
                    .and_then(|v| {
                        if let Value::I64(n) = v {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);

                let default_val = row.get_by_index(4).and_then(|v| match v {
                    Value::Text(s) => Some(serde_json::Value::String(s.clone())),
                    Value::Null => None,
                    other => Some(serde_json::json!(format!("{:?}", other))),
                });

                let pk_flag = row
                    .get_by_index(5)
                    .and_then(|v| {
                        if let Value::I64(n) = v {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);

                let data_type = match type_str.to_uppercase().as_str() {
                    "INTEGER" => DataType::Integer,
                    "REAL" => DataType::Double,
                    "TEXT" => DataType::Text,
                    "BLOB" => DataType::Binary,
                    _ => DataType::Text,
                };

                columns.push(ColumnDef {
                    name: name.clone(),
                    data_type,
                    nullable: notnull == 0,
                    default: default_val,
                });

                if pk_flag > 0 {
                    primary_key.push(name);
                }
            }

            Ok(TableSchema {
                name: table.to_string(),
                columns,
                primary_key: if primary_key.is_empty() {
                    None
                } else {
                    Some(primary_key)
                },
                indexes: Vec::new(),
            })
        })
    }
}
