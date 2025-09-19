//! DuckDB database implementation

use crate::database::{
    ColumnDef, DataType, DatabaseConfig, DatabaseConnection, Index, QueryBuilder, QueryType,
    ResultSet, TableSchema,
};
use crate::error::{IoError, Result};
use duckdb::{params_from_iter, Connection as DuckdbConn, ToSql};
use ndarray::ArrayView2;
use std::sync::{Arc, Mutex};

/// DuckDB connection wrapper
pub struct DuckDBConnection {
    config: DatabaseConfig,
    connection: Arc<Mutex<Option<DuckdbConn>>>,
}

impl DuckDBConnection {
    /// Create a new DuckDB connection
    pub fn new(config: &DatabaseConfig) -> Result<Self> {
        // Create actual DuckDB connection
        let conn = if config.database == ":memory:" {
            DuckdbConn::open_in_memory()
        } else {
            DuckdbConn::open(&config.database)
        }
        .map_err(|e| IoError::DatabaseError(format!("DuckDB connection failed: {}", e)))?;

        Ok(Self {
            config: config.clone(),
            connection: Arc::new(Mutex::new(Some(conn))),
        })
    }

    /// Helper to process a DuckDB row into JSON values
    fn process_row(
        row: &duckdb::Row,
        column_count: usize,
    ) -> duckdb::Result<Vec<serde_json::Value>> {
        let mut row_data = Vec::new();
        for i in 0..column_count {
            let value = match row.get_ref(i) {
                Ok(val) => {
                    use duckdb::types::ValueRef;
                    match val {
                        ValueRef::Null => serde_json::Value::Null,
                        ValueRef::Boolean(b) => serde_json::json!(b),
                        ValueRef::TinyInt(i) => serde_json::json!(i),
                        ValueRef::SmallInt(i) => serde_json::json!(i),
                        ValueRef::Int(i) => serde_json::json!(i),
                        ValueRef::BigInt(i) => serde_json::json!(i),
                        // UTinyInt, USmallInt, UInt, UBigInt variants
                        ValueRef::HugeInt(i) => serde_json::json!(i.to_string()),
                        ValueRef::Float(f) => serde_json::json!(f),
                        ValueRef::Double(f) => serde_json::json!(f),
                        ValueRef::Decimal(d) => serde_json::json!(d.to_string()),
                        ValueRef::Text(s) => serde_json::json!(String::from_utf8_lossy(s)),
                        ValueRef::Blob(b) => serde_json::json!(data_encoding::BASE64.encode(b)),
                        ValueRef::Date32(d) => serde_json::json!(d),
                        ValueRef::Time64(unit, t) => {
                            use duckdb::types::TimeUnit;
                            match unit {
                                TimeUnit::Second => serde_json::json!(format!("time_s:{}", t)),
                                TimeUnit::Millisecond => {
                                    serde_json::json!(format!("time_ms:{}", t))
                                }
                                TimeUnit::Microsecond => {
                                    serde_json::json!(format!("time_us:{}", t))
                                }
                                TimeUnit::Nanosecond => serde_json::json!(format!("time_ns:{}", t)),
                            }
                        }
                        ValueRef::Timestamp(unit, v) => {
                            use duckdb::types::TimeUnit;
                            match unit {
                                TimeUnit::Second => serde_json::json!(format!("timestamp_s:{}", v)),
                                TimeUnit::Millisecond => {
                                    serde_json::json!(format!("timestamp_ms:{}", v))
                                }
                                TimeUnit::Microsecond => {
                                    serde_json::json!(format!("timestamp_us:{}", v))
                                }
                                TimeUnit::Nanosecond => {
                                    serde_json::json!(format!("timestamp_ns:{}", v))
                                }
                            }
                        }
                        ValueRef::Interval {
                            months,
                            days,
                            nanos,
                        } => {
                            serde_json::json!({
                                "months": months,
                                "days": days,
                                "nanos": nanos
                            })
                        }
                        ValueRef::Enum(_enum_type, idx) => {
                            serde_json::json!(format!("enum_idx:{}", idx))
                        }
                        _ => serde_json::Value::Null,
                    }
                }
                Err(_) => serde_json::Value::Null,
            };
            row_data.push(value);
        }
        Ok(row_data)
    }
}

impl DatabaseConnection for DuckDBConnection {
    fn query(&self, query: &QueryBuilder) -> Result<ResultSet> {
        let sql = query.build_sql();
        let params = &query.values;
        self.execute_sql(&sql, params)
    }

    fn execute_sql(&self, sql: &str, params: &[serde_json::Value]) -> Result<ResultSet> {
        let conn_guard = self.connection.lock().unwrap();
        let conn = conn_guard.as_ref().ok_or_else(|| {
            IoError::DatabaseError("DuckDB connection not initialized".to_string())
        })?;

        // Convert JSON params to DuckDB values - we need to store owned values
        let mut owned_strings = Vec::new();
        let mut owned_i64s = Vec::new();
        let mut owned_f64s = Vec::new();
        let mut owned_bools = Vec::new();

        // First pass: collect all owned values
        for p in params {
            match p {
                serde_json::Value::String(s) => owned_strings.push(s.clone()),
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        owned_i64s.push(i);
                    } else if let Some(f) = n.as_f64() {
                        owned_f64s.push(f);
                    }
                }
                serde_json::Value::Bool(b) => owned_bools.push(*b),
                serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                    // For complex types, convert to string
                    owned_strings.push(p.to_string());
                }
                _ => {}
            }
        }

        // Second pass: create references to the owned values
        let mut duck_params: Vec<&dyn ToSql> = Vec::new();
        let mut string_idx = 0;
        let mut i64_idx = 0;
        let mut f64_idx = 0;
        let mut bool_idx = 0;

        for p in params {
            match p {
                serde_json::Value::String(_) => {
                    duck_params.push(&owned_strings[string_idx]);
                    string_idx += 1;
                }
                serde_json::Value::Number(n) => {
                    if n.as_i64().is_some() {
                        duck_params.push(&owned_i64s[i64_idx]);
                        i64_idx += 1;
                    } else {
                        duck_params.push(&owned_f64s[f64_idx]);
                        f64_idx += 1;
                    }
                }
                serde_json::Value::Bool(_) => {
                    duck_params.push(&owned_bools[bool_idx]);
                    bool_idx += 1;
                }
                serde_json::Value::Null => {
                    duck_params.push(&None::<String> as &dyn ToSql);
                }
                serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                    duck_params.push(&owned_strings[string_idx]);
                    string_idx += 1;
                }
            }
        }

        let mut stmt = conn.prepare(sql).map_err(|e| {
            IoError::DatabaseError(format!("DuckDB query preparation failed: {}", e))
        })?;

        let column_count = stmt.column_count();
        let mut column_names = Vec::new();
        for i in 0..column_count {
            column_names.push(
                stmt.column_name(i)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|_| format!("column_{}", i)),
            );
        }

        let mut result = ResultSet::new(column_names);

        let process_closure = |row: &duckdb::Row| -> duckdb::Result<Vec<serde_json::Value>> {
            Self::process_row(row, column_count)
        };

        let rows = if duck_params.is_empty() {
            stmt.query_map([], process_closure)
        } else {
            stmt.query_map(params_from_iter(&duck_params), process_closure)
        };

        match rows {
            Ok(row_iter) => {
                for row_result in row_iter {
                    match row_result {
                        Ok(row_data) => result.add_row(row_data),
                        Err(e) => {
                            return Err(IoError::DatabaseError(format!(
                                "Row processing failed: {}",
                                e
                            )))
                        }
                    }
                }
                Ok(result)
            }
            Err(e) => Err(IoError::DatabaseError(format!(
                "DuckDB query execution failed: {}",
                e
            ))),
        }
    }

    fn insert_array(&self, table: &str, data: ArrayView2<f64>, columns: &[&str]) -> Result<usize> {
        let conn_guard = self.connection.lock().unwrap();
        let conn = conn_guard.as_ref().ok_or_else(|| {
            IoError::DatabaseError("DuckDB connection not initialized".to_string())
        })?;

        if columns.len() != data.ncols() {
            return Err(IoError::ValidationError(
                "Number of columns doesn't match data dimensions".to_string(),
            ));
        }

        // DuckDB excels at bulk inserts, so we'll use a single INSERT with multiple VALUES
        let placeholders: Vec<String> = (0..columns.len()).map(|_| "?".to_string()).collect();

        let values_clause = format!("({})", placeholders.join(", "));
        let insert_sql = format!(
            "INSERT INTO {} ({}) VALUES {}",
            table,
            columns.join(", "),
            std::iter::repeat(values_clause)
                .take(data.nrows())
                .collect::<Vec<_>>()
                .join(", ")
        );

        // Flatten all data into a single parameter vector
        let mut all_params = Vec::new();
        for row in data.rows() {
            for &value in row.iter() {
                all_params.push(value);
            }
        }

        let duck_params: Vec<&dyn ToSql> = all_params.iter().map(|v| v as &dyn ToSql).collect();

        let rows_affected = conn
            .execute(&insert_sql, params_from_iter(&duck_params))
            .map_err(|e| IoError::DatabaseError(format!("DuckDB bulk insert failed: {}", e)))?;

        Ok(rows_affected)
    }

    fn create_table(&self, table: &str, schema: &TableSchema) -> Result<()> {
        let conn_guard = self.connection.lock().unwrap();
        let conn = conn_guard.as_ref().ok_or_else(|| {
            IoError::DatabaseError("DuckDB connection not initialized".to_string())
        })?;

        let mut create_sql = format!("CREATE TABLE {} (", table);

        let column_defs: Vec<String> = schema
            .columns
            .iter()
            .map(|col| {
                let duck_type = match col.data_type {
                    DataType::Integer => "INTEGER",
                    DataType::BigInt => "BIGINT",
                    DataType::Float => "REAL",
                    DataType::Double => "DOUBLE",
                    DataType::Decimal(p, s) => return format!("DECIMAL({}, {})", p, s),
                    DataType::Varchar(len) => return format!("VARCHAR({})", len),
                    DataType::Text => "TEXT",
                    DataType::Boolean => "BOOLEAN",
                    DataType::Date => "DATE",
                    DataType::Timestamp => "TIMESTAMP",
                    DataType::Json => "JSON",
                    DataType::Binary => "BLOB",
                };

                let nullable = if col.nullable { "" } else { " NOT NULL" };
                format!("{} {}{}", col.name, duck_type, nullable)
            })
            .collect();

        create_sql.push_str(&column_defs.join(", "));

        if let Some(ref pk_cols) = schema.primary_key {
            create_sql.push_str(&format!(", PRIMARY KEY ({})", pk_cols.join(", ")));
        }

        create_sql.push(')');

        conn.execute(&create_sql, [])
            .map_err(|e| IoError::DatabaseError(format!("DuckDB table creation failed: {}", e)))?;

        Ok(())
    }

    fn table_exists(&self, table: &str) -> Result<bool> {
        let conn_guard = self.connection.lock().unwrap();
        let conn = conn_guard.as_ref().ok_or_else(|| {
            IoError::DatabaseError("DuckDB connection not initialized".to_string())
        })?;

        let mut stmt = conn
            .prepare("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?")
            .map_err(|e| {
                IoError::DatabaseError(format!("Table existence query preparation failed: {}", e))
            })?;

        let count: i64 = stmt
            .query_row([table], |row| row.get(0))
            .map_err(|e| IoError::DatabaseError(format!("Table existence check failed: {}", e)))?;

        Ok(count > 0)
    }

    fn get_schema(&self, table: &str) -> Result<TableSchema> {
        let conn_guard = self.connection.lock().unwrap();
        let conn = conn_guard.as_ref().ok_or_else(|| {
            IoError::DatabaseError("DuckDB connection not initialized".to_string())
        })?;

        // Get column information using DuckDB's information schema
        let mut stmt = conn
            .prepare(
                r#"
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = ?
            ORDER BY ordinal_position
            "#,
            )
            .map_err(|e| {
                IoError::DatabaseError(format!("Schema query preparation failed: {}", e))
            })?;

        let column_rows = stmt
            .query_map([table], |row| {
                Ok((
                    row.get::<_, String>(0)?,         // column_name
                    row.get::<_, String>(1)?,         // data_type
                    row.get::<_, String>(2)?,         // is_nullable
                    row.get::<_, Option<String>>(3)?, // column_default
                ))
            })
            .map_err(|e| IoError::DatabaseError(format!("Schema query failed: {}", e)))?;

        let mut columns = Vec::new();
        for row_result in column_rows {
            let (column_name, data_type_str, is_nullable, column_default) =
                row_result.map_err(|e| {
                    IoError::DatabaseError(format!("Schema row processing failed: {}", e))
                })?;

            let data_type = match data_type_str.to_uppercase().as_str() {
                "INTEGER" | "INT" => DataType::Integer,
                "BIGINT" => DataType::BigInt,
                "REAL" | "FLOAT" => DataType::Float,
                "DOUBLE" => DataType::Double,
                s if s.starts_with("VARCHAR") => DataType::Varchar(255),
                "TEXT" => DataType::Text,
                "BOOLEAN" | "BOOL" => DataType::Boolean,
                "DATE" => DataType::Date,
                "TIMESTAMP" => DataType::Timestamp,
                "JSON" => DataType::Json,
                "BLOB" => DataType::Binary,
                _ => DataType::Text, // Default fallback
            };

            columns.push(ColumnDef {
                name: column_name,
                data_type,
                nullable: is_nullable.to_uppercase() == "YES",
                default: column_default.map(|s| serde_json::Value::String(s)),
            });
        }

        // DuckDB doesn't have traditional primary keys like PostgreSQL/MySQL,
        // but we can check for unique constraints
        let primary_key = None; // DuckDB doesn't enforce primary keys at schema level

        // Get index information for DuckDB
        // Note: DuckDB doesn't have persistent indexes like traditional databases
        let indexes = Vec::new(); // DuckDB manages indexes internally

        Ok(TableSchema {
            name: table.to_string(),
            columns,
            primary_key,
            indexes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::DatabaseType;

    #[test]
    fn test_duckdb_connection_in_memory() {
        let config = DatabaseConfig::new(DatabaseType::DuckDB, ":memory:");
        let conn = DuckDBConnection::new(&config);
        assert!(conn.is_ok());
    }
}
