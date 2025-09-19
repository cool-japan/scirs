//! SQLite database implementation

use crate::database::{
    ColumnDef, DataType, DatabaseConfig, DatabaseConnection, Index, QueryBuilder, QueryType,
    ResultSet, TableSchema,
};
use crate::error::{IoError, Result};
use ndarray::ArrayView2;
use rusqlite::{params_from_iter, Connection as SqliteConn, ToSql};
use std::sync::Mutex;

/// SQLite connection wrapper
pub struct SQLiteConnection {
    config: DatabaseConfig,
    connection: Option<Mutex<SqliteConn>>,
}

impl SQLiteConnection {
    /// Create a new SQLite connection
    pub fn new(config: &DatabaseConfig) -> Result<Self> {
        let conn = SqliteConn::open(&config.database)
            .map_err(|e| IoError::DatabaseError(format!("SQLite connection failed: {}", e)))?;

        // Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON", [])
            .map_err(|e| IoError::DatabaseError(format!("Failed to enable foreign keys: {}", e)))?;

        Ok(Self {
            config: config.clone(),
            connection: Some(Mutex::new(conn)),
        })
    }
}

impl DatabaseConnection for SQLiteConnection {
    fn query(&self, query: &QueryBuilder) -> Result<ResultSet> {
        let sql = query.build_sql();
        let params = &query.values;
        self.execute_sql(&sql, params)
    }

    fn execute_sql(&self, sql: &str, params: &[serde_json::Value]) -> Result<ResultSet> {
        let conn = self.connection.as_ref().ok_or_else(|| {
            IoError::DatabaseError("SQLite connection not initialized".to_string())
        })?;

        let mut conn = conn.lock().unwrap();

        // Convert JSON params to SQLite values
        let sqlite_params: Vec<Box<dyn ToSql>> = params
            .iter()
            .map(|p| -> Box<dyn ToSql> {
                match p {
                    serde_json::Value::String(s) => Box::new(s.clone()),
                    serde_json::Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            Box::new(i)
                        } else {
                            Box::new(n.as_f64().unwrap())
                        }
                    }
                    serde_json::Value::Bool(b) => Box::new(*b),
                    serde_json::Value::Null => Box::new(None::<String>),
                    serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                        Box::new(p.to_string())
                    }
                }
            })
            .collect();

        let mut stmt = conn.prepare(sql).map_err(|e| {
            IoError::DatabaseError(format!("SQLite query preparation failed: {}", e))
        })?;

        let column_count = stmt.column_count();
        let mut column_names = Vec::new();
        for i in 0..column_count {
            column_names.push(
                stmt.column_name(i)
                    .map(String::from)
                    .unwrap_or_else(|_| format!("column_{}", i)),
            );
        }

        let mut result = ResultSet::new(column_names);

        // Note: This is a simplified implementation
        // In production, you'd need more sophisticated parameter handling
        if params.is_empty() {
            let mut rows = stmt
                .query([])
                .map_err(|e| IoError::DatabaseError(format!("Query execution failed: {}", e)))?;

            while let Some(row) = rows
                .next()
                .map_err(|e| IoError::DatabaseError(format!("Row fetch failed: {}", e)))?
            {
                let mut row_data = Vec::new();
                for i in 0..column_count {
                    let value = match row.get_ref(i) {
                        Ok(val) => match val {
                            rusqlite::types::ValueRef::Null => serde_json::Value::Null,
                            rusqlite::types::ValueRef::Integer(i) => serde_json::json!(i),
                            rusqlite::types::ValueRef::Real(f) => serde_json::json!(f),
                            rusqlite::types::ValueRef::Text(s) => {
                                serde_json::json!(String::from_utf8_lossy(s))
                            }
                            rusqlite::types::ValueRef::Blob(b) => {
                                serde_json::json!(data_encoding::BASE64.encode(b))
                            }
                        },
                        Err(_) => serde_json::Value::Null,
                    };
                    row_data.push(value);
                }
                result.add_row(row_data);
            }
        }

        Ok(result)
    }

    fn insert_array(&self, table: &str, data: ArrayView2<f64>, columns: &[&str]) -> Result<usize> {
        let conn = self.connection.as_ref().ok_or_else(|| {
            IoError::DatabaseError("SQLite connection not initialized".to_string())
        })?;

        let mut conn = conn.lock().unwrap();

        if columns.len() != data.ncols() {
            return Err(IoError::ValidationError(
                "Number of columns doesn't match data dimensions".to_string(),
            ));
        }

        let placeholders: Vec<String> = (0..columns.len()).map(|_| "?".to_string()).collect();
        let insert_sql = format!(
            "INSERT INTO {} ({}) VALUES ({})",
            table,
            columns.join(", "),
            placeholders.join(", ")
        );

        let mut total_inserted = 0;
        let tx = conn
            .transaction()
            .map_err(|e| IoError::DatabaseError(format!("Transaction start failed: {}", e)))?;

        {
            let mut stmt = tx.prepare(&insert_sql).map_err(|e| {
                IoError::DatabaseError(format!("Insert statement preparation failed: {}", e))
            })?;

            for row in data.rows() {
                let row_params: Vec<f64> = row.iter().copied().collect();
                stmt.execute(params_from_iter(row_params.iter()))
                    .map_err(|e| IoError::DatabaseError(format!("Row insert failed: {}", e)))?;
                total_inserted += 1;
            }
        }

        tx.commit()
            .map_err(|e| IoError::DatabaseError(format!("Transaction commit failed: {}", e)))?;

        Ok(total_inserted)
    }

    fn create_table(&self, table: &str, schema: &TableSchema) -> Result<()> {
        let conn = self.connection.as_ref().ok_or_else(|| {
            IoError::DatabaseError("SQLite connection not initialized".to_string())
        })?;

        let mut conn = conn.lock().unwrap();

        let mut create_sql = format!("CREATE TABLE {} (", table);

        let column_defs: Vec<String> = schema
            .columns
            .iter()
            .map(|col| {
                let sqlite_type = match col.data_type {
                    DataType::Integer => "INTEGER",
                    DataType::BigInt => "INTEGER",
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

        create_sql.push_str(&column_defs.join(", "));

        if let Some(ref pk_cols) = schema.primary_key {
            create_sql.push_str(&format!(", PRIMARY KEY ({})", pk_cols.join(", ")));
        }

        create_sql.push(')');

        conn.execute(&create_sql, [])
            .map_err(|e| IoError::DatabaseError(format!("Table creation failed: {}", e)))?;

        Ok(())
    }

    fn table_exists(&self, table: &str) -> Result<bool> {
        let conn = self.connection.as_ref().ok_or_else(|| {
            IoError::DatabaseError("SQLite connection not initialized".to_string())
        })?;

        let mut conn = conn.lock().unwrap();

        let count: i32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
                [table],
                |row| row.get(0),
            )
            .map_err(|e| IoError::DatabaseError(format!("Table existence check failed: {}", e)))?;

        Ok(count > 0)
    }

    fn get_schema(&self, table: &str) -> Result<TableSchema> {
        let conn = self.connection.as_ref().ok_or_else(|| {
            IoError::DatabaseError("SQLite connection not initialized".to_string())
        })?;

        let mut conn = conn.lock().unwrap();

        // Get column information using PRAGMA
        let mut stmt = conn
            .prepare(&format!("PRAGMA table_info({})", table))
            .map_err(|e| IoError::DatabaseError(format!("Schema query failed: {}", e)))?;

        let column_rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(1)?,         // name
                    row.get::<_, String>(2)?,         // type
                    row.get::<_, i32>(3)?,            // notnull
                    row.get::<_, Option<String>>(4)?, // default
                    row.get::<_, i32>(5)?,            // pk
                ))
            })
            .map_err(|e| IoError::DatabaseError(format!("Schema query failed: {}", e)))?;

        let mut columns = Vec::new();
        let mut primary_key = Vec::new();

        for row_result in column_rows {
            let (name, type_str, notnull, default, pk) = row_result.map_err(|e| {
                IoError::DatabaseError(format!("Schema row processing failed: {}", e))
            })?;

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
                default: default.map(|s| serde_json::Value::String(s)),
            });

            if pk > 0 {
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
            indexes: Vec::new(), // Could query sqlite_master for indexes
        })
    }
}
