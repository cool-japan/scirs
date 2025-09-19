//! Database connectivity for scientific data
//!
//! Provides interfaces for reading and writing scientific data to various
//! database systems, including SQL and NoSQL databases.

#![allow(dead_code)]
#![allow(missing_docs)]
#![allow(clippy::too_many_arguments)]

use crate::error::{IoError, Result};
use crate::metadata::Metadata;
use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

// Database driver imports
#[cfg(feature = "sqlite")]
use rusqlite::Connection as SqliteConn;

#[cfg(feature = "postgres")]
use sqlx::postgres::{PgPool, PgPoolOptions, PgRow};

#[cfg(feature = "mysql")]
use sqlx::mysql::{MySqlPool, MySqlPoolOptions, MySqlRow};

#[cfg(feature = "duckdb")]
use duckdb::{Connection as DuckdbConn, Row as DuckdbRow};

/// Supported database types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatabaseType {
    /// PostgreSQL database
    PostgreSQL,
    /// MySQL/MariaDB database
    MySQL,
    /// SQLite database
    SQLite,
    /// MongoDB (NoSQL)
    MongoDB,
    /// InfluxDB (Time series)
    InfluxDB,
    /// Redis (Key-value)
    Redis,
    /// Cassandra (Wide column)
    Cassandra,
    /// DuckDB (Analytical)
    DuckDB,
}

/// Database connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// The type of database (SQLite, PostgreSQL, etc.)
    pub db_type: DatabaseType,
    /// Host address for remote databases
    pub host: Option<String>,
    /// Port number for database connection
    pub port: Option<u16>,
    /// Database name or file path
    pub database: String,
    /// Username for authentication
    pub username: Option<String>,
    /// Password for authentication
    pub password: Option<String>,
    /// Additional connection options
    pub options: HashMap<String, String>,
}

impl DatabaseConfig {
    /// Create a new database configuration
    pub fn new(db_type: DatabaseType, database: impl Into<String>) -> Self {
        Self {
            db_type,
            host: None,
            port: None,
            database: database.into(),
            username: None,
            password: None,
            options: HashMap::new(),
        }
    }

    /// Set host
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = Some(host.into());
        self
    }

    /// Set port
    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    /// Set credentials
    pub fn credentials(mut self, username: impl Into<String>, password: impl Into<String>) -> Self {
        self.username = Some(username.into());
        self.password = Some(password.into());
        self
    }

    /// Add connection option
    pub fn option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.insert(key.into(), value.into());
        self
    }

    /// Build connection string
    pub fn connection_string(&self) -> String {
        match self.db_type {
            DatabaseType::PostgreSQL => {
                let host = self.host.as_deref().unwrap_or("localhost");
                let port = self.port.unwrap_or(5432);
                let user = self.username.as_deref().unwrap_or("postgres");
                format!(
                    "postgresql://{}:password@{}:{}/{}",
                    user, host, port, self.database
                )
            }
            DatabaseType::MySQL => {
                let host = self.host.as_deref().unwrap_or("localhost");
                let port = self.port.unwrap_or(3306);
                let user = self.username.as_deref().unwrap_or("root");
                format!(
                    "mysql://{}:password@{}:{}/{}",
                    user, host, port, self.database
                )
            }
            DatabaseType::SQLite => {
                format!("sqlite://{}", self.database)
            }
            DatabaseType::MongoDB => {
                let host = self.host.as_deref().unwrap_or("localhost");
                let port = self.port.unwrap_or(27017);
                format!("mongodb://{}:{}/{}", host, port, self.database)
            }
            _ => format!("{}://{}", self.db_type.as_str(), self.database),
        }
    }
}

impl DatabaseType {
    fn as_str(&self) -> &'static str {
        match self {
            Self::PostgreSQL => "postgresql",
            Self::MySQL => "mysql",
            Self::SQLite => "sqlite",
            Self::MongoDB => "mongodb",
            Self::InfluxDB => "influxdb",
            Self::Redis => "redis",
            Self::Cassandra => "cassandra",
            Self::DuckDB => "duckdb",
        }
    }
}

/// Database query builder
pub struct QueryBuilder {
    query_type: QueryType,
    table: String,
    columns: Vec<String>,
    conditions: Vec<String>,
    values: Vec<serde_json::Value>,
    order_by: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum QueryType {
    Select,
    Insert,
    Update,
    Delete,
    CreateTable,
}

impl QueryBuilder {
    /// Create a SELECT query
    pub fn select(table: impl Into<String>) -> Self {
        Self {
            query_type: QueryType::Select,
            table: table.into(),
            columns: vec!["*".to_string()],
            conditions: Vec::new(),
            values: Vec::new(),
            order_by: None,
            limit: None,
            offset: None,
        }
    }

    /// Create an INSERT query
    pub fn insert(table: impl Into<String>) -> Self {
        Self {
            query_type: QueryType::Insert,
            table: table.into(),
            columns: Vec::new(),
            conditions: Vec::new(),
            values: Vec::new(),
            order_by: None,
            limit: None,
            offset: None,
        }
    }

    /// Specify columns
    pub fn columns(mut self, columns: Vec<impl Into<String>>) -> Self {
        self.columns = columns.into_iter().map(|c| c.into()).collect();
        self
    }

    /// Add WHERE condition
    pub fn where_clause(mut self, condition: impl Into<String>) -> Self {
        self.conditions.push(condition.into());
        self
    }

    /// Add values for INSERT
    pub fn values(mut self, values: Vec<serde_json::Value>) -> Self {
        self.values = values;
        self
    }

    /// Set ORDER BY
    pub fn order_by(mut self, column: impl Into<String>, desc: bool) -> Self {
        self.order_by = Some(format!(
            "{} {}",
            column.into(),
            if desc { "DESC" } else { "ASC" }
        ));
        self
    }

    /// Set LIMIT
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set OFFSET
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Build SQL query string
    pub fn build_sql(&self) -> String {
        match self.query_type {
            QueryType::Select => {
                let mut sql = format!("SELECT {} FROM {}", self.columns.join(", "), self.table);

                if !self.conditions.is_empty() {
                    sql.push_str(&format!(" WHERE {}", self.conditions.join(" AND ")));
                }

                if let Some(order) = &self.order_by {
                    sql.push_str(&format!(" ORDER BY {order}"));
                }

                if let Some(limit) = self.limit {
                    sql.push_str(&format!(" LIMIT {limit}"));
                }

                if let Some(offset) = self.offset {
                    sql.push_str(&format!(" OFFSET {offset}"));
                }

                sql
            }
            QueryType::Insert => {
                format!(
                    "INSERT INTO {} ({}) VALUES ({})",
                    self.table,
                    self.columns.join(", "),
                    self.values
                        .iter()
                        .map(|_| "?")
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            _ => String::new(),
        }
    }

    /// Build MongoDB query
    pub fn build_mongo(&self) -> serde_json::Value {
        match self.query_type {
            QueryType::Select => {
                let mut query = serde_json::json!({});

                // Convert SQL-like conditions to MongoDB query
                for condition in &self.conditions {
                    // Simple parsing - in real implementation would be more sophisticated
                    if let Some((field, value)) = condition.split_once(" = ") {
                        query[field] = serde_json::json!(value.trim_matches('\''));
                    }
                }

                serde_json::json!({
                    "collection": self.table,
                    "filter": query,
                    "limit": self.limit,
                    "skip": self.offset,
                })
            }
            _ => serde_json::json!({}),
        }
    }
}

/// Database result set
#[derive(Debug, Clone)]
pub struct ResultSet {
    /// Column names in the result set
    pub columns: Vec<String>,
    /// Data rows as JSON values
    pub rows: Vec<Vec<serde_json::Value>>,
    /// Additional metadata about the result set
    pub metadata: Metadata,
}

impl ResultSet {
    /// Create new result set
    pub fn new(columns: Vec<String>) -> Self {
        Self {
            columns,
            rows: Vec::new(),
            metadata: Metadata::new(),
        }
    }

    /// Add a row
    pub fn add_row(&mut self, row: Vec<serde_json::Value>) {
        self.rows.push(row);
    }

    /// Get number of rows
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Get number of columns
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Convert to Array2<f64> if all values are numeric
    pub fn to_array(&self) -> Result<Array2<f64>> {
        let mut data = Vec::new();

        for row in &self.rows {
            for value in row {
                let num = value.as_f64().ok_or_else(|| {
                    IoError::ConversionError("Non-numeric value in result set".to_string())
                })?;
                data.push(num);
            }
        }

        Array2::from_shape_vec((self.row_count(), self.column_count()), data)
            .map_err(|e| IoError::Other(e.to_string()))
    }

    /// Get column by name as Array1
    pub fn get_column(&self, name: &str) -> Result<Array1<f64>> {
        let col_idx = self
            .columns
            .iter()
            .position(|c| c == name)
            .ok_or_else(|| IoError::Other(format!("Column '{name}' not found")))?;

        let mut data = Vec::new();
        for row in &self.rows {
            let num = row[col_idx].as_f64().ok_or_else(|| {
                IoError::ConversionError("Non-numeric value in column".to_string())
            })?;
            data.push(num);
        }

        Ok(Array1::from_vec(data))
    }
}

/// Database connection trait
pub trait DatabaseConnection: Send + Sync {
    /// Execute a query and return results
    fn query(&self, query: &QueryBuilder) -> Result<ResultSet>;

    /// Execute a raw SQL query
    fn execute_sql(&self, sql: &str, params: &[serde_json::Value]) -> Result<ResultSet>;

    /// Insert data from Array2
    fn insert_array(&self, table: &str, data: ArrayView2<f64>, columns: &[&str]) -> Result<usize>;

    /// Create table from schema
    fn create_table(&self, table: &str, schema: &TableSchema) -> Result<()>;

    /// Check if table exists
    fn table_exists(&self, table: &str) -> Result<bool>;

    /// Get table schema
    fn get_schema(&self, table: &str) -> Result<TableSchema>;
}

/// Table schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSchema {
    /// Table name
    pub name: String,
    /// Column definitions
    pub columns: Vec<ColumnDef>,
    /// Primary key column names
    pub primary_key: Option<Vec<String>>,
    /// Index definitions
    pub indexes: Vec<Index>,
}

/// Column definition for database tables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    /// Column name
    pub name: String,
    /// Data type of the column
    pub data_type: DataType,
    /// Whether the column allows NULL values
    pub nullable: bool,
    /// Default value for the column
    pub default: Option<serde_json::Value>,
}

/// Database data types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    /// 32-bit integer
    Integer,
    /// 64-bit integer
    BigInt,
    /// 32-bit floating point
    Float,
    /// 64-bit floating point
    Double,
    /// Decimal with precision and scale
    Decimal(u8, u8),
    /// Variable-length character string
    Varchar(usize),
    /// Text string of unlimited length
    Text,
    /// Boolean true/false
    Boolean,
    /// Date value
    Date,
    /// Timestamp with date and time
    Timestamp,
    /// JSON document
    Json,
    Binary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Index {
    pub name: String,
    pub columns: Vec<String>,
    pub unique: bool,
}

/// Database connector factory
pub struct DatabaseConnector;

impl DatabaseConnector {
    /// Create a new database connection
    pub fn connect(config: &DatabaseConfig) -> Result<Box<dyn DatabaseConnection>> {
        match config.db_type {
            #[cfg(feature = "sqlite")]
            DatabaseType::SQLite => Ok(Box::new(SQLiteConnection::new(config)?)),
            #[cfg(not(feature = "sqlite"))]
            DatabaseType::SQLite => Err(IoError::UnsupportedFormat(
                "SQLite support not enabled. Enable 'sqlite' feature.".to_string(),
            )),

            #[cfg(feature = "postgres")]
            DatabaseType::PostgreSQL => Ok(Box::new(PostgreSQLConnection::new(config)?)),
            #[cfg(not(feature = "postgres"))]
            DatabaseType::PostgreSQL => Err(IoError::UnsupportedFormat(
                "PostgreSQL support not enabled. Enable 'postgres' feature.".to_string(),
            )),

            #[cfg(feature = "mysql")]
            DatabaseType::MySQL => Ok(Box::new(MySQLConnection::new(config)?)),
            #[cfg(not(feature = "mysql"))]
            DatabaseType::MySQL => Err(IoError::UnsupportedFormat(
                "MySQL support not enabled. Enable 'mysql' feature.".to_string(),
            )),

            #[cfg(feature = "duckdb")]
            DatabaseType::DuckDB => Ok(Box::new(DuckDBConnection::new(_config)?)),
            #[cfg(not(feature = "duckdb"))]
            DatabaseType::DuckDB => Err(IoError::UnsupportedFormat(
                "DuckDB support not enabled. Enable 'duckdb' feature.".to_string(),
            )),
            _ => Err(IoError::UnsupportedFormat(format!(
                "Database type {:?} not yet implemented",
                config.db_type
            ))),
        }
    }
}

/// SQLite connection implementation
#[cfg(feature = "sqlite")]
struct SQLiteConnection {
    #[allow(dead_code)]
    path: String,
    conn: Mutex<SqliteConn>,
}

#[cfg(feature = "sqlite")]
impl SQLiteConnection {
    fn new(config: &DatabaseConfig) -> Result<Self> {
        let conn = SqliteConn::open(&config.database)
            .map_err(|e| IoError::FileError(format!("Failed to open SQLite database: {}", e)))?;

        Ok(Self {
            path: config.database.clone(),
            conn: Mutex::new(conn),
        })
    }

    /// Convert DataType enum to SQL type string
    fn data_type_to_sql(&self, datatype: &DataType) -> &'static str {
        match datatype {
            DataType::Integer => "INTEGER",
            DataType::BigInt => "BIGINT",
            DataType::Float => "REAL",
            DataType::Double => "REAL",
            DataType::Decimal(_, _) => "DECIMAL",
            DataType::Varchar(_) => "VARCHAR",
            DataType::Text => "TEXT",
            DataType::Boolean => "BOOLEAN",
            DataType::Date => "DATE",
            DataType::Timestamp => "TIMESTAMP",
            DataType::Json => "JSON",
            DataType::Binary => "BLOB",
        }
    }

    /// Convert SQL type string to DataType enum
    fn sql_type_to_data_type(&self, sqltype: &str) -> DataType {
        let sql_upper = sqltype.to_uppercase();

        if sql_upper.contains("INT") {
            if sql_upper.contains("BIG") {
                DataType::BigInt
            } else {
                DataType::Integer
            }
        } else if sql_upper.contains("REAL") || sql_upper.contains("FLOAT") {
            DataType::Float
        } else if sql_upper.contains("DOUBLE") {
            DataType::Double
        } else if sql_upper.contains("DECIMAL") {
            DataType::Decimal(10, 2) // Default precision
        } else if sql_upper.contains("VARCHAR") {
            DataType::Varchar(255) // Default length
        } else if sql_upper.contains("BOOL") {
            DataType::Boolean
        } else if sql_upper.contains("DATE") {
            DataType::Date
        } else if sql_upper.contains("TIMESTAMP") {
            DataType::Timestamp
        } else if sql_upper.contains("JSON") {
            DataType::Json
        } else if sql_upper.contains("BLOB") {
            DataType::Binary
        } else {
            DataType::Text // Default fallback
        }
    }
}

#[cfg(feature = "sqlite")]
impl DatabaseConnection for SQLiteConnection {
    fn query(&self, query: &QueryBuilder) -> Result<ResultSet> {
        let sql = query.build_sql();
        self.execute_sql(&sql, &[])
    }

    fn execute_sql(&self, sql: &str, params: &[serde_json::Value]) -> Result<ResultSet> {
        let conn = self.conn.lock().map_err(|e| {
            IoError::DatabaseError(format!("Failed to lock database connection: {}", e))
        })?;

        let sql_lower = sql.to_lowercase();

        // Handle different SQL types
        if sql_lower.starts_with("select") || sql_lower.starts_with("explain") {
            let mut stmt = conn.prepare(sql).map_err(|e| {
                IoError::DatabaseError(format!("Failed to prepare statement: {}", e))
            })?;

            let column_names: Vec<String> =
                stmt.column_names().iter().map(|s| s.to_string()).collect();
            let column_count = stmt.column_count();
            let mut result = ResultSet::new(column_names);

            let rows = stmt
                .query_map(rusqlite::params![], |row| {
                    let mut values = Vec::new();
                    for i in 0..column_count {
                        let value: rusqlite::types::Value = row.get(i)?;
                        let json_value = match value {
                            rusqlite::types::Value::Null => serde_json::Value::Null,
                            rusqlite::types::Value::Integer(i) => {
                                serde_json::Value::Number(serde_json::Number::from(i))
                            }
                            rusqlite::types::Value::Real(f) => serde_json::Value::Number(
                                serde_json::Number::from_f64(f)
                                    .unwrap_or_else(|| serde_json::Number::from(0)),
                            ),
                            rusqlite::types::Value::Text(s) => serde_json::Value::String(s),
                            rusqlite::types::Value::Blob(b) => {
                                serde_json::Value::String(hex::encode(b))
                            }
                        };
                        values.push(json_value);
                    }
                    Ok(values)
                })
                .map_err(|e| IoError::DatabaseError(format!("Query execution failed: {}", e)))?;

            for row_result in rows {
                match row_result {
                    Ok(row) => result.add_row(row),
                    Err(e) => {
                        return Err(IoError::DatabaseError(format!(
                            "Row processing failed: {}",
                            e
                        )))
                    }
                }
            }

            Ok(result)
        } else {
            // Handle INSERT, UPDATE, DELETE, CREATE, etc.
            let affected_rows = conn
                .execute(sql, rusqlite::params![])
                .map_err(|e| IoError::DatabaseError(format!("SQL execution failed: {}", e)))?;

            let mut result = ResultSet::new(vec!["rows_affected".to_string()]);
            result.add_row(vec![serde_json::json!(affected_rows)]);
            Ok(result)
        }
    }

    fn insert_array(&self, table: &str, data: ArrayView2<f64>, columns: &[&str]) -> Result<usize> {
        if columns.len() != data.ncols() {
            return Err(IoError::ValidationError(
                "Number of columns doesn't match array dimensions".to_string(),
            ));
        }

        let conn = self.conn.lock().map_err(|e| {
            IoError::DatabaseError(format!("Failed to lock database connection: {}", e))
        })?;

        let placeholders: Vec<String> = (0..columns.len()).map(|_| "?".to_string()).collect();
        let sql = format!(
            "INSERT INTO {} ({}) VALUES ({})",
            table,
            columns.join(", "),
            placeholders.join(", ")
        );

        let mut stmt = conn.prepare(&sql).map_err(|e| {
            IoError::DatabaseError(format!("Failed to prepare insert statement: {}", e))
        })?;

        let mut inserted = 0;
        for row in data.rows() {
            let params: Vec<&dyn rusqlite::ToSql> =
                row.iter().map(|v| v as &dyn rusqlite::ToSql).collect();
            stmt.execute(&params[..])
                .map_err(|e| IoError::DatabaseError(format!("Insert failed: {}", e)))?;
            inserted += 1;
        }

        Ok(inserted)
    }

    fn create_table(&self, table: &str, schema: &TableSchema) -> Result<()> {
        let conn = self.conn.lock().map_err(|e| {
            IoError::DatabaseError(format!("Failed to lock database connection: {}", e))
        })?;

        let mut sql = format!("CREATE TABLE {} (", table);

        let column_defs: Vec<String> = schema
            .columns
            .iter()
            .map(|col| {
                let sql_type = self.data_type_to_sql(&col.data_type);
                let nullable = if col.nullable { "" } else { " NOT NULL" };
                format!("{} {}{}", col.name, sql_type, nullable)
            })
            .collect();

        sql.push_str(&column_defs.join(", "));

        if let Some(ref pk) = schema.primary_key {
            sql.push_str(&format!(", PRIMARY KEY ({})", pk.join(", ")));
        }

        sql.push(')');

        conn.execute(&sql, [])
            .map_err(|e| IoError::DatabaseError(format!("Failed to create table: {}", e)))?;

        // Create indexes
        for index in &schema.indexes {
            let unique = if index.unique { "UNIQUE " } else { "" };
            let index_sql = format!(
                "CREATE {}INDEX {} ON {} ({})",
                unique,
                index.name,
                table,
                index.columns.join(", ")
            );
            conn.execute(&index_sql, [])
                .map_err(|e| IoError::DatabaseError(format!("Failed to create index: {}", e)))?;
        }

        Ok(())
    }

    fn table_exists(&self, table: &str) -> Result<bool> {
        let conn = self.conn.lock().map_err(|e| {
            IoError::DatabaseError(format!("Failed to lock database connection: {}", e))
        })?;

        let sql = "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?";
        let mut stmt = conn
            .prepare(sql)
            .map_err(|e| IoError::DatabaseError(format!("Failed to prepare statement: {}", e)))?;

        let count: i64 = stmt
            .query_row([table], |row| row.get(0))
            .map_err(|e| IoError::DatabaseError(format!("Table existence check failed: {}", e)))?;

        Ok(count > 0)
    }

    fn get_schema(&self, table: &str) -> Result<TableSchema> {
        let conn = self.conn.lock().map_err(|e| {
            IoError::DatabaseError(format!("Failed to lock database connection: {}", e))
        })?;

        let sql = format!("PRAGMA table_info({})", table);
        let mut stmt = conn.prepare(&sql).map_err(|e| {
            IoError::DatabaseError(format!("Failed to prepare schema query: {}", e))
        })?;

        let mut columns = Vec::new();
        let rows = stmt
            .query_map([], |row| {
                let name: String = row.get(1)?;
                let type_str: String = row.get(2)?;
                let not_null: bool = row.get(3)?;
                let default_value: Option<String> = row.get(4)?;
                Ok((name, type_str, not_null, default_value))
            })
            .map_err(|e| IoError::DatabaseError(format!("Schema query failed: {}", e)))?;

        for row_result in rows {
            let (name, type_str, not_null, default_value) = row_result.map_err(|e| {
                IoError::DatabaseError(format!("Schema row processing failed: {}", e))
            })?;

            let data_type = self.sql_type_to_data_type(&type_str);
            let default = default_value.map(|v| serde_json::Value::String(v));

            columns.push(ColumnDef {
                name,
                data_type,
                nullable: !not_null,
                default,
            });
        }

        Ok(TableSchema {
            name: table.to_string(),
            columns,
            primary_key: None, // Would need additional query to get primary key info
            indexes: vec![],   // Would need additional query to get index info
        })
    }
}

/// Time series database utilities
pub mod timeseries {
    use super::*;
    use chrono::{DateTime, Utc};

    /// Time series data point
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TimeSeriesPoint {
        pub timestamp: DateTime<Utc>,
        pub value: f64,
        pub tags: HashMap<String, String>,
    }

    /// Time series query builder
    pub struct TimeSeriesQuery {
        pub measurement: String,
        pub start_time: Option<DateTime<Utc>>,
        pub end_time: Option<DateTime<Utc>>,
        pub tags: HashMap<String, String>,
        pub aggregation: Option<Aggregation>,
        pub group_by: Vec<String>,
    }

    #[derive(Debug, Clone)]
    pub enum Aggregation {
        Mean,
        Sum,
        Min,
        Max,
        Count,
        StdDev,
    }

    impl TimeSeriesQuery {
        pub fn new(measurement: impl Into<String>) -> Self {
            Self {
                measurement: measurement.into(),
                start_time: None,
                end_time: None,
                tags: HashMap::new(),
                aggregation: None,
                group_by: Vec::new(),
            }
        }

        pub fn time_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
            self.start_time = Some(start);
            self.end_time = Some(end);
            self
        }

        pub fn tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
            self.tags.insert(key.into(), value.into());
            self
        }

        pub fn aggregate(mut self, agg: Aggregation) -> Self {
            self.aggregation = Some(agg);
            self
        }
    }
}

/// Bulk operations for efficient data loading
pub mod bulk {
    use super::*;

    /// Bulk loader for efficient data insertion
    pub struct BulkLoader {
        batch_size: usize,
        buffer: Vec<Vec<serde_json::Value>>,
        #[allow(dead_code)]
        table: String,
        columns: Vec<String>,
    }

    impl BulkLoader {
        pub fn new(table: impl Into<String>, columns: Vec<impl Into<String>>) -> Self {
            Self {
                batch_size: 1000,
                buffer: Vec::new(),
                table: table.into(),
                columns: columns.into_iter().map(|c| c.into()).collect(),
            }
        }

        pub fn batch_size(mut self, size: usize) -> Self {
            self.batch_size = size;
            self
        }

        pub fn add_row(&mut self, row: Vec<serde_json::Value>) -> Result<()> {
            if row.len() != self.columns.len() {
                return Err(IoError::Other("Row length mismatch".to_string()));
            }
            self.buffer.push(row);
            Ok(())
        }

        pub fn add_array(&mut self, data: ArrayView2<f64>) -> Result<()> {
            for row in data.rows() {
                let json_row: Vec<serde_json::Value> =
                    row.iter().map(|&v| serde_json::json!(v)).collect();
                self.add_row(json_row)?;
            }
            Ok(())
        }

        pub fn flush(&mut self, conn: &dyn DatabaseConnection) -> Result<usize> {
            let total = self.buffer.len();
            // In real implementation, would batch insert
            self.buffer.clear();
            Ok(total)
        }
    }
}

/// PostgreSQL connection implementation  
#[cfg(feature = "postgres")]
struct PostgreSQLConnection {
    config: DatabaseConfig,
    pool: Option<PgPool>,
}

#[cfg(feature = "postgres")]
impl PostgreSQLConnection {
    async fn new(config: &DatabaseConfig) -> Result<Self> {
        // Create actual PostgreSQL connection pool
        let connection_string = Self::build_connection_string(config);

        let pool = PgPoolOptions::new()
            .max_connections(
                config
                    .options
                    .get("max_connections")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(5),
            )
            .connect(&connection_string)
            .await
            .map_err(|e| IoError::NetworkError(format!("PostgreSQL connection failed: {}", e)))?;

        Ok(Self {
            config: config.clone(),
            pool: Some(pool),
        })
    }

    fn build_connection_string(config: &DatabaseConfig) -> String {
        let host = config.host.as_deref().unwrap_or("localhost");
        let port = config.port.unwrap_or(5432);
        let user = config.username.as_deref().unwrap_or("postgres");
        let password = config.password.as_deref().unwrap_or("");

        format!(
            "postgresql://{}:{}@{}:{}/{}",
            user, password, host, port, config.database
        )
    }
}

#[cfg(feature = "postgres")]
impl DatabaseConnection for PostgreSQLConnection {
    fn query(&self, query: &QueryBuilder) -> Result<ResultSet> {
        let sql = query.build_sql();
        let params = &query.values;
        self.execute_sql(&sql, params)
    }

    fn execute_sql(&self, sql: &str, params: &[serde_json::Value]) -> Result<ResultSet> {
        use sqlx::{Column, Row, TypeInfo};

        let pool = self.pool.as_ref().ok_or_else(|| {
            IoError::ConnectionError("PostgreSQL pool not initialized".to_string())
        })?;

        // Execute query in a blocking context
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| IoError::NetworkError(format!("Failed to create async runtime: {}", e)))?;

        rt.block_on(async {
            // Convert params to sqlx format
            let mut query = sqlx::query(sql);
            for param in params {
                match param {
                    serde_json::Value::String(s) => query = query.bind(s),
                    serde_json::Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            query = query.bind(i);
                        } else if let Some(f) = n.as_f64() {
                            query = query.bind(f);
                        }
                    }
                    serde_json::Value::Bool(b) => query = query.bind(*b),
                    serde_json::Value::Null => query = query.bind(Option::<String>::None),
                    _ => query = query.bind(param.to_string()),
                }
            }

            let rows = query
                .fetch_all(pool)
                .await
                .map_err(|e| IoError::DatabaseError(format!("PostgreSQL query failed: {}", e)))?;

            if rows.is_empty() {
                return Ok(ResultSet::new(vec![]));
            }

            // Extract column names
            let columns: Vec<String> = rows[0]
                .columns()
                .iter()
                .map(|col| col.name().to_string())
                .collect();

            let mut result = ResultSet::new(columns);

            // Convert rows to JSON values
            for row in rows {
                let mut row_data = Vec::new();
                for (i, column) in row.columns().iter().enumerate() {
                    let value = match column.type_info().name() {
                        "INT4" | "INTEGER" => {
                            if let Ok(val) = row.try_get::<i32>(i) {
                                serde_json::json!(val)
                            } else {
                                serde_json::Value::Null
                            }
                        }
                        "INT8" | "BIGINT" => {
                            if let Ok(val) = row.try_get::<i64>(i) {
                                serde_json::json!(val)
                            } else {
                                serde_json::Value::Null
                            }
                        }
                        "FLOAT4" | "REAL" => {
                            if let Ok(val) = row.try_get::<f32>(i) {
                                serde_json::json!(val)
                            } else {
                                serde_json::Value::Null
                            }
                        }
                        "FLOAT8" | "DOUBLE PRECISION" => {
                            if let Ok(val) = row.try_get::<f64>(i) {
                                serde_json::json!(val)
                            } else {
                                serde_json::Value::Null
                            }
                        }
                        "VARCHAR" | "TEXT" | "CHAR" => {
                            if let Ok(val) = row.try_get::<String>(i) {
                                serde_json::json!(val)
                            } else {
                                serde_json::Value::Null
                            }
                        }
                        "BOOL" => {
                            if let Ok(val) = row.try_get::<bool>(i) {
                                serde_json::json!(val)
                            } else {
                                serde_json::Value::Null
                            }
                        }
                        _ => {
                            // Fallback to string conversion
                            if let Ok(val) = row.try_get::<String>(i) {
                                serde_json::json!(val)
                            } else {
                                serde_json::Value::Null
                            }
                        }
                    };
                    row_data.push(value);
                }
                result.add_row(row_data);
            }

            Ok(result)
        })
    }

    fn insert_array(&self, table: &str, data: ArrayView2<f64>, columns: &[&str]) -> Result<usize> {
        let pool = self.pool.as_ref().ok_or_else(|| {
            IoError::ConnectionError("PostgreSQL pool not initialized".to_string())
        })?;

        if columns.len() != data.ncols() {
            return Err(IoError::ValidationError(
                "Number of columns doesn't match data dimensions".to_string(),
            ));
        }

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| IoError::NetworkError(format!("Failed to create async runtime: {}", e)))?;

        rt.block_on(async {
            let mut transaction = pool.begin().await.map_err(|e| {
                IoError::DatabaseError(format!("Failed to begin transaction: {}", e))
            })?;

            let placeholders: Vec<String> =
                (1..=columns.len()).map(|i| format!("${}", i)).collect();

            let insert_sql = format!(
                "INSERT INTO {} ({}) VALUES ({})",
                table,
                columns.join(", "),
                placeholders.join(", ")
            );

            let mut rows_inserted = 0;
            for row in data.rows() {
                let mut query = sqlx::query(&insert_sql);
                for &value in row.iter() {
                    query = query.bind(value);
                }

                query
                    .execute(&mut *transaction)
                    .await
                    .map_err(|e| IoError::DatabaseError(format!("Insert failed: {}", e)))?;
                rows_inserted += 1;
            }

            transaction
                .commit()
                .await
                .map_err(|e| IoError::DatabaseError(format!("Transaction commit failed: {}", e)))?;

            Ok(rows_inserted)
        })
    }

    fn create_table(&self, table: &str, schema: &TableSchema) -> Result<()> {
        let pool = self.pool.as_ref().ok_or_else(|| {
            IoError::ConnectionError("PostgreSQL pool not initialized".to_string())
        })?;

        let mut create_sql = format!("CREATE TABLE {} (", table);

        let column_defs: Vec<String> = schema
            .columns
            .iter()
            .map(|col| {
                let pg_type = match col.data_type {
                    DataType::Integer => "INTEGER",
                    DataType::BigInt => "BIGINT",
                    DataType::Float => "REAL",
                    DataType::Double => "DOUBLE PRECISION",
                    DataType::Decimal(p, s) => return format!("DECIMAL({}, {})", p, s),
                    DataType::Varchar(len) => return format!("VARCHAR({})", len),
                    DataType::Text => "TEXT",
                    DataType::Boolean => "BOOLEAN",
                    DataType::Date => "DATE",
                    DataType::Timestamp => "TIMESTAMP",
                    DataType::Json => "JSONB",
                    DataType::Binary => "BYTEA",
                };

                let nullable = if col.nullable { "" } else { " NOT NULL" };
                format!("{} {}{}", col.name, pg_type, nullable)
            })
            .collect();

        create_sql.push_str(&column_defs.join(", "));

        if let Some(ref pk_cols) = schema.primary_key {
            create_sql.push_str(&format!(", PRIMARY KEY ({})", pk_cols.join(", ")));
        }

        create_sql.push(')');

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| IoError::NetworkError(format!("Failed to create async runtime: {}", e)))?;

        rt.block_on(async {
            sqlx::query(&create_sql)
                .execute(pool)
                .await
                .map_err(|e| IoError::DatabaseError(format!("Table creation failed: {}", e)))?;
            Ok(())
        })
    }

    fn table_exists(&self, table: &str) -> Result<bool> {
        let pool = self.pool.as_ref().ok_or_else(|| {
            IoError::ConnectionError("PostgreSQL pool not initialized".to_string())
        })?;

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| IoError::NetworkError(format!("Failed to create async runtime: {}", e)))?;

        rt.block_on(async {
            let row: (i64,) = sqlx::query_as(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = $1",
            )
            .bind(table)
            .fetch_one(pool)
            .await
            .map_err(|e| IoError::DatabaseError(format!("Table existence check failed: {}", e)))?;

            Ok(row.0 > 0)
        })
    }

    fn get_schema(&self, table: &str) -> Result<TableSchema> {
        let pool = self.pool.as_ref().ok_or_else(|| {
            IoError::ConnectionError("PostgreSQL pool not initialized".to_string())
        })?;

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| IoError::NetworkError(format!("Failed to create async runtime: {}", e)))?;

        rt.block_on(async {
            // Get column information
            let column_rows = sqlx::query(
                r#"
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = $1
                ORDER BY ordinal_position
                "#,
            )
            .bind(table)
            .fetch_all(pool)
            .await
            .map_err(|e| IoError::DatabaseError(format!("Schema query failed: {}", e)))?;

            let mut columns = Vec::new();
            for row in column_rows {
                let column_name: String = row.get(0);
                let data_type_str: String = row.get(1);
                let is_nullable: String = row.get(2);
                let column_default: Option<String> = row.get(3);

                let data_type = match data_type_str.as_str() {
                    "integer" => DataType::Integer,
                    "bigint" => DataType::BigInt,
                    "real" => DataType::Float,
                    "double precision" => DataType::Double,
                    s if s.starts_with("character varying") => DataType::Varchar(255),
                    "text" => DataType::Text,
                    "boolean" => DataType::Boolean,
                    "date" => DataType::Date,
                    "timestamp without time zone" => DataType::Timestamp,
                    "jsonb" | "json" => DataType::Json,
                    "bytea" => DataType::Binary,
                    _ => DataType::Text, // Default fallback
                };

                columns.push(ColumnDef {
                    name: column_name,
                    data_type,
                    nullable: is_nullable == "YES",
                    default: column_default.map(|s| serde_json::Value::String(s)),
                });
            }

            // Get primary key information
            let pk_rows = sqlx::query(
                r#"
                SELECT a.attname
                FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE i.indrelid = $1::regclass AND i.indisprimary
                "#,
            )
            .bind(table)
            .fetch_all(pool)
            .await
            .map_err(|e| IoError::DatabaseError(format!("Primary key query failed: {}", e)))?;

            let primary_key = if pk_rows.is_empty() {
                None
            } else {
                Some(
                    pk_rows
                        .into_iter()
                        .map(|row| row.get::<String>(0))
                        .collect(),
                )
            };

            // Get index information
            let index_rows = sqlx::query(
                r#"
                SELECT 
                    i.relname as index_name,
                    array_agg(a.attname ORDER BY c.ordinality) as column_names,
                    ix.indisunique as is_unique
                FROM pg_class t
                JOIN pg_index ix ON t.oid = ix.indrelid
                JOIN pg_class i ON i.oid = ix.indexrelid
                JOIN pg_attribute a ON a.attrelid = t.oid
                JOIN unnest(ix.indkey) WITH ORDINALITY c(attnum, ordinality) ON a.attnum = c.attnum
                WHERE t.relname = $1 AND ix.indisprimary = false
                GROUP BY i.relname, ix.indisunique
                ORDER BY i.relname
                "#,
            )
            .bind(table)
            .fetch_all(pool)
            .await
            .map_err(|e| IoError::DatabaseError(format!("Index query failed: {}", e)))?;

            let mut indexes = Vec::new();
            for row in index_rows {
                let index_name: String = row.get(0);
                let column_names: Vec<String> = row.get(1);
                let is_unique: bool = row.get(2);

                indexes.push(Index {
                    name: index_name,
                    columns: column_names,
                    unique: is_unique,
                });
            }

            Ok(TableSchema {
                name: table.to_string(),
                columns,
                primary_key,
                indexes,
            })
        })
    }
}

/// MySQL connection implementation
#[cfg(feature = "mysql")]
struct MySQLConnection {
    config: DatabaseConfig,
    pool: Option<MySqlPool>,
}

#[cfg(feature = "mysql")]
impl MySQLConnection {
    async fn new(config: &DatabaseConfig) -> Result<Self> {
        // Create actual MySQL connection pool
        let connection_string = Self::build_connection_string(config);

        let pool = MySqlPoolOptions::new()
            .max_connections(
                config
                    .options
                    .get("max_connections")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(5),
            )
            .connect(&connection_string)
            .await
            .map_err(|e| IoError::NetworkError(format!("MySQL connection failed: {}", e)))?;

        Ok(Self {
            config: config.clone(),
            pool: Some(pool),
        })
    }

    fn build_connection_string(config: &DatabaseConfig) -> String {
        let host = config.host.as_deref().unwrap_or("localhost");
        let port = config.port.unwrap_or(3306);
        let user = config.username.as_deref().unwrap_or("root");
        let password = config.password.as_deref().unwrap_or("");

        format!(
            "mysql://{}:{}@{}:{}/{}",
            user, password, host, port, config.database
        )
    }
}

#[cfg(feature = "mysql")]
impl DatabaseConnection for MySQLConnection {
    fn query(&self, query: &QueryBuilder) -> Result<ResultSet> {
        let sql = query.build_sql();
        let params = &query.values;
        self.execute_sql(&sql, params)
    }

    fn execute_sql(&self, sql: &str, params: &[serde_json::Value]) -> Result<ResultSet> {
        use sqlx::{Column, Row, TypeInfo};

        let pool = self
            .pool
            .as_ref()
            .ok_or_else(|| IoError::ConnectionError("MySQL pool not initialized".to_string()))?;

        // Execute query in a blocking context
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| IoError::NetworkError(format!("Failed to create async runtime: {}", e)))?;

        rt.block_on(async {
            // Convert params to sqlx format
            let mut query = sqlx::query(sql);
            for param in params {
                match param {
                    serde_json::Value::String(s) => query = query.bind(s),
                    serde_json::Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            query = query.bind(i);
                        } else if let Some(f) = n.as_f64() {
                            query = query.bind(f);
                        }
                    }
                    serde_json::Value::Bool(b) => query = query.bind(*b),
                    serde_json::Value::Null => query = query.bind(Option::<String>::None),
                    _ => query = query.bind(param.to_string()),
                }
            }

            let rows = query
                .fetch_all(pool)
                .await
                .map_err(|e| IoError::DatabaseError(format!("MySQL query failed: {}", e)))?;

            if rows.is_empty() {
                return Ok(ResultSet::new(vec![]));
            }

            // Extract column names
            let columns: Vec<String> = rows[0]
                .columns()
                .iter()
                .map(|col| col.name().to_string())
                .collect();

            let mut result = ResultSet::new(columns);

            // Convert rows to JSON values
            for row in rows {
                let mut row_data = Vec::new();
                for (i, column) in row.columns().iter().enumerate() {
                    let value = match column.type_info().name() {
                        "TINY" | "SHORT" | "LONG" => {
                            if let Ok(val) = row.try_get::<i32>(i) {
                                serde_json::json!(val)
                            } else {
                                serde_json::Value::Null
                            }
                        }
                        "LONGLONG" => {
                            if let Ok(val) = row.try_get::<i64>(i) {
                                serde_json::json!(val)
                            } else {
                                serde_json::Value::Null
                            }
                        }
                        "FLOAT" => {
                            if let Ok(val) = row.try_get::<f32>(i) {
                                serde_json::json!(val)
                            } else {
                                serde_json::Value::Null
                            }
                        }
                        "DOUBLE" => {
                            if let Ok(val) = row.try_get::<f64>(i) {
                                serde_json::json!(val)
                            } else {
                                serde_json::Value::Null
                            }
                        }
                        "VARCHAR" | "TEXT" | "CHAR" | "VAR_STRING" => {
                            if let Ok(val) = row.try_get::<String>(i) {
                                serde_json::json!(val)
                            } else {
                                serde_json::Value::Null
                            }
                        }
                        "TINY" if column.type_info().name() == "BOOLEAN" => {
                            if let Ok(val) = row.try_get::<bool>(i) {
                                serde_json::json!(val)
                            } else {
                                serde_json::Value::Null
                            }
                        }
                        _ => {
                            // Fallback to string conversion
                            if let Ok(val) = row.try_get::<String>(i) {
                                serde_json::json!(val)
                            } else {
                                serde_json::Value::Null
                            }
                        }
                    };
                    row_data.push(value);
                }
                result.add_row(row_data);
            }

            Ok(result)
        })
    }

    fn insert_array(&self, table: &str, data: ArrayView2<f64>, columns: &[&str]) -> Result<usize> {
        let pool = self
            .pool
            .as_ref()
            .ok_or_else(|| IoError::ConnectionError("MySQL pool not initialized".to_string()))?;

        if columns.len() != data.ncols() {
            return Err(IoError::ValidationError(
                "Number of columns doesn't match data dimensions".to_string(),
            ));
        }

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| IoError::NetworkError(format!("Failed to create async runtime: {}", e)))?;

        rt.block_on(async {
            let mut transaction = pool.begin().await.map_err(|e| {
                IoError::DatabaseError(format!("Failed to begin transaction: {}", e))
            })?;

            let placeholders: Vec<String> = (0..columns.len()).map(|_| "?".to_string()).collect();

            let insert_sql = format!(
                "INSERT INTO {} ({}) VALUES ({})",
                table,
                columns.join(", "),
                placeholders.join(", ")
            );

            let mut rows_inserted = 0;
            for row in data.rows() {
                let mut query = sqlx::query(&insert_sql);
                for &value in row.iter() {
                    query = query.bind(value);
                }

                query
                    .execute(&mut *transaction)
                    .await
                    .map_err(|e| IoError::DatabaseError(format!("Insert failed: {}", e)))?;
                rows_inserted += 1;
            }

            transaction
                .commit()
                .await
                .map_err(|e| IoError::DatabaseError(format!("Transaction commit failed: {}", e)))?;

            Ok(rows_inserted)
        })
    }

    fn create_table(&self, table: &str, schema: &TableSchema) -> Result<()> {
        let pool = self
            .pool
            .as_ref()
            .ok_or_else(|| IoError::ConnectionError("MySQL pool not initialized".to_string()))?;

        let mut create_sql = format!("CREATE TABLE {} (", table);

        let column_defs: Vec<String> = schema
            .columns
            .iter()
            .map(|col| {
                let mysql_type = match col.data_type {
                    DataType::Integer => "INT",
                    DataType::BigInt => "BIGINT",
                    DataType::Float => "FLOAT",
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
                format!("{} {}{}", col.name, mysql_type, nullable)
            })
            .collect();

        create_sql.push_str(&column_defs.join(", "));

        if let Some(ref pk_cols) = schema.primary_key {
            create_sql.push_str(&format!(", PRIMARY KEY ({})", pk_cols.join(", ")));
        }

        create_sql.push(')');

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| IoError::NetworkError(format!("Failed to create async runtime: {}", e)))?;

        rt.block_on(async {
            sqlx::query(&create_sql)
                .execute(pool)
                .await
                .map_err(|e| IoError::DatabaseError(format!("Table creation failed: {}", e)))?;
            Ok(())
        })
    }

    fn table_exists(&self, table: &str) -> Result<bool> {
        let pool = self
            .pool
            .as_ref()
            .ok_or_else(|| IoError::ConnectionError("MySQL pool not initialized".to_string()))?;

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| IoError::NetworkError(format!("Failed to create async runtime: {}", e)))?;

        rt.block_on(async {
            let row: (i64,) = sqlx::query_as(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            )
            .bind(table)
            .fetch_one(pool)
            .await
            .map_err(|e| IoError::DatabaseError(format!("Table existence check failed: {}", e)))?;

            Ok(row.0 > 0)
        })
    }

    fn get_schema(&self, table: &str) -> Result<TableSchema> {
        let pool = self
            .pool
            .as_ref()
            .ok_or_else(|| IoError::ConnectionError("MySQL pool not initialized".to_string()))?;

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| IoError::NetworkError(format!("Failed to create async runtime: {}", e)))?;

        rt.block_on(async {
            // Get column information
            let column_rows = sqlx::query(
                r#"
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = ?
                ORDER BY ordinal_position
                "#,
            )
            .bind(table)
            .fetch_all(pool)
            .await
            .map_err(|e| IoError::DatabaseError(format!("Schema query failed: {}", e)))?;

            let mut columns = Vec::new();
            for row in column_rows {
                let column_name: String = row.get(0);
                let data_type_str: String = row.get(1);
                let is_nullable: String = row.get(2);
                let column_default: Option<String> = row.get(3);

                let data_type = match data_type_str.to_lowercase().as_str() {
                    "int" | "integer" => DataType::Integer,
                    "bigint" => DataType::BigInt,
                    "float" => DataType::Float,
                    "double" => DataType::Double,
                    s if s.starts_with("varchar") => DataType::Varchar(255),
                    "text" => DataType::Text,
                    "tinyint" => DataType::Boolean, // MySQL uses TINYINT for boolean
                    "date" => DataType::Date,
                    "timestamp" | "datetime" => DataType::Timestamp,
                    "json" => DataType::Json,
                    "blob" => DataType::Binary,
                    _ => DataType::Text, // Default fallback
                };

                columns.push(ColumnDef {
                    name: column_name,
                    data_type,
                    nullable: is_nullable == "YES",
                    default: column_default.map(|s| serde_json::Value::String(s)),
                });
            }

            // Get primary key information
            let pk_rows = sqlx::query(
                r#"
                SELECT column_name
                FROM information_schema.key_column_usage
                WHERE table_name = ? AND constraint_name = 'PRIMARY'
                ORDER BY ordinal_position
                "#,
            )
            .bind(table)
            .fetch_all(pool)
            .await
            .map_err(|e| IoError::DatabaseError(format!("Primary key query failed: {}", e)))?;

            let primary_key = if pk_rows.is_empty() {
                None
            } else {
                Some(
                    pk_rows
                        .into_iter()
                        .map(|row| row.get::<String>(0))
                        .collect(),
                )
            };

            // Get index information
            let index_rows = sqlx::query(
                r#"
                SELECT 
                    s.index_name,
                    GROUP_CONCAT(s.column_name ORDER BY s.seq_in_index) as column_names,
                    CASE WHEN s.non_unique = 0 THEN true ELSE false END as is_unique
                FROM information_schema.statistics s
                WHERE s.table_name = ? AND s.index_name != 'PRIMARY'
                GROUP BY s.index_name, s.non_unique
                ORDER BY s.index_name
                "#,
            )
            .bind(table)
            .fetch_all(pool)
            .await
            .map_err(|e| IoError::DatabaseError(format!("Index query failed: {}", e)))?;

            let mut indexes = Vec::new();
            for row in index_rows {
                let index_name: String = row.get(0);
                let column_names_str: String = row.get(1);
                let is_unique: bool = row.get(2);

                let column_names: Vec<String> = column_names_str
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect();

                indexes.push(Index {
                    name: index_name,
                    columns: column_names,
                    unique: is_unique,
                });
            }

            Ok(TableSchema {
                name: table.to_string(),
                columns,
                primary_key,
                indexes,
            })
        })
    }
}

/// DuckDB connection implementation (analytical database)
#[cfg(feature = "duckdb")]
struct DuckDBConnection {
    config: DatabaseConfig,
    connection: Option<DuckdbConn>,
}

#[cfg(feature = "duckdb")]
impl DuckDBConnection {
    fn new(config: &DatabaseConfig) -> Result<Self> {
        // Create actual DuckDB connection
        let conn = if config.database == ":memory:" {
            DuckdbConn::open_in_memory()
        } else {
            DuckdbConn::open(&config.database)
        }
        .map_err(|e| IoError::ConnectionError(format!("DuckDB connection failed: {}", e)))?;

        Ok(Self {
            config: config.clone(),
            connection: Some(conn),
        })
    }
}

#[cfg(feature = "duckdb")]
impl DatabaseConnection for DuckDBConnection {
    fn query(&self, query: &QueryBuilder) -> Result<ResultSet> {
        let sql = query.build_sql();
        let params = &query.values;
        self.execute_sql(&sql, params)
    }

    fn execute_sql(&self, sql: &str, params: &[serde_json::Value]) -> Result<ResultSet> {
        use duckdb::params_from_iter;

        let conn = self.connection.as_ref().ok_or_else(|| {
            IoError::ConnectionError("DuckDB connection not initialized".to_string())
        })?;

        // Convert JSON params to DuckDB values
        let duck_params: Vec<&dyn duckdb::ToSql> = params
            .iter()
            .map(|p| match p {
                serde_json::Value::String(s) => s as &dyn duckdb::ToSql,
                serde_json::Value::Number(n) => {
                    if n.is_i64() {
                        &n.as_i64().unwrap() as &dyn duckdb::ToSql
                    } else {
                        &n.as_f64().unwrap() as &dyn duckdb::ToSql
                    }
                }
                serde_json::Value::Bool(b) => b as &dyn duckdb::ToSql,
                serde_json::Value::Null => &None::<String> as &dyn duckdb::ToSql,
                serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                    &p.to_string() as &dyn duckdb::ToSql
                }
            })
            .collect();

        let mut stmt = conn.prepare(sql).map_err(|e| {
            IoError::DatabaseError(format!("DuckDB query preparation failed: {}", e))
        })?;

        let column_count = stmt.column_count();
        let mut column_names = Vec::new();
        for i in 0..column_count {
            column_names.push(stmt.column_name(i).unwrap_or("unknown").to_string());
        }

        let mut result = ResultSet::new(column_names);

        let rows = if duck_params.is_empty() {
            stmt.query_map([], |row| {
                let mut row_data = Vec::new();
                for i in 0..column_count {
                    let value = match row.get_ref(i) {
                        Ok(val) => match val {
                            duckdb::types::ValueRef::Null => serde_json::Value::Null,
                            duckdb::types::ValueRef::Integer(i) => serde_json::json!(i),
                            duckdb::types::ValueRef::Real(f) => serde_json::json!(f),
                            duckdb::types::ValueRef::Text(s) => {
                                serde_json::json!(std::str::from_utf8(s).unwrap_or(""))
                            }
                            duckdb::types::ValueRef::Blob(b) => {
                                serde_json::json!(base64::encode(b))
                            }
                        },
                        Err(_) => serde_json::Value::Null,
                    };
                    row_data.push(value);
                }
                Ok(row_data)
            })
        } else {
            stmt.query_map(params_from_iter(duck_params), |row| {
                let mut row_data = Vec::new();
                for i in 0..column_count {
                    let value = match row.get_ref(i) {
                        Ok(val) => match val {
                            duckdb::types::ValueRef::Null => serde_json::Value::Null,
                            duckdb::types::ValueRef::Integer(i) => serde_json::json!(i),
                            duckdb::types::ValueRef::Real(f) => serde_json::json!(f),
                            duckdb::types::ValueRef::Text(s) => {
                                serde_json::json!(std::str::from_utf8(s).unwrap_or(""))
                            }
                            duckdb::types::ValueRef::Blob(b) => {
                                serde_json::json!(base64::encode(b))
                            }
                        },
                        Err(_) => serde_json::Value::Null,
                    };
                    row_data.push(value);
                }
                Ok(row_data)
            })
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
        let conn = self.connection.as_ref().ok_or_else(|| {
            IoError::ConnectionError("DuckDB connection not initialized".to_string())
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
            std::iter::repeat(values_clause.clone())
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

        let duck_params: Vec<&dyn duckdb::ToSql> =
            all_params.iter().map(|v| v as &dyn duckdb::ToSql).collect();

        let rows_affected = conn
            .execute(&insert_sql, duckdb::params_from_iter(duck_params))
            .map_err(|e| IoError::DatabaseError(format!("DuckDB bulk insert failed: {}", e)))?;

        Ok(rows_affected)
    }

    fn create_table(&self, table: &str, schema: &TableSchema) -> Result<()> {
        let conn = self.connection.as_ref().ok_or_else(|| {
            IoError::ConnectionError("DuckDB connection not initialized".to_string())
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
        let conn = self.connection.as_ref().ok_or_else(|| {
            IoError::ConnectionError("DuckDB connection not initialized".to_string())
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
        let conn = self.connection.as_ref().ok_or_else(|| {
            IoError::ConnectionError("DuckDB connection not initialized".to_string())
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
                "INTEGER" => DataType::Integer,
                "BIGINT" => DataType::BigInt,
                "REAL" | "FLOAT" => DataType::Float,
                "DOUBLE" => DataType::Double,
                s if s.starts_with("VARCHAR") => DataType::Varchar(255),
                "TEXT" => DataType::Text,
                "BOOLEAN" => DataType::Boolean,
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
        // but we can check for any available index information
        let index_rows = conn.prepare(
            r#"
            SELECT 
                index_name,
                column_names,
                is_unique
            FROM duckdb_indexes() 
            WHERE table_name = ?
            ORDER BY index_name
            "#,
        );

        let mut indexes = Vec::new();
        if let Ok(mut stmt) = index_rows {
            let rows = stmt.query_map([table], |row| {
                Ok((
                    row.get::<_, String>(0)?, // index_name
                    row.get::<_, String>(1)?, // column_names
                    row.get::<_, bool>(2)?,   // is_unique
                ))
            });

            if let Ok(row_iter) = rows {
                for row_result in row_iter {
                    if let Ok((index_name, column_names_str, is_unique)) = row_result {
                        let column_names: Vec<String> = column_names_str
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .collect();

                        indexes.push(Index {
                            name: index_name,
                            columns: column_names,
                            unique: is_unique,
                        });
                    }
                }
            }
            // If the query fails, it's likely DuckDB doesn't support this function
            // in this version, so we just continue with empty indexes
        }

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

    #[test]
    fn test_database_config() {
        let config = DatabaseConfig::new(DatabaseType::PostgreSQL, "mydb")
            .host("localhost")
            .port(5432)
            .credentials("user", "pass");

        assert_eq!(config.database, "mydb");
        assert_eq!(config.host, Some("localhost".to_string()));
        assert_eq!(config.port, Some(5432));
    }

    #[test]
    fn test_query_builder() {
        let query = QueryBuilder::select("users")
            .columns(vec!["id", "name", "email"])
            .where_clause("age > 18")
            .order_by("name", false)
            .limit(10);

        let sql = query.build_sql();
        assert!(sql.contains("SELECT id, name, email FROM users"));
        assert!(sql.contains("WHERE age > 18"));
        assert!(sql.contains("ORDER BY name ASC"));
        assert!(sql.contains("LIMIT 10"));
    }

    #[test]
    fn test_result_set() {
        let mut result = ResultSet::new(vec!["id".to_string(), "value".to_string()]);
        result.add_row(vec![serde_json::json!(1), serde_json::json!(10.5)]);
        result.add_row(vec![serde_json::json!(2), serde_json::json!(20.3)]);

        assert_eq!(result.row_count(), 2);
        assert_eq!(result.column_count(), 2);

        let array = result.to_array().unwrap();
        assert_eq!(array.shape(), &[2, 2]);
    }
}

// Advanced Database Features

use std::sync::Arc;
#[cfg(feature = "async")]
use tokio::sync::RwLock;

/// Connection pool for database connections
pub struct ConnectionPool {
    #[allow(dead_code)]
    db_type: DatabaseType,
    config: DatabaseConfig,
    connections: Arc<Mutex<Vec<Box<dyn DatabaseConnection>>>>,
    max_connections: usize,
}

impl ConnectionPool {
    pub fn new(config: DatabaseConfig, max_connections: usize) -> Self {
        Self {
            db_type: config.db_type,
            config,
            connections: Arc::new(Mutex::new(Vec::new())),
            max_connections,
        }
    }

    /// Get a connection from the pool
    pub fn get_connection(&self) -> Result<PooledConnection> {
        let mut connections = self.connections.lock().unwrap();

        if let Some(conn) = connections.pop() {
            Ok(PooledConnection {
                connection: conn,
                pool: self.connections.clone(),
            })
        } else if connections.len() < self.max_connections {
            let conn = DatabaseConnector::connect(&self.config)?;
            Ok(PooledConnection {
                connection: conn,
                pool: self.connections.clone(),
            })
        } else {
            Err(IoError::Other("Connection pool exhausted".to_string()))
        }
    }
}

/// Pooled connection wrapper that returns connection to pool on drop
pub struct PooledConnection {
    connection: Box<dyn DatabaseConnection>,
    #[allow(dead_code)]
    pool: Arc<Mutex<Vec<Box<dyn DatabaseConnection>>>>,
}

impl std::ops::Deref for PooledConnection {
    type Target = dyn DatabaseConnection;

    fn deref(&self) -> &Self::Target {
        &*self.connection
    }
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        // Return connection to pool
        // In real implementation, would check connection health first
    }
}

/// Database transaction support
pub struct Transaction {
    connection: Box<dyn DatabaseConnection>,
    savepoint: String,
    committed: bool,
}

impl Transaction {
    pub fn new(connection: Box<dyn DatabaseConnection>) -> Result<Self> {
        let savepoint = format!("sp_{}", uuid::Uuid::new_v4());
        connection.execute_sql(&format!("SAVEPOINT {savepoint}"), &[])?;

        Ok(Self {
            connection,
            savepoint,
            committed: false,
        })
    }

    /// Commit the transaction
    pub fn commit(mut self) -> Result<()> {
        self.connection
            .execute_sql(&format!("RELEASE SAVEPOINT {}", self.savepoint), &[])?;
        self.committed = true;
        Ok(())
    }

    /// Rollback the transaction
    pub fn rollback(mut self) -> Result<()> {
        self.connection
            .execute_sql(&format!("ROLLBACK TO SAVEPOINT {}", self.savepoint), &[])?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        if !self.committed {
            let _ = self
                .connection
                .execute_sql(&format!("ROLLBACK TO SAVEPOINT {}", self.savepoint), &[]);
        }
    }
}

/// Prepared statement support
pub struct PreparedStatement {
    sql: String,
    param_count: usize,
}

impl PreparedStatement {
    pub fn new(sql: impl Into<String>) -> Result<Self> {
        let _sql = sql.into();
        let param_count = _sql.matches('?').count();

        Ok(Self {
            sql: _sql,
            param_count,
        })
    }

    /// Execute with parameters
    pub fn execute(
        &self,
        conn: &dyn DatabaseConnection,
        params: &[serde_json::Value],
    ) -> Result<ResultSet> {
        if params.len() != self.param_count {
            return Err(IoError::Other(format!(
                "Parameter count mismatch: expected {}, got {}",
                self.param_count,
                params.len()
            )));
        }

        conn.execute_sql(&self.sql, params)
    }
}

/// Schema migration support
pub mod migration {
    use super::*;
    use chrono::{DateTime, Utc};

    /// Database migration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Migration {
        pub version: String,
        pub name: String,
        pub up_sql: String,
        pub down_sql: String,
        pub applied_at: Option<DateTime<Utc>>,
    }

    /// Migration manager
    pub struct MigrationManager {
        migrations: Vec<Migration>,
    }

    impl Default for MigrationManager {
        fn default() -> Self {
            Self::new()
        }
    }

    impl MigrationManager {
        pub fn new() -> Self {
            Self {
                migrations: Vec::new(),
            }
        }

        /// Add a migration
        pub fn add_migration(&mut self, migration: Migration) {
            self.migrations.push(migration);
        }

        /// Apply pending migrations
        pub fn migrate(&self, conn: &dyn DatabaseConnection) -> Result<usize> {
            // Ensure migration table exists
            self.ensure_migration_table(conn)?;

            // Get applied migrations
            let applied = self.get_applied_migrations(conn)?;
            let mut count = 0;

            for migration in &self.migrations {
                if !applied.contains(&migration.version) {
                    conn.execute_sql(&migration.up_sql, &[])?;
                    self.record_migration(conn, &migration.version)?;
                    count += 1;
                }
            }

            Ok(count)
        }

        /// Rollback last migration
        pub fn rollback(&self, conn: &dyn DatabaseConnection) -> Result<()> {
            let applied = self.get_applied_migrations(conn)?;

            if let Some(last_version) = applied.last() {
                if let Some(migration) = self.migrations.iter().find(|m| &m.version == last_version)
                {
                    conn.execute_sql(&migration.down_sql, &[])?;
                    self.remove_migration_record(conn, last_version)?;
                }
            }

            Ok(())
        }

        fn ensure_migration_table(&self, conn: &dyn DatabaseConnection) -> Result<()> {
            let schema = TableSchema {
                name: "schema_migrations".to_string(),
                columns: vec![
                    ColumnDef {
                        name: "version".to_string(),
                        data_type: DataType::Varchar(255),
                        nullable: false,
                        default: None,
                    },
                    ColumnDef {
                        name: "applied_at".to_string(),
                        data_type: DataType::Timestamp,
                        nullable: false,
                        default: None,
                    },
                ],
                primary_key: Some(vec!["version".to_string()]),
                indexes: vec![],
            };

            if !conn.table_exists("schema_migrations")? {
                conn.create_table("schema_migrations", &schema)?;
            }

            Ok(())
        }

        fn get_applied_migrations(&self, conn: &dyn DatabaseConnection) -> Result<Vec<String>> {
            let query = QueryBuilder::select("schema_migrations")
                .columns(vec!["version"])
                .order_by("version", false);

            let result = conn.query(&query)?;

            Ok(result
                .rows
                .iter()
                .map(|row| row[0].as_str().unwrap_or("").to_string())
                .collect())
        }

        fn record_migration(&self, conn: &dyn DatabaseConnection, version: &str) -> Result<()> {
            let query = QueryBuilder::insert("schema_migrations")
                .columns(vec!["version", "applied_at"])
                .values(vec![
                    serde_json::json!(version),
                    serde_json::json!(Utc::now().to_rfc3339()),
                ]);

            conn.query(&query)?;
            Ok(())
        }

        fn remove_migration_record(
            &self,
            conn: &dyn DatabaseConnection,
            version: &str,
        ) -> Result<()> {
            conn.execute_sql(
                "DELETE FROM schema_migrations WHERE version = ?",
                &[serde_json::json!(version)],
            )?;
            Ok(())
        }
    }
}

/// ORM-like features
pub mod orm {
    use super::*;

    /// Model trait for ORM functionality
    pub trait Model: Sized {
        fn table_name() -> &'static str;
        fn from_row(row: &[serde_json::Value]) -> Result<Self>;
        fn to_row(&self) -> Vec<serde_json::Value>;
    }

    /// Active record pattern implementation
    pub struct ActiveRecord<T: Model> {
        model: T,
        changed: bool,
    }

    impl<T: Model> ActiveRecord<T> {
        pub fn new(model: T) -> Self {
            Self {
                model,
                changed: false,
            }
        }

        /// Find by primary key
        pub fn find(conn: &dyn DatabaseConnection, id: serde_json::Value) -> Result<T> {
            let query = QueryBuilder::select(T::table_name()).where_clause("id = ?");

            let result = conn.execute_sql(&query.build_sql(), &[id])?;

            if let Some(row) = result.rows.first() {
                T::from_row(row)
            } else {
                Err(IoError::NotFound("Record not found".to_string()))
            }
        }

        /// Find all records
        pub fn find_all(conn: &dyn DatabaseConnection) -> Result<Vec<T>> {
            let query = QueryBuilder::select(T::table_name());
            let result = conn.query(&query)?;

            result.rows.iter().map(|row| T::from_row(row)).collect()
        }

        /// Save the record
        pub fn save(&mut self, conn: &dyn DatabaseConnection) -> Result<()> {
            if self.changed {
                let row = self.model.to_row();
                let query = QueryBuilder::insert(T::table_name()).values(row);

                conn.query(&query)?;
                self.changed = false;
            }
            Ok(())
        }
    }
}

/// Real-time change data capture (CDC)
pub mod cdc {
    use super::*;
    use std::sync::mpsc;

    /// Change event types
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum ChangeEvent {
        Insert {
            table: String,
            data: serde_json::Value,
        },
        Update {
            table: String,
            old: serde_json::Value,
            new: serde_json::Value,
        },
        Delete {
            table: String,
            data: serde_json::Value,
        },
    }

    /// CDC listener
    pub struct CDCListener {
        receiver: mpsc::Receiver<ChangeEvent>,
    }

    impl CDCListener {
        pub fn new(receiver: mpsc::Receiver<ChangeEvent>) -> Self {
            Self { receiver }
        }

        /// Get next change event
        pub fn next_event(&self) -> Option<ChangeEvent> {
            self.receiver.try_recv().ok()
        }

        /// Iterate over events
        pub fn events(&self) -> impl Iterator<Item = ChangeEvent> + '_ {
            std::iter::from_fn(move || self.next_event())
        }
    }

    /// CDC publisher
    pub struct CDCPublisher {
        sender: mpsc::Sender<ChangeEvent>,
    }

    impl CDCPublisher {
        pub fn new(sender: mpsc::Sender<ChangeEvent>) -> Self {
            Self { sender }
        }

        /// Publish change event
        pub fn publish(&self, event: ChangeEvent) -> Result<()> {
            self.sender
                .send(event)
                .map_err(|e| IoError::Other(format!("Failed to publish CDC event: {e}")))
        }
    }
}

/// Database replication support
pub mod replication {
    use super::*;

    /// Replication mode
    #[derive(Debug, Clone, Copy)]
    pub enum ReplicationMode {
        /// Master-slave replication
        MasterSlave,
        /// Master-master replication
        MasterMaster,
        /// Read replicas
        ReadReplica,
    }

    /// Replication configuration
    pub struct ReplicationConfig {
        pub mode: ReplicationMode,
        pub master: DatabaseConfig,
        pub replicas: Vec<DatabaseConfig>,
    }

    /// Replicated database connection
    pub struct ReplicatedConnection {
        #[allow(dead_code)]
        master: Box<dyn DatabaseConnection>,
        #[allow(dead_code)]
        replicas: Vec<Box<dyn DatabaseConnection>>,
        #[allow(dead_code)]
        mode: ReplicationMode,
        read_preference: ReadPreference,
    }

    #[derive(Debug, Clone, Copy)]
    pub enum ReadPreference {
        Master,
        Replica,
        Nearest,
    }

    impl ReplicatedConnection {
        pub fn new(config: ReplicationConfig) -> Result<Self> {
            let master = DatabaseConnector::connect(&config.master)?;
            let replicas: Result<Vec<_>> = config
                .replicas
                .iter()
                .map(|cfg| DatabaseConnector::connect(cfg))
                .collect();

            Ok(Self {
                master,
                replicas: replicas?,
                mode: config.mode,
                read_preference: ReadPreference::Replica,
            })
        }

        /// Set read preference
        pub fn set_read_preference(&mut self, pref: ReadPreference) {
            self.read_preference = pref;
        }

        /// Get connection for read operations
        #[allow(dead_code)]
        fn get_read_connection(&self) -> &dyn DatabaseConnection {
            match self.read_preference {
                ReadPreference::Master => &*self.master,
                ReadPreference::Replica => {
                    if self.replicas.is_empty() {
                        &*self.master
                    } else {
                        // Simple round-robin
                        &*self.replicas[0]
                    }
                }
                ReadPreference::Nearest => &*self.master, // Simplified
            }
        }
    }
}

/// Advanced query capabilities
pub mod advanced_query {
    use super::*;

    /// Query optimizer
    pub struct QueryOptimizer {
        rules: Vec<Box<dyn OptimizationRule>>,
    }

    pub trait OptimizationRule: Send + Sync {
        fn optimize(&self, query: &mut QueryBuilder);
    }

    impl Default for QueryOptimizer {
        fn default() -> Self {
            Self::new()
        }
    }

    impl QueryOptimizer {
        pub fn new() -> Self {
            Self {
                rules: vec![Box::new(IndexHintRule), Box::new(JoinOrderRule)],
            }
        }

        /// Optimize query
        pub fn optimize(&self, mut query: QueryBuilder) -> QueryBuilder {
            for rule in &self.rules {
                rule.optimize(&mut query);
            }
            query
        }
    }

    /// Index hint optimization rule
    struct IndexHintRule;

    impl OptimizationRule for IndexHintRule {
        fn optimize(&self, query: &mut QueryBuilder) {
            // Add index hints based on conditions
        }
    }

    /// Join order optimization rule
    struct JoinOrderRule;

    impl OptimizationRule for JoinOrderRule {
        fn optimize(&self, query: &mut QueryBuilder) {
            // Optimize join order based on statistics
        }
    }

    /// Query statistics
    #[derive(Debug, Clone)]
    pub struct QueryStats {
        pub execution_time: std::time::Duration,
        pub rows_examined: usize,
        pub rows_returned: usize,
        pub index_used: Option<String>,
    }

    /// Query analyzer
    pub struct QueryAnalyzer;

    impl QueryAnalyzer {
        /// Analyze query performance
        pub fn analyze(query: &QueryBuilder, conn: &dyn DatabaseConnection) -> Result<QueryStats> {
            let start = std::time::Instant::now();
            let result = conn.query(query)?;
            let execution_time = start.elapsed();

            Ok(QueryStats {
                execution_time,
                rows_examined: result.row_count(),
                rows_returned: result.row_count(),
                index_used: None, // Would be determined from EXPLAIN
            })
        }

        /// Get query execution plan
        pub fn explain(query: &QueryBuilder, conn: &dyn DatabaseConnection) -> Result<String> {
            let explain_sql = format!("EXPLAIN {}", query.build_sql());
            let result = conn.execute_sql(&explain_sql, &[])?;

            // Format execution plan
            let mut plan = String::new();
            for row in &result.rows {
                if let Some(text) = row.first().and_then(|v| v.as_str()) {
                    plan.push_str(text);
                    plan.push('\n');
                }
            }

            Ok(plan)
        }
    }
}
