//! Database connectivity for scientific data
//!
//! Provides interfaces for reading and writing scientific data to various
//! database systems, including SQL and NoSQL databases.

#![allow(dead_code)]
#![allow(missing_docs)]
#![allow(clippy::too_many_arguments)]

use crate::error::{IoError, Result};
use crate::metadata::Metadata;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export database implementations
#[cfg(feature = "sqlite")]
pub mod sqlite;

#[cfg(feature = "postgres")]
pub mod postgres;

#[cfg(feature = "mysql")]
pub mod mysql;

#[cfg(feature = "duckdb")]
pub mod duckdb;

// Connection pooling
pub mod pool;

// Specialized modules
pub mod bulk;
pub mod timeseries;

// Re-export commonly used types
pub use self::pool::ConnectionPool;

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
    pub(crate) query_type: QueryType,
    pub(crate) table: String,
    pub(crate) columns: Vec<String>,
    pub(crate) conditions: Vec<String>,
    pub(crate) values: Vec<serde_json::Value>,
    pub(crate) order_by: Option<String>,
    pub(crate) limit: Option<usize>,
    pub(crate) offset: Option<usize>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) enum QueryType {
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
            DatabaseType::SQLite => Ok(Box::new(sqlite::SQLiteConnection::new(config)?)),
            #[cfg(not(feature = "sqlite"))]
            DatabaseType::SQLite => Err(IoError::UnsupportedFormat(
                "SQLite support not enabled. Enable 'sqlite' feature.".to_string(),
            )),

            #[cfg(feature = "postgres")]
            DatabaseType::PostgreSQL => Ok(Box::new(postgres::PostgreSQLConnection::new(config)?)),
            #[cfg(not(feature = "postgres"))]
            DatabaseType::PostgreSQL => Err(IoError::UnsupportedFormat(
                "PostgreSQL support not enabled. Enable 'postgres' feature.".to_string(),
            )),

            #[cfg(feature = "mysql")]
            DatabaseType::MySQL => Ok(Box::new(mysql::MySQLConnection::new(config)?)),
            #[cfg(not(feature = "mysql"))]
            DatabaseType::MySQL => Err(IoError::UnsupportedFormat(
                "MySQL support not enabled. Enable 'mysql' feature.".to_string(),
            )),

            #[cfg(feature = "duckdb")]
            DatabaseType::DuckDB => Ok(Box::new(duckdb::DuckDBConnection::new(config)?)),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_config() {
        let config = DatabaseConfig::new(DatabaseType::SQLite, "test.db");
        assert_eq!(config.db_type, DatabaseType::SQLite);
        assert_eq!(config.database, "test.db");
        assert_eq!(config.connection_string(), "sqlite://test.db");
    }

    #[test]
    fn test_query_builder() {
        let query = QueryBuilder::select("users")
            .columns(vec!["id", "name", "email"])
            .where_clause("age > 21")
            .limit(10);

        let sql = query.build_sql();
        assert!(sql.contains("SELECT id, name, email FROM users"));
        assert!(sql.contains("WHERE age > 21"));
        assert!(sql.contains("LIMIT 10"));
    }
}
