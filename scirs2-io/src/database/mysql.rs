//! MySQL database implementation
//!
//! This module is temporarily stubbed for refactoring.
//! Full async implementation will be added back.

use crate::database::{DatabaseConfig, DatabaseConnection};
use crate::error::{IoError, Result};

/// MySQL connection wrapper (stub)
pub struct MySQLConnection {
    #[allow(dead_code)]
    config: DatabaseConfig,
}

impl MySQLConnection {
    /// Create a new MySQL connection (stub)
    pub fn new(config: &DatabaseConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
}

// Stub implementation - will be properly implemented
impl DatabaseConnection for MySQLConnection {
    fn query(&self, _query: &crate::database::QueryBuilder) -> Result<crate::database::ResultSet> {
        Err(IoError::UnsupportedFormat(
            "MySQL implementation pending".to_string(),
        ))
    }

    fn execute_sql(
        &self,
        _sql: &str,
        _params: &[serde_json::Value],
    ) -> Result<crate::database::ResultSet> {
        Err(IoError::UnsupportedFormat(
            "MySQL implementation pending".to_string(),
        ))
    }

    fn insert_array(
        &self,
        _table: &str,
        _data: scirs2_core::ndarray::ArrayView2<f64>,
        _columns: &[&str],
    ) -> Result<usize> {
        Err(IoError::UnsupportedFormat(
            "MySQL implementation pending".to_string(),
        ))
    }

    fn create_table(&self, _table: &str, _schema: &crate::database::TableSchema) -> Result<()> {
        Err(IoError::UnsupportedFormat(
            "MySQL implementation pending".to_string(),
        ))
    }

    fn table_exists(&self, _table: &str) -> Result<bool> {
        Err(IoError::UnsupportedFormat(
            "MySQL implementation pending".to_string(),
        ))
    }

    fn get_schema(&self, _table: &str) -> Result<crate::database::TableSchema> {
        Err(IoError::UnsupportedFormat(
            "MySQL implementation pending".to_string(),
        ))
    }
}
