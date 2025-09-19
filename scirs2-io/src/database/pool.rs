//! Connection pool for database connections

use crate::database::{DatabaseConfig, DatabaseConnection, DatabaseConnector, DatabaseType};
use crate::error::Result;
use std::sync::{Arc, Mutex};

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

    pub fn get_connection(&self) -> Result<Box<dyn DatabaseConnection>> {
        let mut pool = self.connections.lock().unwrap();

        if let Some(conn) = pool.pop() {
            Ok(conn)
        } else if pool.len() < self.max_connections {
            DatabaseConnector::connect(&self.config)
        } else {
            // In production, this would wait for a connection to be available
            DatabaseConnector::connect(&self.config)
        }
    }

    #[allow(dead_code)]
    pub fn return_connection(&self, conn: Box<dyn DatabaseConnection>) {
        let mut pool = self.connections.lock().unwrap();
        if pool.len() < self.max_connections {
            pool.push(conn);
        }
    }
}
