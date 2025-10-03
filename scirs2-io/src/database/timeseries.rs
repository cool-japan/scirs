//! Time series database operations

use crate::error::Result;
use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimePoint {
    pub timestamp: i64,
    pub value: f64,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    pub name: String,
    pub points: Vec<TimePoint>,
    pub tags: std::collections::HashMap<String, String>,
}

impl TimeSeries {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            points: Vec::new(),
            tags: std::collections::HashMap::new(),
        }
    }

    pub fn add_point(&mut self, timestamp: i64, value: f64) {
        self.points.push(TimePoint {
            timestamp,
            value,
            metadata: None,
        });
    }

    pub fn to_array(&self) -> Result<Array2<f64>> {
        let data: Vec<f64> = self
            .points
            .iter()
            .flat_map(|p| vec![p.timestamp as f64, p.value])
            .collect();

        Array2::from_shape_vec((self.points.len(), 2), data)
            .map_err(|e| crate::error::IoError::Other(e.to_string()))
    }
}
