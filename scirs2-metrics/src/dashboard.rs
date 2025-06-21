//! Interactive visualization dashboard for metrics
//!
//! This module provides a web-based interactive dashboard for visualizing
//! machine learning metrics in real-time, with export capabilities and
//! customizable visualizations.

use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Configuration for the interactive dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Server listening address
    pub address: SocketAddr,
    /// Auto-refresh interval in seconds
    pub refresh_interval: u64,
    /// Maximum number of data points to keep in memory
    pub max_data_points: usize,
    /// Enable real-time updates
    pub enable_realtime: bool,
    /// Dashboard title
    pub title: String,
    /// Theme configuration
    pub theme: DashboardTheme,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            address: "127.0.0.1:8080".parse().unwrap(),
            refresh_interval: 5,
            max_data_points: 1000,
            enable_realtime: true,
            title: "ML Metrics Dashboard".to_string(),
            theme: DashboardTheme::default(),
        }
    }
}

/// Theme configuration for the dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardTheme {
    /// Primary color (hex)
    pub primary_color: String,
    /// Background color (hex)
    pub background_color: String,
    /// Text color (hex)
    pub text_color: String,
    /// Chart colors
    pub chart_colors: Vec<String>,
}

impl Default for DashboardTheme {
    fn default() -> Self {
        Self {
            primary_color: "#2563eb".to_string(),
            background_color: "#ffffff".to_string(),
            text_color: "#1f2937".to_string(),
            chart_colors: vec![
                "#2563eb".to_string(),
                "#dc2626".to_string(),
                "#059669".to_string(),
                "#d97706".to_string(),
                "#7c3aed".to_string(),
                "#db2777".to_string(),
            ],
        }
    }
}

/// Metric data point for dashboard visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDataPoint {
    /// Timestamp of the measurement
    pub timestamp: u64,
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl MetricDataPoint {
    /// Create a new metric data point
    pub fn new(name: String, value: f64) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_secs();

        Self {
            timestamp,
            name,
            value,
            metadata: HashMap::new(),
        }
    }

    /// Create a new metric data point with metadata
    pub fn with_metadata(name: String, value: f64, metadata: HashMap<String, String>) -> Self {
        let mut point = Self::new(name, value);
        point.metadata = metadata;
        point
    }
}

/// Dashboard data storage and management
#[derive(Debug)]
pub struct DashboardData {
    /// Stored metric data points
    data_points: Arc<Mutex<Vec<MetricDataPoint>>>,
    /// Configuration
    config: DashboardConfig,
}

impl DashboardData {
    /// Create new dashboard data storage
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            data_points: Arc::new(Mutex::new(Vec::new())),
            config,
        }
    }

    /// Add a metric data point
    pub fn add_metric(&self, point: MetricDataPoint) -> Result<()> {
        let mut data = self
            .data_points
            .lock()
            .map_err(|_| MetricsError::InvalidInput("Failed to acquire data lock".to_string()))?;

        data.push(point);

        // Keep only the most recent data points
        if data.len() > self.config.max_data_points {
            let excess = data.len() - self.config.max_data_points;
            data.drain(0..excess);
        }

        Ok(())
    }

    /// Add multiple metric data points
    pub fn add_metrics(&self, points: Vec<MetricDataPoint>) -> Result<()> {
        for point in points {
            self.add_metric(point)?;
        }
        Ok(())
    }

    /// Get all metric data points
    pub fn get_all_metrics(&self) -> Result<Vec<MetricDataPoint>> {
        let data = self
            .data_points
            .lock()
            .map_err(|_| MetricsError::InvalidInput("Failed to acquire data lock".to_string()))?;

        Ok(data.clone())
    }

    /// Get metric data points by name
    pub fn get_metrics_by_name(&self, name: &str) -> Result<Vec<MetricDataPoint>> {
        let data = self
            .data_points
            .lock()
            .map_err(|_| MetricsError::InvalidInput("Failed to acquire data lock".to_string()))?;

        let filtered: Vec<MetricDataPoint> = data
            .iter()
            .filter(|point| point.name == name)
            .cloned()
            .collect();

        Ok(filtered)
    }

    /// Get metric names
    pub fn get_metric_names(&self) -> Result<Vec<String>> {
        let data = self
            .data_points
            .lock()
            .map_err(|_| MetricsError::InvalidInput("Failed to acquire data lock".to_string()))?;

        let mut names: Vec<String> = data.iter().map(|point| point.name.clone()).collect();
        names.sort();
        names.dedup();

        Ok(names)
    }

    /// Clear all data
    pub fn clear(&self) -> Result<()> {
        let mut data = self
            .data_points
            .lock()
            .map_err(|_| MetricsError::InvalidInput("Failed to acquire data lock".to_string()))?;

        data.clear();
        Ok(())
    }

    /// Get data points within time range
    pub fn get_metrics_in_range(
        &self,
        start_time: u64,
        end_time: u64,
    ) -> Result<Vec<MetricDataPoint>> {
        let data = self
            .data_points
            .lock()
            .map_err(|_| MetricsError::InvalidInput("Failed to acquire data lock".to_string()))?;

        let filtered: Vec<MetricDataPoint> = data
            .iter()
            .filter(|point| point.timestamp >= start_time && point.timestamp <= end_time)
            .cloned()
            .collect();

        Ok(filtered)
    }
}

/// Interactive dashboard server
#[derive(Debug)]
pub struct InteractiveDashboard {
    /// Dashboard data
    data: DashboardData,
    /// Server configuration
    config: DashboardConfig,
}

impl InteractiveDashboard {
    /// Create new interactive dashboard
    pub fn new(config: DashboardConfig) -> Self {
        let data = DashboardData::new(config.clone());

        Self { data, config }
    }
}

impl Default for InteractiveDashboard {
    fn default() -> Self {
        Self::new(DashboardConfig::default())
    }
}

impl InteractiveDashboard {
    /// Add metric measurement to dashboard
    pub fn add_metric(&self, name: &str, value: f64) -> Result<()> {
        let point = MetricDataPoint::new(name.to_string(), value);
        self.data.add_metric(point)
    }

    /// Add metric measurement with metadata
    pub fn add_metric_with_metadata(
        &self,
        name: &str,
        value: f64,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        let point = MetricDataPoint::with_metadata(name.to_string(), value, metadata);
        self.data.add_metric(point)
    }

    /// Add batch of metrics from arrays
    pub fn add_metrics_from_arrays(
        &self,
        metric_names: &[String],
        values: &Array1<f64>,
    ) -> Result<()> {
        if metric_names.len() != values.len() {
            return Err(MetricsError::InvalidInput(
                "Metric names and values must have same length".to_string(),
            ));
        }

        let points: Vec<MetricDataPoint> = metric_names
            .iter()
            .zip(values.iter())
            .map(|(name, &value)| MetricDataPoint::new(name.clone(), value))
            .collect();

        self.data.add_metrics(points)
    }

    /// Start the dashboard server (placeholder for actual web server)
    pub fn start_server(&self) -> Result<DashboardServer> {
        println!(
            "Starting dashboard server at http://{}",
            self.config.address
        );
        println!("Dashboard title: {}", self.config.title);
        println!("Refresh interval: {} seconds", self.config.refresh_interval);
        println!("Real-time updates: {}", self.config.enable_realtime);

        // In a real implementation, this would start an actual web server
        // For now, we return a mock server
        Ok(DashboardServer {
            address: self.config.address,
            is_running: true,
        })
    }

    /// Export data to JSON
    pub fn export_to_json(&self) -> Result<String> {
        let data = self.data.get_all_metrics()?;
        serde_json::to_string_pretty(&data)
            .map_err(|e| MetricsError::InvalidInput(format!("Failed to serialize data: {}", e)))
    }

    /// Export data to CSV
    pub fn export_to_csv(&self) -> Result<String> {
        let data = self.data.get_all_metrics()?;

        let mut csv = "timestamp,name,value,metadata\n".to_string();

        for point in data {
            let metadata_str = if point.metadata.is_empty() {
                String::new()
            } else {
                serde_json::to_string(&point.metadata).unwrap_or_default()
            };

            csv.push_str(&format!(
                "{},{},{},{}\n",
                point.timestamp, point.name, point.value, metadata_str
            ));
        }

        Ok(csv)
    }

    /// Get all metric data points
    pub fn get_all_metrics(&self) -> Result<Vec<MetricDataPoint>> {
        self.data.get_all_metrics()
    }

    /// Get metric data points by name
    pub fn get_metrics_by_name(&self, name: &str) -> Result<Vec<MetricDataPoint>> {
        self.data.get_metrics_by_name(name)
    }

    /// Get metric names
    pub fn get_metric_names(&self) -> Result<Vec<String>> {
        self.data.get_metric_names()
    }

    /// Get data points within time range
    pub fn get_metrics_in_range(
        &self,
        start_time: u64,
        end_time: u64,
    ) -> Result<Vec<MetricDataPoint>> {
        self.data.get_metrics_in_range(start_time, end_time)
    }

    /// Clear all data
    pub fn clear_data(&self) -> Result<()> {
        self.data.clear()
    }

    /// Get dashboard statistics
    pub fn get_statistics(&self) -> Result<DashboardStatistics> {
        let data = self.data.get_all_metrics()?;
        let metric_names = self.data.get_metric_names()?;

        let mut metric_counts = HashMap::new();
        let mut latest_values = HashMap::new();

        for point in &data {
            *metric_counts.entry(point.name.clone()).or_insert(0) += 1;
            latest_values.insert(point.name.clone(), point.value);
        }

        let total_points = data.len();
        let unique_metrics = metric_names.len();
        let time_range = if data.is_empty() {
            (0, 0)
        } else {
            let timestamps: Vec<u64> = data.iter().map(|p| p.timestamp).collect();
            (
                *timestamps.iter().min().unwrap(),
                *timestamps.iter().max().unwrap(),
            )
        };

        Ok(DashboardStatistics {
            total_data_points: total_points,
            unique_metrics,
            metric_counts,
            latest_values,
            time_range,
        })
    }

    /// Generate HTML dashboard (basic implementation)
    pub fn generate_html(&self) -> Result<String> {
        let stats = self.get_statistics()?;
        let data = self.data.get_all_metrics()?;

        let html = format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: {};
            color: {};
            margin: 0;
            padding: 20px;
        }}
        .header {{
            background-color: {};
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .data-table {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: {};
            color: white;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{}</h1>
        <p>Interactive Machine Learning Metrics Dashboard</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <h3>Total Data Points</h3>
            <p style="font-size: 24px; margin: 0;">{}</p>
        </div>
        <div class="stat-card">
            <h3>Unique Metrics</h3>
            <p style="font-size: 24px; margin: 0;">{}</p>
        </div>
        <div class="stat-card">
            <h3>Time Range</h3>
            <p style="font-size: 14px; margin: 0;">{} - {}</p>
        </div>
    </div>
    
    <div class="data-table">
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Metric Name</th>
                    <th>Value</th>
                    <th>Metadata</th>
                </tr>
            </thead>
            <tbody>
"#,
            self.config.title,
            self.config.theme.background_color,
            self.config.theme.text_color,
            self.config.theme.primary_color,
            self.config.theme.primary_color,
            self.config.title,
            stats.total_data_points,
            stats.unique_metrics,
            stats.time_range.0,
            stats.time_range.1
        );

        let mut rows = String::new();
        for point in data.iter().take(100) {
            // Show only first 100 points
            let metadata_display = if point.metadata.is_empty() {
                "-".to_string()
            } else {
                format!("{} keys", point.metadata.len())
            };

            rows.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{:.6}</td><td>{}</td></tr>\n",
                point.timestamp, point.name, point.value, metadata_display
            ));
        }

        let footer = r#"
            </tbody>
        </table>
    </div>
    
    <script>
        // Auto-refresh functionality (placeholder)
        setInterval(() => {
            console.log('Refreshing dashboard data...');
        }, 5000);
    </script>
</body>
</html>
"#;

        Ok(format!("{}{}{}", html, rows, footer))
    }
}

/// Dashboard server handle (placeholder for actual server)
#[derive(Debug)]
pub struct DashboardServer {
    /// Server address
    pub address: SocketAddr,
    /// Server running status
    pub is_running: bool,
}

impl DashboardServer {
    /// Stop the server
    pub fn stop(&mut self) {
        self.is_running = false;
        println!("Dashboard server stopped");
    }

    /// Check if server is running
    pub fn is_running(&self) -> bool {
        self.is_running
    }
}

/// Dashboard statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardStatistics {
    /// Total number of data points
    pub total_data_points: usize,
    /// Number of unique metrics
    pub unique_metrics: usize,
    /// Count of data points per metric
    pub metric_counts: HashMap<String, usize>,
    /// Latest value for each metric
    pub latest_values: HashMap<String, f64>,
    /// Time range (start, end) timestamps
    pub time_range: (u64, u64),
}

/// Dashboard widget for embedding metrics
#[derive(Debug, Clone)]
pub struct DashboardWidget {
    /// Widget identifier
    pub id: String,
    /// Widget title
    pub title: String,
    /// Metric names to display
    pub metrics: Vec<String>,
    /// Widget type
    pub widget_type: WidgetType,
    /// Configuration options
    pub config: HashMap<String, String>,
}

/// Types of dashboard widgets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetType {
    /// Line chart for time series data
    LineChart,
    /// Bar chart for categorical data
    BarChart,
    /// Gauge for single value metrics
    Gauge,
    /// Table for tabular data
    Table,
    /// Heatmap for correlation matrices
    Heatmap,
    /// Confusion matrix visualization
    ConfusionMatrix,
    /// ROC curve
    RocCurve,
    /// Custom widget type
    Custom(String),
}

impl DashboardWidget {
    /// Create new line chart widget
    pub fn line_chart(id: String, title: String, metrics: Vec<String>) -> Self {
        Self {
            id,
            title,
            metrics,
            widget_type: WidgetType::LineChart,
            config: HashMap::new(),
        }
    }

    /// Create new gauge widget
    pub fn gauge(id: String, title: String, metric: String) -> Self {
        Self {
            id,
            title,
            metrics: vec![metric],
            widget_type: WidgetType::Gauge,
            config: HashMap::new(),
        }
    }

    /// Create new table widget
    pub fn table(id: String, title: String, metrics: Vec<String>) -> Self {
        Self {
            id,
            title,
            metrics,
            widget_type: WidgetType::Table,
            config: HashMap::new(),
        }
    }

    /// Add configuration option
    pub fn with_config(mut self, key: String, value: String) -> Self {
        self.config.insert(key, value);
        self
    }
}

/// Utility functions for dashboard creation
pub mod utils {
    use super::*;

    /// Create a dashboard from classification metrics
    pub fn create_classification_dashboard(
        accuracy: f64,
        precision: f64,
        recall: f64,
        f1_score: f64,
    ) -> Result<InteractiveDashboard> {
        let dashboard = InteractiveDashboard::default();

        dashboard.add_metric("accuracy", accuracy)?;
        dashboard.add_metric("precision", precision)?;
        dashboard.add_metric("recall", recall)?;
        dashboard.add_metric("f1_score", f1_score)?;

        Ok(dashboard)
    }

    /// Create a dashboard from regression metrics
    pub fn create_regression_dashboard(
        mse: f64,
        rmse: f64,
        mae: f64,
        r2: f64,
    ) -> Result<InteractiveDashboard> {
        let dashboard = InteractiveDashboard::default();

        dashboard.add_metric("mse", mse)?;
        dashboard.add_metric("rmse", rmse)?;
        dashboard.add_metric("mae", mae)?;
        dashboard.add_metric("r2", r2)?;

        Ok(dashboard)
    }

    /// Create a dashboard from clustering metrics
    pub fn create_clustering_dashboard(
        silhouette_score: f64,
        davies_bouldin: f64,
        calinski_harabasz: f64,
    ) -> Result<InteractiveDashboard> {
        let dashboard = InteractiveDashboard::default();

        dashboard.add_metric("silhouette_score", silhouette_score)?;
        dashboard.add_metric("davies_bouldin", davies_bouldin)?;
        dashboard.add_metric("calinski_harabasz", calinski_harabasz)?;

        Ok(dashboard)
    }

    /// Export dashboard data to file
    pub fn export_dashboard_to_file(
        dashboard: &InteractiveDashboard,
        file_path: &str,
        format: ExportFormat,
    ) -> Result<()> {
        let content = match format {
            ExportFormat::Json => dashboard.export_to_json()?,
            ExportFormat::Csv => dashboard.export_to_csv()?,
            ExportFormat::Html => dashboard.generate_html()?,
        };

        std::fs::write(file_path, content)
            .map_err(|e| MetricsError::InvalidInput(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Export formats
    #[derive(Debug, Clone)]
    pub enum ExportFormat {
        Json,
        Csv,
        Html,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_creation() {
        let config = DashboardConfig::default();
        let dashboard = InteractiveDashboard::new(config);

        assert!(dashboard.add_metric("accuracy", 0.95).is_ok());
        assert!(dashboard.add_metric("precision", 0.92).is_ok());
        assert!(dashboard.add_metric("recall", 0.88).is_ok());
    }

    #[test]
    fn test_metric_data_point() {
        let point = MetricDataPoint::new("accuracy".to_string(), 0.95);
        assert_eq!(point.name, "accuracy");
        assert_eq!(point.value, 0.95);
        assert!(point.timestamp > 0);
    }

    #[test]
    fn test_dashboard_data() {
        let config = DashboardConfig::default();
        let data = DashboardData::new(config);

        let point1 = MetricDataPoint::new("accuracy".to_string(), 0.95);
        let point2 = MetricDataPoint::new("precision".to_string(), 0.92);

        assert!(data.add_metric(point1).is_ok());
        assert!(data.add_metric(point2).is_ok());

        let all_metrics = data.get_all_metrics().unwrap();
        assert_eq!(all_metrics.len(), 2);

        let accuracy_metrics = data.get_metrics_by_name("accuracy").unwrap();
        assert_eq!(accuracy_metrics.len(), 1);
        assert_eq!(accuracy_metrics[0].value, 0.95);
    }

    #[test]
    fn test_dashboard_statistics() {
        let dashboard = InteractiveDashboard::default();

        assert!(dashboard.add_metric("accuracy", 0.95).is_ok());
        assert!(dashboard.add_metric("precision", 0.92).is_ok());
        assert!(dashboard.add_metric("accuracy", 0.97).is_ok());

        let stats = dashboard.get_statistics().unwrap();
        assert_eq!(stats.total_data_points, 3);
        assert_eq!(stats.unique_metrics, 2);
        assert_eq!(stats.metric_counts["accuracy"], 2);
        assert_eq!(stats.metric_counts["precision"], 1);
    }

    #[test]
    fn test_export_functions() {
        let dashboard = InteractiveDashboard::default();

        assert!(dashboard.add_metric("accuracy", 0.95).is_ok());
        assert!(dashboard.add_metric("precision", 0.92).is_ok());

        let json_export = dashboard.export_to_json();
        assert!(json_export.is_ok());
        assert!(json_export.unwrap().contains("accuracy"));

        let csv_export = dashboard.export_to_csv();
        assert!(csv_export.is_ok());
        assert!(csv_export.unwrap().contains("timestamp,name,value"));

        let html_export = dashboard.generate_html();
        assert!(html_export.is_ok());
        assert!(html_export.unwrap().contains("<!DOCTYPE html>"));
    }

    #[test]
    fn test_dashboard_widgets() {
        let widget = DashboardWidget::line_chart(
            "accuracy_chart".to_string(),
            "Model Accuracy".to_string(),
            vec!["accuracy".to_string()],
        );

        assert_eq!(widget.id, "accuracy_chart");
        assert_eq!(widget.title, "Model Accuracy");
        assert!(matches!(widget.widget_type, WidgetType::LineChart));
    }

    #[test]
    fn test_utility_functions() {
        let dashboard = utils::create_classification_dashboard(0.95, 0.92, 0.88, 0.90).unwrap();
        let stats = dashboard.get_statistics().unwrap();

        assert_eq!(stats.total_data_points, 4);
        assert_eq!(stats.unique_metrics, 4);
        assert!(stats.latest_values.contains_key("accuracy"));
    }
}
