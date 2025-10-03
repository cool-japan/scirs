//! Actual web server implementation for the dashboard
//!
//! This module provides a simple HTTP server implementation using tokio
//! for serving the interactive dashboard without requiring heavy web framework dependencies.

use super::{DashboardConfig, DashboardData, InteractiveDashboard, MetricDataPoint};
use crate::error::{MetricsError, Result};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::runtime::Runtime;
use tokio::sync::RwLock;

/// HTTP response builder
struct HttpResponse {
    status_code: u16,
    statustext: &'static str,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl HttpResponse {
    /// Create a new OK response
    fn ok() -> Self {
        Self {
            status_code: 200,
            statustext: "OK",
            headers: vec![],
            body: vec![],
        }
    }

    /// Create a not found response
    fn not_found() -> Self {
        Self {
            status_code: 404,
            statustext: "Not Found",
            headers: vec![],
            body: b"404 Not Found".to_vec(),
        }
    }

    /// Create an internal server error response
    fn internal_error() -> Self {
        Self {
            status_code: 500,
            statustext: "Internal Server Error",
            headers: vec![],
            body: b"500 Internal Server Error".to_vec(),
        }
    }

    /// Set content type header
    fn content_type(mut self, content_type: &str) -> Self {
        self.headers
            .push(("Content-Type".to_string(), content_type.to_string()));
        self
    }

    /// Set response body
    fn body(mut self, body: Vec<u8>) -> Self {
        self.body = body;
        self
    }

    /// Convert to HTTP response bytes
    fn into_bytes(self) -> Vec<u8> {
        let mut response = format!("HTTP/1.1 {} {}\r\n", self.status_code, self.statustext);

        // Add content length header
        response.push_str(&format!("Content-Length: {}\r\n", self.body.len()));

        // Add other headers
        for (key, value) in self.headers {
            response.push_str(&format!("{}: {}\r\n", key, value));
        }

        // Add CORS headers for development
        response.push_str("Access-Control-Allow-Origin: *\r\n");
        response.push_str("Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n");
        response.push_str("Access-Control-Allow-Headers: Content-Type\r\n");

        // End headers
        response.push_str("\r\n");

        // Combine headers and body
        let mut bytes = response.into_bytes();
        bytes.extend_from_slice(&self.body);
        bytes
    }
}

/// Parse HTTP request path from raw request
#[allow(dead_code)]
fn parse_request_path(request: &str) -> Option<String> {
    let lines: Vec<&str> = request.lines().collect();
    if lines.is_empty() {
        return None;
    }

    let parts: Vec<&str> = lines[0].split(' ').collect();
    if parts.len() >= 2 {
        Some(parts[1].to_string())
    } else {
        None
    }
}

/// Actual dashboard server implementation
pub struct DashboardHttpServer {
    dashboard: InteractiveDashboard,
    runtime: Option<Runtime>,
}

impl DashboardHttpServer {
    /// Create a new dashboard HTTP server
    pub fn new(dashboard: InteractiveDashboard) -> Result<Self> {
        // Create a runtime for the server
        let runtime = Runtime::new()
            .map_err(|e| MetricsError::InvalidInput(format!("Failed to create runtime: {}", e)))?;

        Ok(Self {
            dashboard,
            runtime: Some(runtime),
        })
    }

    /// Start the HTTP server
    pub fn start(&mut self) -> Result<()> {
        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| MetricsError::InvalidInput("Runtime not initialized".to_string()))?;

        let addr = self.dashboard.config.address;
        let dashboard = self.dashboard.clone();

        runtime.spawn(async move {
            if let Err(e) = serve_dashboard(addr, dashboard).await {
                eprintln!("Dashboard server error: {}", e);
            }
        });

        println!(
            "Dashboard server started at http://{}",
            self.dashboard.config.address
        );
        Ok(())
    }

    /// Add metric data point
    pub fn add_metric(&self, name: &str, value: f64) -> Result<()> {
        self.dashboard.add_metric(name, value)
    }

    /// Get all metrics
    pub fn get_all_metrics(&self) -> Result<Vec<MetricDataPoint>> {
        self.dashboard.get_all_metrics()
    }

    /// Stop the server
    pub fn stop(mut self) {
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
        println!("Dashboard server stopped");
    }
}

/// Serve the dashboard over HTTP
async fn serve_dashboard(addr: SocketAddr, dashboard: InteractiveDashboard) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    println!("Dashboard listening on http://{}", addr);

    let dashboard = Arc::new(dashboard);

    loop {
        let (stream, _) = listener.accept().await?;
        let dashboard = Arc::clone(&dashboard);

        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, dashboard).await {
                eprintln!("Error handling connection: {}", e);
            }
        });
    }
}

/// Handle individual HTTP connections
async fn handle_connection(
    mut stream: TcpStream,
    dashboard: Arc<InteractiveDashboard>,
) -> std::io::Result<()> {
    let mut buffer = [0; 1024];
    let n = stream.read(&mut buffer).await?;
    let request = String::from_utf8_lossy(&buffer[..n]);

    let response = if let Some(path) = parse_request_path(&request) {
        match path.as_str() {
            "/" => {
                // Serve the main dashboard HTML
                let html = generate_dashboard_html(&dashboard).await;
                HttpResponse::ok()
                    .content_type("text/html; charset=utf-8")
                    .body(html.into_bytes())
            }
            "/api/metrics" => {
                // Serve metrics data as JSON
                match dashboard.get_all_metrics() {
                    Ok(metrics) => {
                        let json = serde_json::to_string(&metrics).unwrap_or_default();
                        HttpResponse::ok()
                            .content_type("application/json")
                            .body(json.into_bytes())
                    }
                    Err(_) => HttpResponse::internal_error(),
                }
            }
            "/api/metrics/names" => {
                // Serve metric names as JSON
                match dashboard.get_metric_names() {
                    Ok(names) => {
                        let json = serde_json::to_string(&names).unwrap_or_default();
                        HttpResponse::ok()
                            .content_type("application/json")
                            .body(json.into_bytes())
                    }
                    Err(_) => HttpResponse::internal_error(),
                }
            }
            _ => HttpResponse::not_found(),
        }
    } else {
        HttpResponse::not_found()
    };

    stream.write_all(&response.into_bytes()).await?;
    stream.flush().await?;
    Ok(())
}

/// AI-driven insights structure
#[derive(Debug, Clone, serde::Serialize)]
struct AiInsight {
    pub insight_type: String,
    pub message: String,
    pub confidence: f64,
    pub severity: String,
    pub timestamp: u64,
}

/// Anomaly alert structure
#[derive(Debug, Clone, serde::Serialize)]
struct AnomalyAlert {
    pub metric_name: String,
    pub anomaly_type: String,
    pub severity: String,
    pub value: f64,
    pub expected_range: (f64, f64),
    pub timestamp: u64,
}

/// Performance prediction structure
#[derive(Debug, Clone, serde::Serialize)]
struct PerformancePrediction {
    pub metric_name: String,
    pub predicted_value: f64,
    pub confidence_interval: (f64, f64),
    pub prediction_horizon: u64, // seconds into future
    pub trend_direction: String,
}

/// Generate AI-driven insights from metrics data
#[allow(dead_code)]
fn generate_ai_insights(metrics: &[MetricDataPoint]) -> Vec<AiInsight> {
    let mut insights = Vec::new();

    if metrics.is_empty() {
        return insights;
    }

    // Analyze metric trends
    let mut metric_groups: std::collections::HashMap<String, Vec<&MetricDataPoint>> =
        std::collections::HashMap::new();
    for metric in metrics {
        metric_groups
            .entry(metric.name.clone())
            .or_default()
            .push(metric);
    }

    for (metric_name, points) in metric_groups {
        if points.len() < 3 {
            continue;
        }

        // Sort by timestamp
        let mut sorted_points = points;
        sorted_points.sort_by_key(|p| p.timestamp);

        // Calculate trend
        let first_val = sorted_points.first().unwrap().value;
        let last_val = sorted_points.last().unwrap().value;
        let change_percent = ((last_val - first_val) / first_val.abs().max(0.001)) * 100.0;

        if change_percent.abs() > 20.0 {
            let trend_type = if change_percent > 0.0 {
                "improvement"
            } else {
                "degradation"
            };
            let severity = if change_percent.abs() > 50.0 {
                "high"
            } else {
                "medium"
            };

            insights.push(AiInsight {
                insight_type: "trend_analysis".to_string(),
                message: format!(
                    "{} shows significant {} ({:.1}% change)",
                    metric_name, trend_type, change_percent
                ),
                confidence: 0.85,
                severity: severity.to_string(),
                timestamp: sorted_points.last().unwrap().timestamp,
            });
        }

        // Check for volatility
        if sorted_points.len() > 5 {
            let values: Vec<f64> = sorted_points.iter().map(|p| p.value).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
            let std_dev = variance.sqrt();
            let cv = std_dev / mean.abs().max(0.001); // Coefficient of variation

            if cv > 0.3 {
                insights.push(AiInsight {
                    insight_type: "volatility_analysis".to_string(),
                    message: format!("{} shows high volatility (CV: {:.2})", metric_name, cv),
                    confidence: 0.75,
                    severity: "medium".to_string(),
                    timestamp: sorted_points.last().unwrap().timestamp,
                });
            }
        }
    }

    // Overall system health insight
    let avg_values: Vec<f64> = metrics.iter().map(|m| m.value).collect();
    if !avg_values.is_empty() {
        let overall_avg = avg_values.iter().sum::<f64>() / avg_values.len() as f64;

        if overall_avg > 0.9 {
            insights.push(AiInsight {
                insight_type: "system_health".to_string(),
                message: "System performance is excellent across all metrics".to_string(),
                confidence: 0.90,
                severity: "info".to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            });
        } else if overall_avg < 0.5 {
            insights.push(AiInsight {
                insight_type: "system_health".to_string(),
                message: "System performance may need attention - multiple metrics below optimal"
                    .to_string(),
                confidence: 0.80,
                severity: "high".to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            });
        }
    }

    insights
}

/// Detect anomalies in metrics data using statistical methods
#[allow(dead_code)]
fn detect_anomalies(metrics: &[MetricDataPoint]) -> Vec<AnomalyAlert> {
    let mut alerts = Vec::new();

    if metrics.is_empty() {
        return alerts;
    }

    let mut metric_groups: std::collections::HashMap<String, Vec<&MetricDataPoint>> =
        std::collections::HashMap::new();
    for metric in metrics {
        metric_groups
            .entry(metric.name.clone())
            .or_default()
            .push(metric);
    }

    for (metric_name, points) in metric_groups {
        if points.len() < 5 {
            continue;
        }

        let values: Vec<f64> = points.iter().map(|p| p.value).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        // Use z-score for anomaly detection (threshold = 2.5 sigma)
        let threshold = 2.5;

        for point in &points {
            let z_score = (point.value - mean) / std_dev.max(0.001);

            if z_score.abs() > threshold {
                let expected_range = (mean - threshold * std_dev, mean + threshold * std_dev);
                let severity = if z_score.abs() > 3.0 {
                    "critical"
                } else {
                    "warning"
                };
                let anomaly_type = if z_score > 0.0 { "spike" } else { "drop" };

                alerts.push(AnomalyAlert {
                    metric_name: metric_name.clone(),
                    anomaly_type: anomaly_type.to_string(),
                    severity: severity.to_string(),
                    value: point.value,
                    expected_range,
                    timestamp: point.timestamp,
                });
            }
        }
    }

    alerts
}

/// Predict future performance using simple linear regression
#[allow(dead_code)]
fn predict_future_performance(metrics: &[MetricDataPoint]) -> Vec<PerformancePrediction> {
    let mut predictions = Vec::new();

    if metrics.is_empty() {
        return predictions;
    }

    let mut metric_groups: std::collections::HashMap<String, Vec<&MetricDataPoint>> =
        std::collections::HashMap::new();
    for metric in metrics {
        metric_groups
            .entry(metric.name.clone())
            .or_default()
            .push(metric);
    }

    for (metric_name, points) in metric_groups {
        if points.len() < 3 {
            continue;
        }

        // Sort by timestamp
        let mut sorted_points = points;
        sorted_points.sort_by_key(|p| p.timestamp);

        // Simple linear regression
        let n = sorted_points.len() as f64;
        let x_values: Vec<f64> = sorted_points
            .iter()
            .enumerate()
            .map(|(i, _)| i as f64)
            .collect();
        let y_values: Vec<f64> = sorted_points.iter().map(|p| p.value).collect();

        let sum_x = x_values.iter().sum::<f64>();
        let sum_y = y_values.iter().sum::<f64>();
        let sum_xy = x_values
            .iter()
            .zip(y_values.iter())
            .map(|(x, y)| x * y)
            .sum::<f64>();
        let sum_x2 = x_values.iter().map(|x| x * x).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Predict 5 minutes into the future
        let prediction_horizon = 300; // 5 minutes in seconds
        let future_x = n; // Next time point
        let predicted_value = slope * future_x + intercept;

        // Calculate confidence interval (simplified)
        let residuals: Vec<f64> = x_values
            .iter()
            .zip(y_values.iter())
            .map(|(x, y)| y - (slope * x + intercept))
            .collect();
        let residual_var = residuals.iter().map(|r| r * r).sum::<f64>() / (n - 2.0).max(1.0);
        let std_error = residual_var.sqrt();

        let confidence_interval = (
            predicted_value - 1.96 * std_error,
            predicted_value + 1.96 * std_error,
        );

        let trend_direction = if slope > 0.01 {
            "increasing"
        } else if slope < -0.01 {
            "decreasing"
        } else {
            "stable"
        };

        predictions.push(PerformancePrediction {
            metric_name,
            predicted_value,
            confidence_interval,
            prediction_horizon,
            trend_direction: trend_direction.to_string(),
        });
    }

    predictions
}

/// Generate enhanced dashboard HTML with advanced Advanced features
async fn generate_dashboard_html(dashboard: &Arc<InteractiveDashboard>) -> String {
    let metrics = dashboard.get_all_metrics().unwrap_or_default();
    let metric_names = dashboard.get_metric_names().unwrap_or_default();
    let config = &dashboard.config;

    let metrics_json = serde_json::to_string(&metrics).unwrap_or_default();

    // Generate AI-driven insights
    let _ai_insights = generate_ai_insights(&metrics);
    let _anomaly_alerts = detect_anomalies(&metrics);
    let _performance_predictions = predict_future_performance(&metrics);

    format!(
        r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: {};
            color: {};
            margin: 0;
            padding: 0;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, {}, #667eea);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: {};
            margin: 10px 0;
        }}
        .chart-container {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
        }}
        .chart-wrapper {{
            position: relative;
            height: 400px;
        }}
        .metric-selector {{
            margin-bottom: 20px;
        }}
        .metric-selector select {{
            padding: 10px 15px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 16px;
            background: white;
            cursor: pointer;
        }}
        .refresh-info {{
            text-align: right;
            color: #666;
            font-size: 14px;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{}</h1>
            <p>Float-time Machine Learning Metrics Dashboard</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Metrics</h3>
                <div class="stat-value">{}</div>
                <p>Data points collected</p>
            </div>
            <div class="stat-card">
                <h3>Unique Metrics</h3>
                <div class="stat-value">{}</div>
                <p>Different metric types</p>
            </div>
            <div class="stat-card">
                <h3>Latest Update</h3>
                <div class="stat-value" id="latest-update">--:--:--</div>
                <p>Last data received</p>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>Metric Visualization</h2>
            <div class="metric-selector">
                <label for="metric-select">Select Metric: </label>
                <select id="metric-select" onchange="updateChart()">
                    <option value="">All Metrics</option>
                    {}
                </select>
            </div>
            <div class="chart-wrapper">
                <canvas id="metricsChart"></canvas>
            </div>
            <div class="refresh-info">
                Auto-refresh: every {} seconds
            </div>
        </div>
    </div>
    
    <script>
        // Initial metrics data
        let metricsData = {};
        
        // Chart instance
        let chart = null;
        
        // Initialize chart
        function initChart() {{
            const ctx = document.getElementById('metricsChart').getContext('2d');
            chart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: [],
                    datasets: []
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        x: {{
                            display: true,
                            title: {{
                                display: true,
                                text: 'Timestamp'
                            }}
                        }},
                        y: {{
                            display: true,
                            title: {{
                                display: true,
                                text: 'Value'
                            }}
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            display: true,
                            position: 'top'
                        }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false
                        }}
                    }}
                }}
            }});
        }}
        
        // Update chart with new data
        function updateChart() {{
            const selectedMetric = document.getElementById('metric-select').value;
            
            // Group metrics by name
            const groupedMetrics = {{}};
            metricsData.forEach(point => {{
                if (!selectedMetric || point.name === selectedMetric) {{
                    if (!groupedMetrics[point.name]) {{
                        groupedMetrics[point.name] = [];
                    }}
                    groupedMetrics[point.name].push(point);
                }}
            }});
            
            // Prepare datasets
            const datasets = [];
            const colors = {};
            
            Object.keys(groupedMetrics).forEach((name, index) => {{
                const color = colors[index % colors.length];
                datasets.push({{
                    label: name,
                    data: groupedMetrics[name].map(p => ({{
                        x: new Date(p.timestamp * 1000),
                        y: p.value
                    }})),
                    borderColor: color,
                    backgroundColor: color + '20',
                    tension: 0.1
                }});
            }});
            
            // Update chart
            chart.data.datasets = datasets;
            chart.update();
        }}
        
        // Fetch latest metrics
        async function fetchMetrics() {{
            try {{
                const response = await fetch('/api/metrics');
                const data = await response.json();
                metricsData = data;
                updateChart();
                updateLatestTime();
            }} catch (error) {{
                console.error('Error fetching metrics:', error);
            }}
        }}
        
        // Update latest time
        function updateLatestTime() {{
            const now = new Date();
            const timeStr = now.toLocaleTimeString();
            document.getElementById('latest-update').textContent = timeStr;
        }}
        
        // Initialize on page load
        window.onload = function() {{
            initChart();
            fetchMetrics();
            
            // Auto-refresh
            setInterval(fetchMetrics, {} * 1000);
        }};
    </script>
</body>
</html>
"#,
        config.title,
        config.theme.background_color,
        config.theme.text_color,
        config.theme.primary_color,
        config.theme.primary_color,
        config.title,
        metrics.len(),
        metric_names.len(),
        metric_names
            .iter()
            .map(|name| format!(r#"<option value="{}">{}</option>"#, name, name))
            .collect::<Vec<_>>()
            .join("\n                    "),
        config.refresh_interval,
        serde_json::to_string(&config.theme.chart_colors).unwrap_or_default(),
        metrics_json,
        config.refresh_interval
    )
}

/// Create and start an HTTP server for the given dashboard
#[allow(dead_code)]
pub fn start_http_server(dashboard: InteractiveDashboard) -> Result<DashboardHttpServer> {
    let mut server = DashboardHttpServer::new(dashboard)?;
    server.start()?;
    Ok(server)
}
