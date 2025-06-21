//! Example of interactive dashboard for metrics visualization
//!
//! This example demonstrates how to create and use an interactive dashboard
//! for real-time metrics monitoring and visualization.

use ndarray::Array1;
use scirs2_metrics::dashboard::utils::*;
use scirs2_metrics::dashboard::*;
use scirs2_metrics::error::Result;
use std::collections::HashMap;
use std::thread;
use std::time::Duration;

fn main() -> Result<()> {
    println!("Interactive Dashboard Example");
    println!("===========================");

    // Example 1: Basic Dashboard Creation
    println!("\n1. Basic Dashboard Creation");
    println!("-------------------------");

    basic_dashboard_example()?;

    // Example 2: Real-time Metrics Monitoring
    println!("\n2. Real-time Metrics Monitoring Simulation");
    println!("----------------------------------------");

    realtime_monitoring_example()?;

    // Example 3: Domain-specific Dashboards
    println!("\n3. Domain-specific Dashboards");
    println!("----------------------------");

    domain_specific_dashboards_example()?;

    // Example 4: Dashboard Export and Data Management
    println!("\n4. Dashboard Export and Data Management");
    println!("-------------------------------------");

    export_and_data_management_example()?;

    // Example 5: Advanced Dashboard Configuration
    println!("\n5. Advanced Dashboard Configuration");
    println!("---------------------------------");

    advanced_configuration_example()?;

    // Example 6: Widget System
    println!("\n6. Widget System");
    println!("--------------");

    widget_system_example()?;

    println!("\nInteractive dashboard example completed successfully!");
    Ok(())
}

/// Example of basic dashboard creation and usage
fn basic_dashboard_example() -> Result<()> {
    // Create a dashboard with default configuration
    let dashboard = InteractiveDashboard::default();

    // Add some sample metrics
    dashboard.add_metric("accuracy", 0.95)?;
    dashboard.add_metric("precision", 0.92)?;
    dashboard.add_metric("recall", 0.88)?;
    dashboard.add_metric("f1_score", 0.90)?;

    // Add metrics with metadata
    let mut metadata = HashMap::new();
    metadata.insert("model".to_string(), "Random Forest".to_string());
    metadata.insert("epoch".to_string(), "10".to_string());

    dashboard.add_metric_with_metadata("validation_accuracy", 0.93, metadata)?;

    // Get dashboard statistics
    let stats = dashboard.get_statistics()?;
    println!("Dashboard Statistics:");
    println!("  Total data points: {}", stats.total_data_points);
    println!("  Unique metrics: {}", stats.unique_metrics);
    println!("  Latest values:");
    for (metric, value) in &stats.latest_values {
        println!("    {}: {:.4}", metric, value);
    }

    // Start the dashboard server (mock implementation)
    let server = dashboard.start_server()?;
    println!("Dashboard server started at: http://{}", server.address);

    // In a real application, the server would run continuously
    // For this example, we just show it can be started
    println!("Server is running: {}", server.is_running());

    Ok(())
}

/// Example of real-time metrics monitoring
fn realtime_monitoring_example() -> Result<()> {
    // Create dashboard with custom configuration
    let config = DashboardConfig {
        address: "127.0.0.1:8081".parse().unwrap(),
        refresh_interval: 2,
        max_data_points: 100,
        enable_realtime: true,
        title: "Real-time ML Training Dashboard".to_string(),
        theme: DashboardTheme {
            primary_color: "#059669".to_string(),
            background_color: "#f9fafb".to_string(),
            text_color: "#111827".to_string(),
            chart_colors: vec![
                "#059669".to_string(),
                "#dc2626".to_string(),
                "#d97706".to_string(),
                "#7c3aed".to_string(),
            ],
        },
    };

    let dashboard = InteractiveDashboard::new(config);

    println!("Simulating real-time training metrics...");

    // Simulate a training loop with improving metrics
    for epoch in 1..=10 {
        // Simulate metrics that improve over time
        let base_accuracy = 0.6;
        let improvement = (epoch as f64) * 0.03;
        let noise = (epoch as f64 * 0.1).sin() * 0.01;

        let accuracy = base_accuracy + improvement + noise;
        let loss = 2.0 - improvement * 2.0 + noise.abs();
        let val_accuracy = accuracy - 0.02 + noise * 0.5;
        let val_loss = loss + 0.1 - noise.abs();

        // Add training metrics
        let mut train_metadata = HashMap::new();
        train_metadata.insert("epoch".to_string(), epoch.to_string());
        train_metadata.insert("phase".to_string(), "training".to_string());

        dashboard.add_metric_with_metadata("train_accuracy", accuracy, train_metadata.clone())?;
        dashboard.add_metric_with_metadata("train_loss", loss, train_metadata)?;

        // Add validation metrics
        let mut val_metadata = HashMap::new();
        val_metadata.insert("epoch".to_string(), epoch.to_string());
        val_metadata.insert("phase".to_string(), "validation".to_string());

        dashboard.add_metric_with_metadata("val_accuracy", val_accuracy, val_metadata.clone())?;
        dashboard.add_metric_with_metadata("val_loss", val_loss, val_metadata)?;

        println!(
            "  Epoch {}: Train Acc={:.4}, Train Loss={:.4}, Val Acc={:.4}, Val Loss={:.4}",
            epoch, accuracy, loss, val_accuracy, val_loss
        );

        // Simulate processing time
        thread::sleep(Duration::from_millis(100));
    }

    // Get final statistics
    let stats = dashboard.get_statistics()?;
    println!("\nTraining complete! Final statistics:");
    println!("  Total data points: {}", stats.total_data_points);
    println!("  Metrics tracked: {}", stats.unique_metrics);

    // Show metrics by name
    for metric_name in dashboard.get_metric_names()? {
        let metrics = dashboard.get_metrics_by_name(&metric_name)?;
        if let Some(last_metric) = metrics.last() {
            println!("  Final {}: {:.4}", metric_name, last_metric.value);
        }
    }

    Ok(())
}

/// Example of domain-specific dashboards
fn domain_specific_dashboards_example() -> Result<()> {
    println!("Creating domain-specific dashboards...");

    // Classification dashboard
    println!("\n  Classification Dashboard:");
    let classification_dashboard = create_classification_dashboard(0.95, 0.92, 0.88, 0.90)?;
    let class_stats = classification_dashboard.get_statistics()?;
    println!("    Metrics: {:?}", class_stats.latest_values);

    // Regression dashboard
    println!("\n  Regression Dashboard:");
    let regression_dashboard = create_regression_dashboard(0.05, 0.22, 0.18, 0.94)?;
    let reg_stats = regression_dashboard.get_statistics()?;
    println!("    Metrics: {:?}", reg_stats.latest_values);

    // Clustering dashboard
    println!("\n  Clustering Dashboard:");
    let clustering_dashboard = create_clustering_dashboard(0.75, 0.45, 850.2)?;
    let cluster_stats = clustering_dashboard.get_statistics()?;
    println!("    Metrics: {:?}", cluster_stats.latest_values);

    // Add additional metrics to classification dashboard
    let mut model_metadata = HashMap::new();
    model_metadata.insert("model_type".to_string(), "SVM".to_string());
    model_metadata.insert("kernel".to_string(), "rbf".to_string());

    classification_dashboard.add_metric_with_metadata("auc_score", 0.96, model_metadata)?;

    // Batch add metrics from arrays
    let metric_names = vec![
        "specificity".to_string(),
        "sensitivity".to_string(),
        "mcc".to_string(),
    ];
    let values = Array1::from_vec(vec![0.89, 0.91, 0.85]);

    classification_dashboard.add_metrics_from_arrays(&metric_names, &values)?;

    let final_class_stats = classification_dashboard.get_statistics()?;
    println!(
        "    Final classification metrics count: {}",
        final_class_stats.total_data_points
    );

    Ok(())
}

/// Example of export and data management functionality
fn export_and_data_management_example() -> Result<()> {
    let dashboard = InteractiveDashboard::default();

    // Add sample data with time-based metadata
    for i in 0..20 {
        let timestamp_offset = i * 60; // 1 minute intervals
        let mut metadata = HashMap::new();
        metadata.insert("batch".to_string(), (i / 5).to_string());
        metadata.insert("timestamp_offset".to_string(), timestamp_offset.to_string());

        let accuracy = 0.8 + (i as f64 * 0.01) + (i as f64 * 0.1).sin() * 0.02;
        let loss = 1.0 - (i as f64 * 0.03) + (i as f64 * 0.15).cos() * 0.05;

        dashboard.add_metric_with_metadata("batch_accuracy", accuracy, metadata.clone())?;
        dashboard.add_metric_with_metadata("batch_loss", loss, metadata)?;
    }

    // Export to different formats
    println!("Exporting dashboard data...");

    // JSON export
    let json_data = dashboard.export_to_json()?;
    println!("  JSON export size: {} characters", json_data.len());

    // CSV export
    let csv_data = dashboard.export_to_csv()?;
    println!("  CSV export size: {} characters", csv_data.len());
    println!("  CSV header: {}", csv_data.lines().next().unwrap_or(""));

    // HTML export
    let html_data = dashboard.generate_html()?;
    println!("  HTML export size: {} characters", html_data.len());

    // Data filtering examples
    let all_metrics = dashboard.get_all_metrics()?;
    let first_timestamp = all_metrics.first().map(|m| m.timestamp).unwrap_or(0);
    let last_timestamp = all_metrics.last().map(|m| m.timestamp).unwrap_or(0);

    println!("\nData filtering examples:");
    println!("  Total data points: {}", all_metrics.len());
    println!("  Time range: {} to {}", first_timestamp, last_timestamp);

    // Get metrics in time range (first half)
    let mid_timestamp = first_timestamp + (last_timestamp - first_timestamp) / 2;
    let first_half = dashboard.get_metrics_in_range(first_timestamp, mid_timestamp)?;
    println!("  First half data points: {}", first_half.len());

    // Get specific metric data
    let accuracy_data = dashboard.get_metrics_by_name("batch_accuracy")?;
    println!("  Accuracy data points: {}", accuracy_data.len());

    // Export to files (in a real application)
    // export_dashboard_to_file(&dashboard, "dashboard_data.json", ExportFormat::Json)?;
    // export_dashboard_to_file(&dashboard, "dashboard_data.csv", ExportFormat::Csv)?;
    // export_dashboard_to_file(&dashboard, "dashboard.html", ExportFormat::Html)?;

    println!("  Export completed (files would be saved in real application)");

    // Data management
    println!("\nData management:");
    let stats_before = dashboard.get_statistics()?;
    println!(
        "  Data points before clear: {}",
        stats_before.total_data_points
    );

    // In a real application, you might clear old data periodically
    // dashboard.clear_data()?;
    // let stats_after = dashboard.get_statistics()?;
    // println!("  Data points after clear: {}", stats_after.total_data_points);

    Ok(())
}

/// Example of advanced dashboard configuration
fn advanced_configuration_example() -> Result<()> {
    // Create custom theme
    let custom_theme = DashboardTheme {
        primary_color: "#6366f1".to_string(),
        background_color: "#1f2937".to_string(),
        text_color: "#f9fafb".to_string(),
        chart_colors: vec![
            "#6366f1".to_string(),
            "#ec4899".to_string(),
            "#10b981".to_string(),
            "#f59e0b".to_string(),
            "#ef4444".to_string(),
        ],
    };

    // Advanced configuration
    let config = DashboardConfig {
        address: "0.0.0.0:8082".parse().unwrap(),
        refresh_interval: 1,   // Fast refresh
        max_data_points: 5000, // Large buffer
        enable_realtime: true,
        title: "Advanced ML Monitoring Dashboard".to_string(),
        theme: custom_theme,
    };

    let dashboard = InteractiveDashboard::new(config.clone());

    println!("Advanced dashboard configuration:");
    println!("  Address: {}", config.address);
    println!("  Refresh interval: {} seconds", config.refresh_interval);
    println!("  Max data points: {}", config.max_data_points);
    println!("  Real-time enabled: {}", config.enable_realtime);
    println!("  Theme primary color: {}", config.theme.primary_color);

    // Add metrics with complex metadata
    let experiments = ["baseline", "experiment_a", "experiment_b", "experiment_c"];
    let models = ["linear", "tree", "neural", "ensemble"];

    for (exp_idx, experiment) in experiments.iter().enumerate() {
        for (model_idx, model) in models.iter().enumerate() {
            let mut metadata = HashMap::new();
            metadata.insert("experiment".to_string(), experiment.to_string());
            metadata.insert("model".to_string(), model.to_string());
            metadata.insert("fold".to_string(), "1".to_string());

            // Simulate different performance for different combinations
            let base_perf = 0.7 + (exp_idx as f64 * 0.05) + (model_idx as f64 * 0.03);
            let noise = ((exp_idx + model_idx) as f64 * 0.3).sin() * 0.02;

            let accuracy = (base_perf + noise).clamp(0.0, 1.0);
            let f1_score = accuracy - 0.02 + noise * 0.5;

            dashboard.add_metric_with_metadata(
                &format!("{}_accuracy", experiment),
                accuracy,
                metadata.clone(),
            )?;

            dashboard.add_metric_with_metadata(
                &format!("{}_f1", experiment),
                f1_score,
                metadata,
            )?;
        }
    }

    let stats = dashboard.get_statistics()?;
    println!(
        "  Generated {} data points across {} unique metrics",
        stats.total_data_points, stats.unique_metrics
    );

    // Generate and show HTML preview
    let html = dashboard.generate_html()?;
    let lines: Vec<&str> = html.lines().collect();
    println!("  Generated HTML dashboard ({} lines)", lines.len());

    // Show a sample of the HTML
    println!("  HTML preview (first few lines):");
    for line in lines.iter().take(10) {
        if !line.trim().is_empty() {
            println!("    {}", line.trim());
        }
    }

    Ok(())
}

/// Example of widget system usage
fn widget_system_example() -> Result<()> {
    println!("Creating dashboard widgets...");

    // Create different types of widgets
    let accuracy_chart = DashboardWidget::line_chart(
        "accuracy_timeline".to_string(),
        "Model Accuracy Over Time".to_string(),
        vec!["train_accuracy".to_string(), "val_accuracy".to_string()],
    )
    .with_config("line_style".to_string(), "smooth".to_string())
    .with_config("y_axis_range".to_string(), "0.0-1.0".to_string());

    let loss_chart = DashboardWidget::line_chart(
        "loss_timeline".to_string(),
        "Training Loss".to_string(),
        vec!["train_loss".to_string(), "val_loss".to_string()],
    )
    .with_config("y_axis_scale".to_string(), "log".to_string());

    let current_accuracy_gauge = DashboardWidget::gauge(
        "current_acc".to_string(),
        "Current Accuracy".to_string(),
        "val_accuracy".to_string(),
    )
    .with_config("min_value".to_string(), "0".to_string())
    .with_config("max_value".to_string(), "1".to_string())
    .with_config("threshold_good".to_string(), "0.9".to_string())
    .with_config("threshold_warning".to_string(), "0.7".to_string());

    let metrics_table = DashboardWidget::table(
        "metrics_summary".to_string(),
        "Metrics Summary".to_string(),
        vec![
            "accuracy".to_string(),
            "precision".to_string(),
            "recall".to_string(),
            "f1_score".to_string(),
        ],
    )
    .with_config("sort_by".to_string(), "timestamp".to_string())
    .with_config("show_latest_only".to_string(), "true".to_string());

    // Display widget information
    let widgets = vec![
        accuracy_chart,
        loss_chart,
        current_accuracy_gauge,
        metrics_table,
    ];

    for widget in &widgets {
        println!("  Widget: {}", widget.title);
        println!("    ID: {}", widget.id);
        println!("    Type: {:?}", widget.widget_type);
        println!("    Metrics: {:?}", widget.metrics);
        println!("    Config: {:?}", widget.config);
        println!();
    }

    // Create dashboard and add sample data for widgets
    let dashboard = InteractiveDashboard::default();

    // Add sample time series data
    for i in 0..50 {
        let time_step = i as f64 * 0.1;
        let train_acc = 0.6 + (time_step * 0.1) + (time_step).sin() * 0.05;
        let val_acc = train_acc - 0.05 + (time_step * 1.2).cos() * 0.03;
        let train_loss = 2.0 - (time_step * 0.08) + (time_step * 0.5).sin().abs() * 0.1;
        let val_loss = train_loss + 0.1 + (time_step * 0.7).cos().abs() * 0.05;

        dashboard.add_metric("train_accuracy", train_acc)?;
        dashboard.add_metric("val_accuracy", val_acc)?;
        dashboard.add_metric("train_loss", train_loss)?;
        dashboard.add_metric("val_loss", val_loss)?;
    }

    // Add final summary metrics
    dashboard.add_metric("accuracy", 0.92)?;
    dashboard.add_metric("precision", 0.89)?;
    dashboard.add_metric("recall", 0.94)?;
    dashboard.add_metric("f1_score", 0.91)?;

    let stats = dashboard.get_statistics()?;
    println!("Widget dashboard data:");
    println!("  Total data points: {}", stats.total_data_points);
    println!("  Latest metric values:");
    for (metric, value) in &stats.latest_values {
        println!("    {}: {:.4}", metric, value);
    }

    Ok(())
}
