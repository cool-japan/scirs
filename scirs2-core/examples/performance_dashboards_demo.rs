//! # Performance Dashboards System Demo
//!
//! This example demonstrates the comprehensive performance dashboards system
//! with real-time visualization and historical trend analysis for enterprise environments.

use scirs2_core::profiling::dashboards::{
    AlertCondition, AlertConfig, AlertSeverity, ChartType, DashboardConfig, DashboardTheme,
    MetricSource, NotificationChannel, PerformanceDashboard, Widget,
};
use scirs2_core::CoreResult;
use std::thread;
use std::time::Duration;

#[allow(dead_code)]
fn main() -> CoreResult<()> {
    println!("📊 SciRS2 Core Performance Dashboards Demo");
    println!("==========================================\n");

    // Configuration examples
    demo_dashboard_configurations()?;
    println!();

    // Basic dashboard setup
    demo_basic_dashboard_setup()?;
    println!();

    // Widget creation and management
    demo_widget_management()?;
    println!();

    // Real-time metrics and alerts
    demo_real_time_metrics_and_alerts()?;
    println!();

    // Dashboard themes and customization
    demo_dashboard_customization()?;
    println!();

    // Export and import functionality
    demo_export_import()?;

    println!("\n✨ Performance dashboards demo completed successfully!");
    println!("\nThe performance dashboards system provides:");
    println!("  🔹 Real-time performance visualization with live updates");
    println!("  🔹 Historical trend analysis with configurable retention");
    println!("  🔹 Customizable widgets with multiple chart types");
    println!("  🔹 Intelligent alerting with threshold-based notifications");
    println!("  🔹 Multi-dimensional metrics aggregation and filtering");
    println!("  🔹 Export/import capabilities for dashboard configurations");
    println!("  🔹 Web-based interface with REST API support");
    println!("  🔹 Mobile-responsive design for monitoring on-the-go");

    Ok(())
}

#[allow(dead_code)]
fn demo_dashboard_configurations() -> CoreResult<()> {
    println!("⚙️  Dashboard Configuration Examples");
    println!("-----------------------------------");

    // Production environment dashboard
    let production_config = DashboardConfig::default()
        .with_title("Production System Monitor")
        .with_refresh_interval(Duration::from_secs(30))
        .with_retention_period(Duration::from_secs(30 * 24 * 60 * 60))
        .with_web_interface(8080)
        .with_api_token("prod-dashboard-token")
        .with_theme(DashboardTheme::Dark);

    println!("🏭 Production Configuration:");
    println!("  - Title: {}", production_config.title);
    println!(
        "  - Refresh: {}s",
        production_config.refresh_interval.as_secs()
    );
    println!(
        "  - Retention: {} days",
        production_config.retention_period.as_secs() / 86400
    );
    println!("  - Web Port: {}", production_config.web_port);
    println!("  - Theme: {:?}", production_config.theme);
    println!("  - Max Data Points: {}", production_config.max_data_points);

    // Development environment dashboard
    let dev_config = DashboardConfig::default()
        .with_title("Development Dashboard")
        .with_refresh_interval(Duration::from_secs(5))
        .with_retention_period(Duration::from_secs(7 * 24 * 60 * 60))
        .with_web_interface(3000)
        .with_theme(DashboardTheme::Light);

    println!("\n🔧 Development Configuration:");
    println!("  - Title: {}", dev_config.title);
    println!("  - Refresh: {}s", dev_config.refresh_interval.as_secs());
    println!(
        "  - Retention: {} days",
        dev_config.retention_period.as_secs() / 86400
    );
    println!("  - Web Port: {}", dev_config.web_port);
    println!("  - Theme: {:?}", dev_config.theme);
    println!("  - Alerts Enabled: {}", dev_config.enable_alerts);

    // Monitoring dashboard
    let monitoring_config = DashboardConfig::default()
        .with_title("System Monitoring Dashboard")
        .with_refresh_interval(Duration::from_secs(10))
        .with_retention_period(Duration::from_secs(14 * 24 * 60 * 60))
        .with_web_interface(9090)
        .with_theme(DashboardTheme::HighContrast);

    println!("\n📊 Monitoring Configuration:");
    println!("  - Title: {}", monitoring_config.title);
    println!(
        "  - Refresh: {}s",
        monitoring_config.refresh_interval.as_secs()
    );
    println!("  - API Enabled: {}", monitoring_config.enable_rest_api);
    println!(
        "  - Auto-save: {}s",
        monitoring_config.auto_save_interval.as_secs()
    );

    Ok(())
}

#[allow(dead_code)]
fn demo_basic_dashboard_setup() -> CoreResult<()> {
    println!("🚀 Basic Dashboard Setup");
    println!("------------------------");

    // Create dashboard with custom configuration
    let config = DashboardConfig::default()
        .with_title("SciRS2 Performance Monitor")
        .with_refresh_interval(Duration::from_secs(5))
        .with_retention_period(Duration::from_secs(7 * 24 * 60 * 60));

    let mut dashboard = PerformanceDashboard::new(config)?;
    println!("✅ Dashboard created: 'SciRS2 Performance Monitor'");

    // Add basic system monitoring widgets
    let cpu_widget = Widget::new()
        .with_title("CPU Usage (%)")
        .with_chart_type(ChartType::LineChart)
        .with_metric_source(MetricSource::SystemCpu)
        .with_layout(0, 0, 6, 3)
        .with_alert_threshold(80.0)
        .with_colors(vec!["#007acc", "#0099ff"]);

    let cpu_widget_id = dashboard.add_widget(cpu_widget)?;
    println!("📈 Added CPU usage widget (ID: {})", cpu_widget_id);

    let memory_widget = Widget::new()
        .with_title("Memory Usage (MB)")
        .with_chart_type(ChartType::AreaChart)
        .with_metric_source(MetricSource::SystemMemory)
        .with_layout(6, 0, 6, 3)
        .with_alert_threshold(85.0)
        .with_colors(vec!["#ff6b35", "#ff8c42"]);

    let memory_widget_id = dashboard.add_widget(memory_widget)?;
    println!("💾 Added memory usage widget (ID: {})", memory_widget_id);

    let network_widget = Widget::new()
        .with_title("Network I/O (KB/s)")
        .with_chart_type(ChartType::AreaChart)
        .with_metric_source(MetricSource::NetworkIO)
        .with_layout(0, 3, 12, 3);

    let network_widget_id = dashboard.add_widget(network_widget)?;
    println!("🌐 Added network I/O widget (ID: {})", network_widget_id);

    // Start the dashboard
    dashboard.start()?;
    println!("🚀 Dashboard started!");

    // Get initial statistics
    let stats = dashboard.get_statistics();
    println!("\n📊 Dashboard Statistics:");
    println!("  - Total Widgets: {}", stats.total_widgets);
    println!("  - Total Metrics: {}", stats.total_metrics);
    println!("  - Active Alerts: {}", stats.active_alerts);
    println!("  - State: {:?}", stats.state);

    // Simulate some metric updates
    println!("\n📊 Simulating metric updates...");
    for i in 0..10 {
        let cpu_value = 45.0 + (i as f64 * 3.5); // Gradually increasing CPU
        let memory_value = 2048.0 + (i as f64 * 128.0); // Gradually increasing memory
        let network_value = 100.0 + (i as f64 * 50.0); // Gradually increasing network

        dashboard.update_metric(&MetricSource::SystemCpu, cpu_value)?;
        dashboard.update_metric(&MetricSource::SystemMemory, memory_value)?;
        dashboard.update_metric(&MetricSource::NetworkIO, network_value)?;

        if i % 3 == 0 {
            println!(
                "  📈 Update {}: CPU={:.1}%, Memory={:.0}MB, Network={:.0}KB/s",
                i + 1,
                cpu_value,
                memory_value,
                network_value
            );
        }

        thread::sleep(Duration::from_millis(100));
    }

    // Stop the dashboard
    dashboard.stop()?;
    println!("🛑 Dashboard stopped");

    Ok(())
}

#[allow(dead_code)]
fn demo_widget_management() -> CoreResult<()> {
    println!("🎛️  Widget Management Demo");
    println!("-------------------------");

    let config = DashboardConfig::default().with_title("Widget Management Dashboard");

    let mut dashboard = PerformanceDashboard::new(config)?;

    // Demonstrate different chart types
    let chart_types = [
        (
            ChartType::LineChart,
            "Response Time",
            MetricSource::Application("api_response".to_string()),
        ),
        (
            ChartType::BarChart,
            "Request Count",
            MetricSource::Application("request_count".to_string()),
        ),
        (
            ChartType::GaugeChart,
            "Error Rate",
            MetricSource::Application("error_rate".to_string()),
        ),
        (
            ChartType::PieChart,
            "Status Distribution",
            MetricSource::Application("status_dist".to_string()),
        ),
        (
            ChartType::Heatmap,
            "Request Heatmap",
            MetricSource::Application("request_heatmap".to_string()),
        ),
        (
            ChartType::Histogram,
            "Latency Distribution",
            MetricSource::Application("latency_hist".to_string()),
        ),
    ];

    println!("📊 Adding widgets with different chart types:");
    let mut widget_ids = Vec::new();

    for (i, (chart_type, title, metric_source)) in chart_types.iter().enumerate() {
        let x = (i % 3) as u32 * 4;
        let y = (i / 3) as u32 * 3;

        let widget = Widget::new()
            .with_title(title)
            .with_chart_type(*chart_type)
            .with_metric_source(metric_source.clone())
            .with_layout(x, y, 4, 3)
            .with_refresh_interval(Duration::from_secs(2));

        let widget_id = dashboard.add_widget(widget)?;
        widget_ids.push(widget_id.clone());

        println!("  📈 {} ({:?}) - ID: {}", title, chart_type, widget_id);
    }

    // Demonstrate widget removal
    println!("\n🗑️  Removing widgets:");
    for (i, widget_id) in widget_ids.iter().take(2).enumerate() {
        dashboard.remove_widget(widget_id)?;
        println!("  ❌ Removed widget {} (ID: {})", i + 1, widget_id);
    }

    let final_stats = dashboard.get_statistics();
    println!("\n📊 Final widget count: {}", final_stats.total_widgets);

    Ok(())
}

#[allow(dead_code)]
fn demo_real_time_metrics_and_alerts() -> CoreResult<()> {
    println!("🚨 Real-time Metrics and Alerts Demo");
    println!("------------------------------------");

    let config = DashboardConfig::default()
        .with_title("Alert Monitoring Dashboard")
        .with_refresh_interval(Duration::from_secs(1));

    let mut dashboard = PerformanceDashboard::new(config)?;

    // Add widgets with different alert configurations
    let cpu_widget = Widget::new()
        .with_title("CPU Usage Monitor")
        .with_chart_type(ChartType::GaugeChart)
        .with_metric_source(MetricSource::SystemCpu)
        .with_alert_threshold(75.0); // Will trigger alert when CPU > 75%

    let memory_widget = Widget::new()
        .with_title("Memory Usage Monitor")
        .with_chart_type(ChartType::LineChart)
        .with_metric_source(MetricSource::SystemMemory)
        .with_alert_threshold(2000.0); // Will trigger alert when Memory > 2000MB

    let custom_alert_config = AlertConfig {
        threshold: 50.0,
        condition: AlertCondition::GreaterThan,
        severity: AlertSeverity::Critical,
        notification_channels: vec![
            NotificationChannel::Console,
            NotificationChannel::File("/tmp/dashboard_alerts.log".to_string()),
        ],
        cooldown_period: Duration::from_secs(60),
    };

    let error_rate_widget = Widget::new()
        .with_title("Error Rate Monitor")
        .with_chart_type(ChartType::LineChart)
        .with_metric_source(MetricSource::Application("error_rate".to_string()));

    // Manually set alert config for error rate widget
    let mut error_widget = error_rate_widget;
    error_widget.alert_config = Some(custom_alert_config);

    dashboard.add_widget(cpu_widget)?;
    dashboard.add_widget(memory_widget)?;
    dashboard.add_widget(error_widget)?;

    dashboard.start()?;
    println!("✅ Alert monitoring dashboard started");

    // Simulate metrics that will trigger alerts
    println!("\n📊 Simulating metrics with alert triggers:");

    let scenarios = [
        (
            "Normal Operation",
            vec![
                (MetricSource::SystemCpu, 45.0),
                (MetricSource::SystemMemory, 1500.0),
                (MetricSource::Application("error_rate".to_string()), 2.5),
            ],
        ),
        (
            "High CPU Load",
            vec![
                (MetricSource::SystemCpu, 85.0), // Will trigger alert
                (MetricSource::SystemMemory, 1600.0),
                (MetricSource::Application("error_rate".to_string()), 3.2),
            ],
        ),
        (
            "Memory Pressure",
            vec![
                (MetricSource::SystemCpu, 65.0),
                (MetricSource::SystemMemory, 2500.0), // Will trigger alert
                (MetricSource::Application("error_rate".to_string()), 4.1),
            ],
        ),
        (
            "Critical Error Rate",
            vec![
                (MetricSource::SystemCpu, 55.0),
                (MetricSource::SystemMemory, 1800.0),
                (MetricSource::Application("error_rate".to_string()), 75.0), // Will trigger critical alert
            ],
        ),
        (
            "Recovery",
            vec![
                (MetricSource::SystemCpu, 35.0),
                (MetricSource::SystemMemory, 1400.0),
                (MetricSource::Application("error_rate".to_string()), 1.8),
            ],
        ),
    ];

    for (scenario_name, metrics) in &scenarios {
        println!("\n🎯 Scenario: {}", scenario_name);

        for (metric_source, value) in metrics {
            dashboard.update_metric(metric_source, *value)?;
            println!("  📊 Updated {:?}: {:.1}", metric_source, value);
        }

        // Check alert status
        let stats = dashboard.get_statistics();
        if stats.active_alerts > 0 {
            println!("  🚨 Active alerts: {}", stats.active_alerts);
        } else {
            println!("  ✅ No active alerts");
        }

        thread::sleep(Duration::from_millis(500));
    }

    dashboard.stop()?;
    println!("\n🛑 Alert monitoring completed");

    Ok(())
}

#[allow(dead_code)]
fn demo_dashboard_customization() -> CoreResult<()> {
    println!("🎨 Dashboard Customization Demo");
    println!("-------------------------------");

    // Demonstrate different themes
    let themes = [
        DashboardTheme::Light,
        DashboardTheme::Dark,
        DashboardTheme::HighContrast,
        DashboardTheme::Custom,
    ];

    for theme in &themes {
        let config = DashboardConfig::default()
            .with_title(&format!("{:?} Theme Dashboard", theme))
            .with_theme(*theme);

        println!("🎨 Theme: {:?}", theme);
        println!("  - Title: {}", config.title);
        println!("  - Web Interface: {}", config.enable_web_interface);
        println!("  - API Enabled: {}", config.enable_rest_api);
    }

    // Demonstrate custom widget styling
    println!("\n🎨 Custom Widget Styling:");

    let config = DashboardConfig::default()
        .with_title("Customized Dashboard")
        .with_theme(DashboardTheme::Custom);

    let mut dashboard = PerformanceDashboard::new(config)?;

    // Performance widget with custom colors
    let perf_widget = Widget::new()
        .with_title("Performance Metrics")
        .with_chart_type(ChartType::LineChart)
        .with_metric_source(MetricSource::Application("performance".to_string()))
        .with_layout(0, 0, 8, 4)
        .with_colors(vec![
            "#2E8B57", // Sea Green
            "#4682B4", // Steel Blue
            "#D2691E", // Chocolate
            "#8A2BE2", // Blue Violet
            "#FF6347", // Tomato
        ]);

    // Status widget with gauge visualization
    let status_widget = Widget::new()
        .with_title("System Health")
        .with_chart_type(ChartType::GaugeChart)
        .with_metric_source(MetricSource::Custom("health_score".to_string()))
        .with_layout(8, 0, 4, 4)
        .with_colors(vec!["#32CD32", "#FFD700", "#FF4500"]); // Green, Gold, Red

    dashboard.add_widget(perf_widget)?;
    dashboard.add_widget(status_widget)?;

    println!("  ✅ Added performance widget with sea green color scheme");
    println!("  ✅ Added status widget with health-based colors");

    // Demonstrate number formatting options
    println!("\n🔢 Number Formatting Examples:");
    println!("  - Auto: 1234.56 → 1.23K");
    println!("  - Integer: 1234.56 → 1235");
    println!("  - Decimal(2): 1234.56 → 1234.56");
    println!("  - Percentage: 0.85 → 85%");
    println!("  - Scientific: 1234.56 → 1.23e+3");
    println!("  - Bytes: 1048576 → 1.0 MB");

    Ok(())
}

#[allow(dead_code)]
fn demo_export_import() -> CoreResult<()> {
    println!("💾 Export/Import Functionality Demo");
    println!("-----------------------------------");

    // Create a dashboard with some widgets
    let config = DashboardConfig::default()
        .with_title("Export Test Dashboard")
        .with_refresh_interval(Duration::from_secs(15))
        .with_retention_period(Duration::from_secs(14 * 24 * 60 * 60));

    let mut original_dashboard = PerformanceDashboard::new(config)?;

    // Add multiple widgets
    let widgets_to_add = [
        ("CPU Usage", ChartType::LineChart, MetricSource::SystemCpu),
        (
            "Memory Usage",
            ChartType::AreaChart,
            MetricSource::SystemMemory,
        ),
        ("Network I/O", ChartType::BarChart, MetricSource::NetworkIO),
        (
            "Error Rate",
            ChartType::GaugeChart,
            MetricSource::Application("errors".to_string()),
        ),
    ];

    for (title, chart_type, metric_source) in &widgets_to_add {
        let widget = Widget::new()
            .with_title(title)
            .with_chart_type(*chart_type)
            .with_metric_source(metric_source.clone())
            .with_alert_threshold(80.0);

        original_dashboard.add_widget(widget)?;
    }

    println!("📊 Created dashboard with {} widgets", widgets_to_add.len());

    // Export dashboard configuration
    println!("\n📤 Exporting dashboard configuration...");
    let exported_config = original_dashboard.export_config()?;

    println!("✅ Export completed");
    println!("  - Configuration size: {} bytes", exported_config.len());

    // Show a preview of the exported data
    let preview = if exported_config.len() > 200 {
        format!("{}...[truncated]", &exported_config[..200])
    } else {
        exported_config.clone()
    };
    println!("  - Preview: {}", preview);

    // Create a new dashboard and import the configuration
    println!("\n📥 Importing configuration to new dashboard...");
    let new_config = DashboardConfig::default().with_title("Imported Dashboard");

    let mut imported_dashboard = PerformanceDashboard::new(new_config)?;

    // Import the configuration
    imported_dashboard.import_configuration(&exported_config)?;

    let imported_stats = imported_dashboard.get_statistics();
    println!("✅ Import completed");
    println!("  - Imported widgets: {}", imported_stats.total_widgets);

    // Verify the import was successful
    if imported_stats.total_widgets == widgets_to_add.len() {
        println!("✅ Import verification successful - widget count matches");
    } else {
        println!("❌ Import verification failed - widget count mismatch");
    }

    println!("\n💾 Export/Import Features:");
    println!("  🔹 Complete dashboard configuration export");
    println!("  🔹 Widget layout and styling preservation");
    println!("  🔹 Alert configuration backup and restore");
    println!("  🔹 Cross-environment deployment support");
    println!("  🔹 Version control friendly JSON format");

    Ok(())
}

#[allow(dead_code)]
fn metric_source_name(source: &MetricSource) -> String {
    match source {
        MetricSource::SystemCpu => "System CPU".to_string(),
        MetricSource::SystemMemory => "System Memory".to_string(),
        MetricSource::NetworkIO => "Network I/O".to_string(),
        MetricSource::DiskIO => "Disk I/O".to_string(),
        MetricSource::Application(name) => format!("App: {}", name),
        MetricSource::Custom(name) => format!("Custom: {}", name),
        MetricSource::Database(name) => format!("DB: {}", name),
        MetricSource::Cache(name) => format!("Cache: {}", name),
    }
}
