//! Visual analysis of FFT benchmarks
//!
//! This example creates visual comparisons of performance, memory usage,
//! and accuracy between different FFT implementations.

use plotly::common::{Mode, Title};
use plotly::layout::{Axis, AxisType, Layout};
use plotly::{Plot, Scatter};
use std::collections::HashMap;
use std::error::Error;
use std::fs;

/// Benchmark data structure
#[derive(Debug)]
struct BenchmarkData {
    operation: String,
    size: usize,
    time_ms: f64,
    memory_kb: Option<f64>,
    accuracy: Option<f64>,
}

/// Parse benchmark results from files
fn parse_benchmark_results() -> Result<Vec<BenchmarkData>, Box<dyn Error>> {
    let mut results = Vec::new();

    // This is a placeholder - in practice, you would parse actual benchmark output files
    // For demonstration, let's create some synthetic data
    let operations = vec!["fft", "rfft", "fft2", "frft"];
    let sizes = vec![64, 256, 1024, 4096, 16384];

    for op in &operations {
        for &size in &sizes {
            results.push(BenchmarkData {
                operation: op.to_string(),
                size,
                time_ms: (size as f64).log2() * 0.1 * rand::random::<f64>(),
                memory_kb: Some((size as f64) * 0.008 * (1.0 + rand::random::<f64>())),
                accuracy: Some(1e-10 * (1.0 + rand::random::<f64>())),
            });
        }
    }

    Ok(results)
}

/// Create performance comparison plot
fn create_performance_plot(data: &[BenchmarkData]) -> Result<(), Box<dyn Error>> {
    let mut plot = Plot::new();

    // Group data by operation
    let mut grouped: HashMap<String, Vec<&BenchmarkData>> = HashMap::new();
    for item in data {
        grouped
            .entry(item.operation.clone())
            .or_default()
            .push(item);
    }

    // Create traces for each operation
    for (operation, items) in grouped {
        let mut sizes = Vec::new();
        let mut times = Vec::new();

        for item in items {
            sizes.push(item.size as f64);
            times.push(item.time_ms);
        }

        let trace = Scatter::new(sizes, times)
            .mode(Mode::LinesMarkers)
            .name(&operation);
        plot.add_trace(trace);
    }

    let layout = Layout::new()
        .title(Title::with_text("FFT Performance Comparison"))
        .x_axis(
            Axis::new()
                .title(Title::with_text("Input Size"))
                .type_(AxisType::Log),
        )
        .y_axis(
            Axis::new()
                .title(Title::with_text("Time (ms)"))
                .type_(AxisType::Log),
        );

    plot.set_layout(layout);
    plot.write_html("benchmark_results/performance_comparison.html");

    Ok(())
}

/// Create memory usage plot
fn create_memory_plot(data: &[BenchmarkData]) -> Result<(), Box<dyn Error>> {
    let mut plot = Plot::new();

    // Group data by operation
    let mut grouped: HashMap<String, Vec<&BenchmarkData>> = HashMap::new();
    for item in data {
        if item.memory_kb.is_some() {
            grouped
                .entry(item.operation.clone())
                .or_default()
                .push(item);
        }
    }

    // Create traces for each operation
    for (operation, items) in grouped {
        let mut sizes = Vec::new();
        let mut memory = Vec::new();

        for item in items {
            sizes.push(item.size as f64);
            memory.push(item.memory_kb.unwrap());
        }

        let trace = Scatter::new(sizes, memory)
            .mode(Mode::LinesMarkers)
            .name(&operation);
        plot.add_trace(trace);
    }

    let layout = Layout::new()
        .title(Title::with_text("FFT Memory Usage Comparison"))
        .x_axis(
            Axis::new()
                .title(Title::with_text("Input Size"))
                .type_(AxisType::Log),
        )
        .y_axis(
            Axis::new()
                .title(Title::with_text("Memory (KB)"))
                .type_(AxisType::Log),
        );

    plot.set_layout(layout);
    plot.write_html("benchmark_results/memory_comparison.html");

    Ok(())
}

/// Create accuracy comparison plot
fn create_accuracy_plot(data: &[BenchmarkData]) -> Result<(), Box<dyn Error>> {
    let mut plot = Plot::new();

    // Group data by operation
    let mut grouped: HashMap<String, Vec<&BenchmarkData>> = HashMap::new();
    for item in data {
        if item.accuracy.is_some() {
            grouped
                .entry(item.operation.clone())
                .or_default()
                .push(item);
        }
    }

    // Create traces for each operation
    for (operation, items) in grouped {
        let mut sizes = Vec::new();
        let mut accuracy = Vec::new();

        for item in items {
            sizes.push(item.size as f64);
            accuracy.push(item.accuracy.unwrap());
        }

        let trace = Scatter::new(sizes, accuracy)
            .mode(Mode::LinesMarkers)
            .name(&operation);
        plot.add_trace(trace);
    }

    let layout = Layout::new()
        .title(Title::with_text("FFT Accuracy Comparison"))
        .x_axis(
            Axis::new()
                .title(Title::with_text("Input Size"))
                .type_(AxisType::Log),
        )
        .y_axis(
            Axis::new()
                .title(Title::with_text("Error"))
                .type_(AxisType::Log),
        );

    plot.set_layout(layout);
    plot.write_html("benchmark_results/accuracy_comparison.html");

    Ok(())
}

/// Create comparison table
fn create_comparison_table(data: &[BenchmarkData]) -> Result<(), Box<dyn Error>> {
    let mut html = String::from(
        r#"
<!DOCTYPE html>
<html>
<head>
    <title>FFT Benchmark Comparison Table</title>
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: right;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .operation {
            text-align: left;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>FFT Benchmark Comparison</h1>
    <table>
        <tr>
            <th>Operation</th>
            <th>Size</th>
            <th>Time (ms)</th>
            <th>Memory (KB)</th>
            <th>Error</th>
        </tr>
"#,
    );

    for item in data {
        html.push_str(&format!(
            "<tr>
                <td class='operation'>{}</td>
                <td>{}</td>
                <td>{:.3}</td>
                <td>{}</td>
                <td>{}</td>
            </tr>\n",
            item.operation,
            item.size,
            item.time_ms,
            item.memory_kb
                .map(|m| format!("{:.1}", m))
                .unwrap_or_else(|| "N/A".to_string()),
            item.accuracy
                .map(|a| format!("{:.2e}", a))
                .unwrap_or_else(|| "N/A".to_string())
        ));
    }

    html.push_str("</table>\n</body>\n</html>");

    fs::write("benchmark_results/comparison_table.html", html)?;

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Create results directory
    fs::create_dir_all("benchmark_results")?;

    // Parse benchmark results
    let data = parse_benchmark_results()?;

    // Create visualizations
    println!("Creating performance comparison plot...");
    create_performance_plot(&data)?;

    println!("Creating memory usage plot...");
    create_memory_plot(&data)?;

    println!("Creating accuracy comparison plot...");
    create_accuracy_plot(&data)?;

    println!("Creating comparison table...");
    create_comparison_table(&data)?;

    println!("\nBenchmark analysis complete!");
    println!("Results saved in benchmark_results/");
    println!("- performance_comparison.html");
    println!("- memory_comparison.html");
    println!("- accuracy_comparison.html");
    println!("- comparison_table.html");

    Ok(())
}
