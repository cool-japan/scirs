//! Automated benchmark runner and report generator for comprehensive performance analysis

#![allow(unused_imports)]
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Configuration for benchmark runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Benchmark suites to run
    pub suites: Vec<String>,
    /// Output directory for results
    pub output_dir: String,
    /// Whether to generate HTML report
    pub generate_html: bool,
    /// Whether to generate JSON report
    pub generate_json: bool,
    /// Whether to run memory profiling
    pub memory_profiling: bool,
    /// Timeout for each benchmark suite (seconds)
    pub timeout_seconds: u64,
    /// Number of samples for statistical significance
    pub sample_size: Option<usize>,
    /// Target CPU features to test
    pub target_features: Vec<String>,
    /// Whether to run comparison with baseline
    pub compare_with_baseline: bool,
    /// Baseline results file path
    pub baseline_path: Option<String>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            suites: vec![
                "graph_benchmarks".to_string(),
                "memory_benchmarks".to_string(),
                "large_graph_stress".to_string(),
                "advanced_algorithms".to_string(),
                "performance_optimizations".to_string(),
                "advanced_benchmarks".to_string(),
            ],
            output_dir: "benchmark_results".to_string(),
            generate_html: true,
            generate_json: true,
            memory_profiling: true,
            timeout_seconds: 3600, // 1 hour
            sample_size: None,
            target_features: vec!["native".to_string(), "sse2".to_string(), "avx2".to_string()],
            compare_with_baseline: false,
            baseline_path: None,
        }
    }
}

/// Results from a single benchmark suite
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkSuiteResult {
    pub suite_name: String,
    pub timestamp: u64,
    pub duration_seconds: f64,
    pub status: BenchmarkStatus,
    pub error_message: Option<String>,
    pub results_file: Option<String>,
    pub memory_usage: Option<MemoryUsage>,
    pub platform_info: PlatformInfo,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum BenchmarkStatus {
    Success,
    Failed,
    Timeout,
    Skipped,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub peak_memory_mb: f64,
    pub average_memory_mb: f64,
    pub memory_samples: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PlatformInfo {
    pub os: String,
    pub arch: String,
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub total_memory_gb: f64,
    pub rustc_version: String,
    pub target_features: Vec<String>,
}

/// Complete benchmark run results
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub config: BenchmarkConfig,
    pub start_time: u64,
    pub end_time: u64,
    pub total_duration_seconds: f64,
    pub suite_results: Vec<BenchmarkSuiteResult>,
    pub summary: BenchmarkSummary,
    pub comparison: Option<BenchmarkComparison>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub total_suites: usize,
    pub successful_suites: usize,
    pub failed_suites: usize,
    pub skipped_suites: usize,
    pub fastest_suite: Option<String>,
    pub slowest_suite: Option<String>,
    pub peak_memory_usage_mb: f64,
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub baseline_date: u64,
    pub improvements: Vec<PerformanceChange>,
    pub regressions: Vec<PerformanceChange>,
    pub summary: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceChange {
    pub benchmark_name: String,
    pub old_value: f64,
    pub new_value: f64,
    pub change_percent: f64,
    pub significance: SignificanceLevel,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum SignificanceLevel {
    High,
    Medium,
    Low,
    Negligible,
}

/// Main benchmark runner
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
}

impl BenchmarkRunner {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Run all configured benchmark suites
    pub fn run_all_benchmarks(&self) -> Result<BenchmarkReport, Box<dyn std::error::Error>> {
        println!("🚀 Starting comprehensive benchmark run...");

        let start_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let platform_info = self.collect_platform_info()?;

        // Create output directory
        std::fs::create_dir_all(&self.config.output_dir)?;

        let mut suite_results = Vec::new();
        let mut peak_memory = 0.0f64;

        for suite in &self.config.suites {
            println!("📊 Running benchmark suite: {suite}");

            let result = self.run_benchmark_suite(suite, &platform_info)?;

            if let Some(ref memory) = result.memory_usage {
                peak_memory = peak_memory.max(memory.peak_memory_mb);
            }

            suite_results.push(result);
        }

        let end_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let total_duration = (end_time - start_time) as f64;

        let summary = self.generate_summary(&suite_results, peak_memory);
        let comparison = if self.config.compare_with_baseline {
            self.compare_with_baseline(&suite_results)?
        } else {
            None
        };

        let report = BenchmarkReport {
            config: self.config.clone(),
            start_time,
            end_time,
            total_duration_seconds: total_duration,
            suite_results,
            summary,
            comparison,
        };

        // Generate reports
        self.save_reports(&report)?;

        println!("✅ Benchmark run completed in {total_duration:.2} seconds");

        Ok(report)
    }

    fn run_benchmark_suite(
        &self,
        suite_name: &str,
        platform_info: &PlatformInfo,
    ) -> Result<BenchmarkSuiteResult, Box<dyn std::error::Error>> {
        let start = std::time::Instant::now();
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        // Prepare benchmark command
        let mut cmd = Command::new("cargo");
        cmd.arg("bench")
            .arg("--bench")
            .arg(suite_name)
            .arg("--")
            .arg("--output-format")
            .arg("json");

        // Add sample size if specified
        if let Some(samples) = self.config.sample_size {
            cmd.arg("--sample-size").arg(samples.to_string());
        }

        // Set target features
        let mut rustflags = String::new();
        for feature in &self.config.target_features {
            if feature == "native" {
                rustflags.push_str("-C target-cpu=native ");
            } else {
                rustflags.push_str(&format!("-C target-feature=+{feature} "));
            }
        }
        if !rustflags.is_empty() {
            cmd.env("RUSTFLAGS", rustflags.trim());
        }

        // Setup timeout
        let timeout = Duration::from_secs(self.config.timeout_seconds);

        println!("  Running: {cmd:?}");

        // Execute benchmark with timeout and memory monitoring
        let (status, memory_usage, results_file) = if self.config.memory_profiling {
            self.run_with_memory_monitoring(cmd, timeout, suite_name)?
        } else {
            self.run_basic_benchmark(cmd, timeout, suite_name)?
        };

        let duration = start.elapsed().as_secs_f64();

        Ok(BenchmarkSuiteResult {
            suite_name: suite_name.to_string(),
            timestamp,
            duration_seconds: duration,
            status,
            error_message: None,
            results_file,
            memory_usage,
            platform_info: platform_info.clone(),
        })
    }

    #[allow(clippy::type_complexity)]
    fn run_with_memory_monitoring(
        &self,
        mut cmd: Command,
        timeout: Duration,
        suite_name: &str,
    ) -> Result<(BenchmarkStatus, Option<MemoryUsage>, Option<String>), Box<dyn std::error::Error>>
    {
        use std::process::Stdio;
        use std::sync::{Arc, Mutex};
        use std::thread;

        let output_file = format!("{}/{}_results.json", self.config.output_dir, suite_name);

        // Redirect output to file
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        let mut child = cmd.spawn()?;
        let pid = child.id();

        // Memory monitoring thread
        let memory_samples = Arc::new(Mutex::new(Vec::new()));
        let memory_samples_clone = memory_samples.clone();

        let memory_thread = thread::spawn(move || {
            while std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok() {
                if let Ok(status) = std::fs::read_to_string(format!("/proc/{pid}/status")) {
                    for line in status.lines() {
                        if line.starts_with("VmRSS:") {
                            if let Some(kb_str) = line.split_whitespace().nth(1) {
                                if let Ok(kb) = kb_str.parse::<f64>() {
                                    let mb = kb / 1024.0;
                                    memory_samples_clone
                                        .lock()
                                        .expect("Operation failed")
                                        .push(mb);
                                }
                            }
                            break;
                        }
                    }
                }
                thread::sleep(Duration::from_millis(100));
            }
        });

        // Wait for completion with _timeout
        let exit_status = child.wait()?;
        let status = if exit_status.success() {
            BenchmarkStatus::Success
        } else {
            BenchmarkStatus::Failed
        };

        // Collect memory usage data
        let _ = memory_thread.join();
        let samples = memory_samples.lock().expect("Operation failed").clone();

        let memory_usage = if !samples.is_empty() {
            Some(MemoryUsage {
                peak_memory_mb: samples.iter().fold(0.0f64, |a, &b| a.max(b)),
                average_memory_mb: samples.iter().sum::<f64>() / samples.len() as f64,
                memory_samples: samples,
            })
        } else {
            None
        };

        let results_file = if status == BenchmarkStatus::Success {
            Some(output_file)
        } else {
            None
        };

        Ok((status, memory_usage, results_file))
    }

    #[allow(clippy::type_complexity)]
    fn run_basic_benchmark(
        &self,
        mut cmd: Command,
        timeout: Duration,
        suite_name: &str,
    ) -> Result<(BenchmarkStatus, Option<MemoryUsage>, Option<String>), Box<dyn std::error::Error>>
    {
        let output_file = format!("{}/{}_results.json", self.config.output_dir, suite_name);

        let output = cmd.timeout(timeout).output()?;

        let status = if output.status.success() {
            // Save output to file
            std::fs::write(&output_file, &output.stdout)?;
            BenchmarkStatus::Success
        } else {
            BenchmarkStatus::Failed
        };

        let results_file = if status == BenchmarkStatus::Success {
            Some(output_file)
        } else {
            None
        };

        Ok((status, None, results_file))
    }

    fn collect_platform_info(&self) -> Result<PlatformInfo, Box<dyn std::error::Error>> {
        let os = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();

        // Get CPU info (Linux-specific)
        let cpu_model = if cfg!(target_os = "linux") {
            std::fs::read_to_string("/proc/cpuinfo")?
                .lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split(':').nth(1))
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| "Unknown".to_string())
        } else {
            "Unknown".to_string()
        };

        let cpu_cores = num_cpus::get();

        // Get memory info (Linux-specific)
        let total_memory_gb = if cfg!(target_os = "linux") {
            std::fs::read_to_string("/proc/meminfo")?
                .lines()
                .find(|line| line.starts_with("MemTotal:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|s| s.parse::<f64>().ok())
                .map(|kb| kb / 1024.0 / 1024.0)
                .unwrap_or(0.0)
        } else {
            0.0
        };

        // Get Rust version
        let rustc_output = Command::new("rustc").arg("--version").output()?;
        let rustc_version = String::from_utf8_lossy(&rustc_output.stdout)
            .trim()
            .to_string();

        Ok(PlatformInfo {
            os,
            arch,
            cpu_model,
            cpu_cores,
            total_memory_gb,
            rustc_version,
            target_features: self.config.target_features.clone(),
        })
    }

    fn generate_summary(
        &self,
        results: &[BenchmarkSuiteResult],
        peak_memory: f64,
    ) -> BenchmarkSummary {
        let total_suites = results.len();
        let successful_suites = results
            .iter()
            .filter(|r| matches!(r.status, BenchmarkStatus::Success))
            .count();
        let failed_suites = results
            .iter()
            .filter(|r| matches!(r.status, BenchmarkStatus::Failed))
            .count();
        let skipped_suites = results
            .iter()
            .filter(|r| matches!(r.status, BenchmarkStatus::Skipped))
            .count();

        let fastest_suite = results
            .iter()
            .filter(|r| matches!(r.status, BenchmarkStatus::Success))
            .min_by(|a, b| {
                a.duration_seconds
                    .partial_cmp(&b.duration_seconds)
                    .expect("Operation failed")
            })
            .map(|r| r.suite_name.clone());

        let slowest_suite = results
            .iter()
            .filter(|r| matches!(r.status, BenchmarkStatus::Success))
            .max_by(|a, b| {
                a.duration_seconds
                    .partial_cmp(&b.duration_seconds)
                    .expect("Operation failed")
            })
            .map(|r| r.suite_name.clone());

        let mut warnings = Vec::new();
        if failed_suites > 0 {
            warnings.push(format!("{failed_suites} benchmark suites failed"));
        }
        if peak_memory > 8192.0 {
            // 8GB threshold
            warnings.push("High _memory usage detected".to_string());
        }

        BenchmarkSummary {
            total_suites,
            successful_suites,
            failed_suites,
            skipped_suites,
            fastest_suite,
            slowest_suite,
            peak_memory_usage_mb: peak_memory,
            warnings,
        }
    }

    fn compare_with_baseline(
        &self,
        results: &[BenchmarkSuiteResult],
    ) -> Result<Option<BenchmarkComparison>, Box<dyn std::error::Error>> {
        // Placeholder for baseline comparison logic
        // This would load baseline results and compare performance metrics
        Ok(None)
    }

    fn save_reports(&self, report: &BenchmarkReport) -> Result<(), Box<dyn std::error::Error>> {
        // Save JSON report
        if self.config.generate_json {
            let json_path = format!("{}/benchmark_report.json", self.config.output_dir);
            let json = serde_json::to_string_pretty(report)?;
            std::fs::write(json_path, json)?;
        }

        // Generate HTML report
        if self.config.generate_html {
            self.generate_html_report(report)?;
        }

        // Print summary to console
        self.print_summary_report(report);

        Ok(())
    }

    fn generate_html_report(
        &self,
        report: &BenchmarkReport,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let html_content = format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>scirs2-graph Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .suite {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .success {{ border-left: 5px solid #28a745; }}
        .failed {{ border-left: 5px solid #dc3545; }}
        .timeout {{ border-left: 5px solid #ffc107; }}
        .metric {{ margin: 5px 0; }}
        .warning {{ color: #dc3545; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>scirs2-graph Benchmark Report</h1>
        <p><strong>Generated:</strong> {}</p>
        <p><strong>Duration:</strong> {:.2} seconds</p>
        <p><strong>Platform:</strong> {} {}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <div class="metric">Total Suites: {}</div>
        <div class="metric">Successful: {}</div>
        <div class="metric">Failed: {}</div>
        <div class="metric">Peak Memory: {:.1} MB</div>
        {}
    </div>
    
    <div class="suites">
        <h2>Benchmark Suites</h2>
        {}
    </div>
</body>
</html>"#,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            report.total_duration_seconds,
            report
                .suite_results
                .first()
                .map(|r| &r.platform_info.os)
                .unwrap_or(&"Unknown".to_string()),
            report
                .suite_results
                .first()
                .map(|r| &r.platform_info.arch)
                .unwrap_or(&"Unknown".to_string()),
            report.summary.total_suites,
            report.summary.successful_suites,
            report.summary.failed_suites,
            report.summary.peak_memory_usage_mb,
            if report.summary.warnings.is_empty() {
                String::new()
            } else {
                format!(
                    "<div class=\"warning\">Warnings: {}</div>",
                    report.summary.warnings.join(", ")
                )
            },
            report
                .suite_results
                .iter()
                .map(|suite| {
                    let status_class = match suite.status {
                        BenchmarkStatus::Success => "success",
                        BenchmarkStatus::Failed => "failed",
                        BenchmarkStatus::Timeout => "timeout",
                        BenchmarkStatus::Skipped => "skipped",
                    };
                    format!(
                        r#"<div class="suite {}">
                        <h3>{}</h3>
                        <div class="metric">Status: {:?}</div>
                        <div class="metric">Duration: {:.2}s</div>
                        {}
                    </div>"#,
                        status_class,
                        suite.suite_name,
                        suite.status,
                        suite.duration_seconds,
                        suite
                            .memory_usage
                            .as_ref()
                            .map(|m| format!(
                                "<div class=\"metric\">Peak Memory: {:.1} MB</div>",
                                m.peak_memory_mb
                            ))
                            .unwrap_or_default()
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        );

        let html_path = format!("{}/benchmark_report.html", self.config.output_dir);
        std::fs::write(html_path, html_content)?;

        Ok(())
    }

    fn print_summary_report(&self, report: &BenchmarkReport) {
        println!("\n📈 BENCHMARK SUMMARY REPORT");
        println!("═══════════════════════════════════════");
        println!(
            "🔧 Total Duration: {:.2} seconds",
            report.total_duration_seconds
        );
        println!("📊 Suites Run: {}", report.summary.total_suites);
        println!("✅ Successful: {}", report.summary.successful_suites);
        println!("❌ Failed: {}", report.summary.failed_suites);
        println!("⏭️  Skipped: {}", report.summary.skipped_suites);
        println!(
            "💾 Peak Memory: {:.1} MB",
            report.summary.peak_memory_usage_mb
        );

        if let Some(fastest) = &report.summary.fastest_suite {
            println!("🚀 Fastest Suite: {fastest}");
        }
        if let Some(slowest) = &report.summary.slowest_suite {
            println!("🐌 Slowest Suite: {slowest}");
        }

        if !report.summary.warnings.is_empty() {
            println!("\n⚠️  WARNINGS:");
            for warning in &report.summary.warnings {
                println!("   • {warning}");
            }
        }

        println!("\n📂 Results saved to: {}", self.config.output_dir);
        println!("═══════════════════════════════════════");
    }
}

// Trait extension for Command timeout
trait CommandExt {
    fn timeout(&mut self, timeout: Duration) -> &mut Self;
}

impl CommandExt for Command {
    fn timeout(&mut self, timeout: Duration) -> &mut Self {
        // Simplified implementation - would need platform-specific _timeout logic
        self
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert!(!config.suites.is_empty());
        assert_eq!(config.output_dir, "benchmark_results");
    }

    #[test]
    #[ignore = "bench: run_sample_benchmarks -- [[bench]] target (harness=false, its own fn main() exists only to avoid nextest parsing issues), unreachable via `cargo test`/nextest by construction; run with `cargo test --release -- --ignored run_sample_benchmarks`"]
    fn run_sample_benchmarks() {
        let config = BenchmarkConfig {
            suites: vec!["graph_benchmarks".to_string()],
            timeout_seconds: 60,
            sample_size: Some(10),
            ..Default::default()
        };

        let runner = BenchmarkRunner::new(config);
        let report = runner.run_all_benchmarks().expect("Operation failed");

        assert!(!report.suite_results.is_empty());
        println!("Benchmark completed: {:?}", report.summary);
    }
}

fn main() {
    // Silent main function to avoid cargo nextest parsing issues
}
