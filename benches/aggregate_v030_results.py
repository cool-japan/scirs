#!/usr/bin/env python3
"""
SciRS2 v0.3.0 Benchmark Results Aggregator and Analyzer

This script aggregates benchmark results from all v0.3.0 benchmark suites
and generates summary reports, comparisons, and visualizations.

Usage:
    python3 aggregate_v030_results.py [--output OUTPUT_DIR]

Copyright: COOLJAPAN OU (Team KitaSan)
License: Apache-2.0
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import statistics

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_section(text: str):
    """Print a formatted section header"""
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{'-'*60}{Colors.ENDC}")


def load_json_results(filepath: str) -> List[Dict[str, Any]]:
    """Load JSON results from a file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{Colors.WARNING}Warning: {filepath} not found{Colors.ENDC}")
        return []
    except json.JSONDecodeError as e:
        print(f"{Colors.FAIL}Error decoding {filepath}: {e}{Colors.ENDC}")
        return []


def format_time(ns: float) -> str:
    """Format time in nanoseconds to human-readable format"""
    if ns < 1_000:
        return f"{ns:.2f} ns"
    elif ns < 1_000_000:
        return f"{ns/1_000:.2f} μs"
    elif ns < 1_000_000_000:
        return f"{ns/1_000_000:.2f} ms"
    else:
        return f"{ns/1_000_000_000:.2f} s"


def format_throughput(ops_per_sec: float) -> str:
    """Format throughput in ops/sec to human-readable format"""
    if ops_per_sec < 1_000:
        return f"{ops_per_sec:.2f} ops/s"
    elif ops_per_sec < 1_000_000:
        return f"{ops_per_sec/1_000:.2f} Kops/s"
    elif ops_per_sec < 1_000_000_000:
        return f"{ops_per_sec/1_000_000:.2f} Mops/s"
    else:
        return f"{ops_per_sec/1_000_000_000:.2f} Gops/s"


def aggregate_by_category(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Aggregate results by category"""
    categorized = defaultdict(list)
    for result in results:
        categorized[result['category']].append(result)
    return dict(categorized)


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate statistics for a set of results"""
    if not results:
        return {}

    times = [r['mean_time_ns'] for r in results]
    throughputs = [r['throughput_ops_per_sec'] for r in results]

    return {
        'mean_time_ns': statistics.mean(times),
        'median_time_ns': statistics.median(times),
        'min_time_ns': min(times),
        'max_time_ns': max(times),
        'mean_throughput': statistics.mean(throughputs),
        'max_throughput': max(throughputs),
    }


def print_category_summary(category: str, results: List[Dict[str, Any]]):
    """Print summary for a category"""
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}Category: {category}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'─'*60}{Colors.ENDC}")

    stats = calculate_statistics(results)
    if stats:
        print(f"  Operations: {len(results)}")
        print(f"  Mean time: {format_time(stats['mean_time_ns'])}")
        print(f"  Median time: {format_time(stats['median_time_ns'])}")
        print(f"  Min time: {format_time(stats['min_time_ns'])}")
        print(f"  Max time: {format_time(stats['max_time_ns'])}")
        print(f"  Mean throughput: {format_throughput(stats['mean_throughput'])}")
        print(f"  Max throughput: {format_throughput(stats['max_throughput'])}")

    # Print top 5 fastest operations
    sorted_results = sorted(results, key=lambda x: x['mean_time_ns'])
    print(f"\n  {Colors.OKGREEN}Top 5 Fastest Operations:{Colors.ENDC}")
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"    {i}. {result['operation']} (size: {result.get('size', 'N/A')}) - {format_time(result['mean_time_ns'])}")


def print_comparison_table(results: List[Dict[str, Any]], title: str):
    """Print a comparison table for results"""
    print(f"\n{Colors.BOLD}{title}{Colors.ENDC}")
    print(f"{'Operation':<30} {'Size':<10} {'Mean Time':<15} {'Throughput':<15}")
    print(f"{'-'*70}")

    for result in sorted(results, key=lambda x: x.get('size', 0)):
        operation = result['operation'][:28]
        size = str(result.get('size', 'N/A'))
        mean_time = format_time(result['mean_time_ns'])
        throughput = format_throughput(result['throughput_ops_per_sec'])
        print(f"{operation:<30} {size:<10} {mean_time:<15} {throughput:<15}")


def generate_markdown_report(all_results: Dict[str, List[Dict[str, Any]]], output_path: str):
    """Generate a Markdown report"""
    with open(output_path, 'w') as f:
        f.write("# SciRS2 v0.3.0 Benchmark Results\n\n")
        f.write("*Generated automatically by aggregate_v030_results.py*\n\n")

        f.write("## Overview\n\n")
        total_benchmarks = sum(len(results) for results in all_results.values())
        f.write(f"- **Total benchmark suites:** {len(all_results)}\n")
        f.write(f"- **Total benchmarks:** {total_benchmarks}\n\n")

        for suite_name, results in all_results.items():
            if not results:
                continue

            f.write(f"## {suite_name.replace('_', ' ').title()}\n\n")

            # Aggregate by category
            categorized = aggregate_by_category(results)

            for category, cat_results in categorized.items():
                f.write(f"### {category.replace('_', ' ').title()}\n\n")

                f.write("| Operation | Size | Mean Time | Throughput |\n")
                f.write("|-----------|------|-----------|------------|\n")

                for result in sorted(cat_results, key=lambda x: x.get('size', 0)):
                    operation = result['operation']
                    size = result.get('size', 'N/A')
                    mean_time = format_time(result['mean_time_ns'])
                    throughput = format_throughput(result['throughput_ops_per_sec'])
                    f.write(f"| {operation} | {size} | {mean_time} | {throughput} |\n")

                f.write("\n")

    print(f"{Colors.OKGREEN}✓ Markdown report saved to: {output_path}{Colors.ENDC}")


def main():
    """Main function"""
    print_header("SciRS2 v0.3.0 Benchmark Results Aggregator")

    # Define result files
    result_files = {
        "Comprehensive Suite": "/tmp/scirs2_v030_comprehensive_results.json",
        "Autograd": "/tmp/scirs2_v030_autograd_results.json",
        "Neural Networks": "/tmp/scirs2_v030_neural_results.json",
        "Time Series": "/tmp/scirs2_v030_series_results.json",
    }

    # Load all results
    all_results = {}
    for suite_name, filepath in result_files.items():
        results = load_json_results(filepath)
        if results:
            all_results[suite_name] = results
            print(f"{Colors.OKGREEN}✓ Loaded {len(results)} results from {suite_name}{Colors.ENDC}")
        else:
            print(f"{Colors.WARNING}✗ No results found for {suite_name}{Colors.ENDC}")

    if not all_results:
        print(f"\n{Colors.FAIL}Error: No benchmark results found!{Colors.ENDC}")
        print(f"Please run the benchmarks first:\n  ./v030_run_all_benchmarks.sh")
        sys.exit(1)

    # Print summaries for each suite
    for suite_name, results in all_results.items():
        print_section(f"{suite_name} Summary")

        # Aggregate by category
        categorized = aggregate_by_category(results)

        for category, cat_results in categorized.items():
            print_category_summary(category, cat_results)

    # Generate overall statistics
    print_section("Overall Statistics")

    total_benchmarks = sum(len(results) for results in all_results.values())
    all_times = []
    all_throughputs = []

    for results in all_results.values():
        for result in results:
            all_times.append(result['mean_time_ns'])
            all_throughputs.append(result['throughput_ops_per_sec'])

    print(f"  Total benchmarks: {total_benchmarks}")
    print(f"  Mean time: {format_time(statistics.mean(all_times))}")
    print(f"  Median time: {format_time(statistics.median(all_times))}")
    print(f"  Min time: {format_time(min(all_times))}")
    print(f"  Max time: {format_time(max(all_times))}")
    print(f"  Mean throughput: {format_throughput(statistics.mean(all_throughputs))}")
    print(f"  Max throughput: {format_throughput(max(all_throughputs))}")

    # Generate reports
    print_section("Generating Reports")

    output_dir = "/tmp/scirs2_v030_reports"
    os.makedirs(output_dir, exist_ok=True)

    # Markdown report
    markdown_path = os.path.join(output_dir, "v030_benchmark_report.md")
    generate_markdown_report(all_results, markdown_path)

    # JSON aggregate
    aggregate_path = os.path.join(output_dir, "v030_aggregate.json")
    with open(aggregate_path, 'w') as f:
        json.dump({
            "suites": all_results,
            "total_benchmarks": total_benchmarks,
            "overall_stats": {
                "mean_time_ns": statistics.mean(all_times),
                "median_time_ns": statistics.median(all_times),
                "min_time_ns": min(all_times),
                "max_time_ns": max(all_times),
                "mean_throughput": statistics.mean(all_throughputs),
                "max_throughput": max(all_throughputs),
            }
        }, f, indent=2)

    print(f"{Colors.OKGREEN}✓ JSON aggregate saved to: {aggregate_path}{Colors.ENDC}")

    print_header("Benchmark Analysis Complete!")
    print(f"\n{Colors.OKGREEN}Reports generated in: {output_dir}{Colors.ENDC}")
    print(f"  - Markdown report: v030_benchmark_report.md")
    print(f"  - JSON aggregate: v030_aggregate.json\n")


if __name__ == "__main__":
    main()
