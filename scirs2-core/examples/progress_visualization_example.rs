//! Enhanced Progress Visualization Example
//!
//! This example demonstrates the comprehensive progress tracking capabilities
//! including multiple styles, multi-progress tracking, and logger integration.

use scirs2_core::logging::progress::{
    MultiProgress, ProgressBuilder, ProgressGroup, ProgressStyle, ProgressSymbols,
};
use scirs2_core::logging::{LogLevel, Logger};
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure logging
    scirs2_core::logging::set_min_log_level(LogLevel::Info);

    println!("=== Enhanced Progress Visualization Demo ===\n");

    // Demo 1: Different progress styles
    demo_progress_styles()?;

    // Demo 2: Multi-progress tracking
    demo_multi_progress()?;

    // Demo 3: Logger integration
    demo_logger_integration()?;

    // Demo 4: Custom themes and templates
    demo_themes_and_templates()?;

    // Demo 5: Progress groups
    demo_progress_groups()?;

    println!("\n=== Demo Complete ===");
    Ok(())
}

fn demo_progress_styles() -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 1: Different Progress Styles\n");

    let styles = [
        (ProgressStyle::Percentage, "Percentage"),
        (ProgressStyle::Bar, "Basic Bar"),
        (ProgressStyle::BlockBar, "Block Bar"),
        (ProgressStyle::Spinner, "Spinner"),
        (ProgressStyle::DetailedBar, "Detailed Bar"),
    ];

    for (style, name) in &styles {
        println!("Demonstrating {}", name);

        let mut progress = ProgressBuilder::new(&format!("Processing ({})", name), 50)
            .style(*style)
            .show_eta(true)
            .show_statistics(true)
            .build();

        progress.start();

        for i in 0..=50 {
            // Simulate variable processing speed
            let delay = if i % 10 == 0 { 100 } else { 20 };
            thread::sleep(Duration::from_millis(delay));
            progress.update(i);
        }

        progress.finish();
        println!(); // Add spacing between demos
    }

    Ok(())
}

fn demo_multi_progress() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nDemo 2: Multi-Progress Tracking\n");

    let mut multi = MultiProgress::new();

    // Add multiple progress trackers
    let task1_id = multi.add(
        ProgressBuilder::new("Download Task", 100)
            .style(ProgressStyle::Bar)
            .build(),
    );

    let task2_id = multi.add(
        ProgressBuilder::new("Processing Task", 75)
            .style(ProgressStyle::DetailedBar)
            .build(),
    );

    let task3_id = multi.add(
        ProgressBuilder::new("Upload Task", 60)
            .style(ProgressStyle::Spinner)
            .build(),
    );

    // Start all trackers
    multi.start_all();

    // Simulate concurrent progress
    for i in 0..100 {
        thread::sleep(Duration::from_millis(50));

        // Update different tasks at different rates
        if i < 100 {
            multi.update(task1_id, i as u64);
        }
        if i < 75 {
            multi.update(task2_id, i as u64);
        }
        if i < 60 {
            multi.update(task3_id, i as u64);
        }

        // Finish tasks when complete
        if i == 60 {
            multi.finish(task3_id);
        }
        if i == 75 {
            multi.finish(task2_id);
        }
        if i == 100 {
            multi.finish(task1_id);
        }
    }

    println!("Overall progress: {:.1}%", multi.overall_progress());
    println!("All tasks complete: {}", multi.all_complete());

    Ok(())
}

fn demo_logger_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nDemo 3: Logger Integration\n");

    let logger = Logger::new("data_processor")
        .with_field("version", "1.0")
        .with_field("mode", "batch");

    logger.info("Starting data processing pipeline");

    // Use logger's built-in progress tracking
    let result = logger.with_progress("Processing dataset", 200, |progress| {
        let mut processed_items = 0;

        for batch in 0..10 {
            // Process a batch
            for _item in 0..20 {
                thread::sleep(Duration::from_millis(25));
                processed_items += 1;
                progress.update(processed_items);
            }

            // Log batch completion
            logger.info(&format!("Completed batch {}/10", batch + 1));
        }

        "Processing completed successfully"
    });

    logger.info(&format!("Result: {}", result));

    Ok(())
}

fn demo_themes_and_templates() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nDemo 4: Custom Themes and Templates\n");

    // Demo with scientific theme
    println!("Scientific Theme:");
    let mut progress = ProgressBuilder::new("Scientific Computation", 80)
        .style(ProgressStyle::DetailedBar)
        .symbols(ProgressSymbols {
            start: "│".to_string(),
            end: "│".to_string(),
            fill: "█".to_string(),
            empty: "░".to_string(),
            spinner: vec![
                "◐".to_string(),
                "◓".to_string(),
                "◑".to_string(),
                "◒".to_string(),
            ],
        })
        .build();

    progress.start();

    for i in 0..=80 {
        thread::sleep(Duration::from_millis(30));
        progress.update(i);
    }

    progress.finish();

    // Demo with minimal theme
    println!("\nMinimal Theme:");
    let mut progress = ProgressBuilder::new("Basic Processing", 50)
        .style(ProgressStyle::Bar)
        .symbols(ProgressSymbols {
            start: "[".to_string(),
            end: "]".to_string(),
            fill: "#".to_string(),
            empty: "-".to_string(),
            spinner: vec![
                "|".to_string(),
                "/".to_string(),
                "-".to_string(),
                "\\".to_string(),
            ],
        })
        .build();

    progress.start();

    for i in 0..=50 {
        thread::sleep(Duration::from_millis(40));
        progress.update(i);
    }

    progress.finish();

    Ok(())
}

fn demo_progress_groups() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nDemo 5: Progress Groups\n");

    let mut group = ProgressGroup::new("Data Pipeline");

    // Add multiple related tasks to the group
    let extract_id = group.add_tracker(ProgressBuilder::new("Extract Data", 100).build());

    let transform_id = group.add_tracker(ProgressBuilder::new("Transform Data", 150).build());

    let load_id = group.add_tracker(ProgressBuilder::new("Load Data", 80).build());

    group.start_all();

    // Simulate ETL pipeline
    println!("Starting ETL Pipeline...");

    // Extract phase
    for i in 0..=100 {
        thread::sleep(Duration::from_millis(20));
        group.update(extract_id, i);
    }
    println!("Extraction complete");

    // Transform phase
    for i in 0..=150 {
        thread::sleep(Duration::from_millis(15));
        group.update(transform_id, i);
    }
    println!("Transformation complete");

    // Load phase
    for i in 0..=80 {
        thread::sleep(Duration::from_millis(25));
        group.update(load_id, i);
    }
    println!("Loading complete");

    group.finish_all();
    println!("Group overall progress: {:.1}%", group.overall_progress());

    Ok(())
}

/// Demonstrate adaptive progress tracking
fn _demo_adaptive_tracking() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nDemo: Adaptive Progress Tracking\n");

    let mut progress = ProgressBuilder::new("Adaptive Processing", 1000)
        .style(ProgressStyle::DetailedBar)
        .adaptive_rate(true)
        .min_update_interval(Duration::from_millis(50))
        .max_update_interval(Duration::from_millis(500))
        .build();

    progress.start();

    for i in 0..=1000 {
        // Simulate variable processing speeds
        let delay = match i {
            0..=100 => 10,   // Fast start
            101..=500 => 5,  // Very fast middle
            501..=800 => 15, // Slower end
            _ => 20,         // Slowest final phase
        };

        thread::sleep(Duration::from_millis(delay));
        progress.update(i);
    }

    progress.finish();

    Ok(())
}
