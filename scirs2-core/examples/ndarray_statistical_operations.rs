use scirs2_core::ndarray_ext::stats::{
    histogram, histogram2d, max_2d, mean_2d, median_2d, min_2d, percentile_2d, quantile,
    std_dev_2d, sum_2d, variance_2d,
};
use scirs2_core::ndarray_ext::{array, Array2, Axis};

#[allow(dead_code)]
fn main() {
    println!("Demonstrating SciPy-like Statistical Functions in scirs2-core\n");

    // Create a sample 2D array
    let data = array![
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0],
        [11.0, 12.0, 13.0, 14.0, 15.0]
    ];

    println!("Sample 2D array:");
    print_array2(&data);
    println!();

    // Basic statistics
    println!("=== Basic Statistics ===");

    // Mean
    let global_mean = mean_2d(&data.view(), None).unwrap();
    let col_means = mean_2d(&data.view(), Some(Axis(0))).unwrap();
    let row_means = mean_2d(&data.view(), Some(Axis(1))).unwrap();

    println!("Global mean: {}", global_mean[0]);
    println!("Column means: {col_means:?}");
    println!("Row means: {row_means:?}");
    println!();

    // Median
    let global_median = median_2d(&data.view(), None).unwrap();
    let col_medians = median_2d(&data.view(), Some(Axis(0))).unwrap();
    let row_medians = median_2d(&data.view(), Some(Axis(1))).unwrap();

    println!("Global median: {}", global_median[0]);
    println!("Column medians: {col_medians:?}");
    println!("Row medians: {row_medians:?}");
    println!();

    // Variance and Standard Deviation
    let global_var = variance_2d(&data.view(), None, 1).unwrap();
    let global_std = std_dev_2d(&data.view(), None, 1).unwrap();

    println!("Global variance (sample): {}", global_var[0]);
    println!("Global standard deviation (sample): {}", global_std[0]);
    println!();

    // Min and Max
    let global_min = min_2d(&data.view(), None).unwrap();
    let global_max = max_2d(&data.view(), None).unwrap();

    println!("Global minimum: {}", global_min[0]);
    println!("Global maximum: {}", global_max[0]);
    println!();

    // Sum
    let global_sum = sum_2d(&data.view(), None).unwrap();
    let col_sums = sum_2d(&data.view(), Some(Axis(0))).unwrap();

    println!("Global sum: {}", global_sum[0]);
    println!("Column sums: {col_sums:?}");
    println!();

    // Percentile
    let percentiles = [0.0, 25.0, 50.0, 75.0, 100.0];
    for p in percentiles.iter() {
        let global_p = percentile_2d(&data.view(), *p, None).unwrap();
        println!("Global {}th percentile: {}", p, global_p[0]);
    }
    println!();

    // Create 1D array for histogram and quantile examples
    let data_1d = array![
        1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0
    ];

    println!("=== Histogram Example ===");
    println!("1D array: {data_1d:?}");

    // Calculate histogram
    let (hist, bin_edges) = histogram(data_1d.view(), 5, None, None).unwrap();

    println!("Histogram with 5 bins:");
    println!("Bin edges: {bin_edges:?}");
    println!("Counts: {hist:?}");
    println!();

    // Visualize histogram as ASCII
    println!("ASCII Histogram:");
    for (i, &count) in hist.iter().enumerate() {
        println!(
            "{:.1} - {:.1}: {} {}",
            bin_edges[i],
            bin_edges[i + 1],
            "#".repeat(count),
            count
        );
    }
    println!();

    // Quantile example
    println!("=== Quantile Example ===");
    let quantiles = array![0.0, 0.25, 0.5, 0.75, 1.0];
    let results = quantile(data_1d.view(), quantiles.view(), None).unwrap();

    println!("Quantiles using 'linear' interpolation:");
    for (&q, &val) in quantiles.iter().zip(results.iter()) {
        println!("  {q:.2} quantile: {val:.2}");
    }
    println!();

    // 2D Histogram example
    println!("=== 2D Histogram Example ===");

    // Create x and y coordinates
    let n = 100;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    // Generate some sample data (roughly correlated)
    for i in 0..n {
        let x_val = (i as f64) / 10.0;
        let y_val = x_val + (scirs2_core::random::random::<f64>() - 0.5) * 3.0;
        x.push(x_val);
        y.push(y_val);
    }

    let x_arr = Array2::from_shape_vec((n, 1), x)
        .unwrap()
        .column(0)
        .to_owned();
    let y_arr = Array2::from_shape_vec((n, 1), y)
        .unwrap()
        .column(0)
        .to_owned();

    println!("Generated {n} data points");

    // Calculate 2D histogram
    let (hist_2d, x_edges, y_edges) =
        histogram2d(x_arr.view(), y_arr.view(), Some((6, 6)), None, None).unwrap();

    println!("2D Histogram (6×6 bins):");
    println!("X edges: {x_edges:?}");
    println!("Y edges: {y_edges:?}");
    println!();

    // Visualize the 2D histogram
    println!("ASCII 2D Histogram (counts):");
    for i in (0..hist_2d.shape()[0]).rev() {
        print!("{:.1}-{:.1}: ", y_edges[i], y_edges[i + 1]);
        for j in 0..hist_2d.shape()[1] {
            let count = hist_2d[[i, j]];
            if count == 0 {
                print!(" . ");
            } else {
                print!(" {count} ");
            }
        }
        println!();
    }

    print!("      ");
    for j in 0..hist_2d.shape()[1] {
        print!("{:.1} ", x_edges[j]);
    }
    println!("{:.1}", x_edges[hist_2d.shape()[1]]);
}

// Helper function to print a 2D array
#[allow(dead_code)]
fn print_array2<T: std::fmt::Display>(arr: &Array2<T>) {
    for row in arr.rows() {
        for item in row.iter() {
            print!("{item:6} ");
        }
        println!();
    }
}
