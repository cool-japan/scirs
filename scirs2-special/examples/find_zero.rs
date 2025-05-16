fn main() {
    use scirs2_special::j0;

    // Search for a more accurate zero
    let mut low = 2.3f64;
    let mut high = 2.5f64;
    let mut mid;
    let mut j0_mid;
    let tolerance = 1e-10;

    println!("Searching for the first zero of J0...");

    let max_iterations = 100;
    let mut iteration = 0;

    while high - low > tolerance && iteration < max_iterations {
        mid = (low + high) / 2.0;
        j0_mid = j0(mid);

        println!("Iteration {}: x = {}, J0(x) = {}", iteration, mid, j0_mid);

        if j0_mid.abs() < tolerance {
            println!("Found zero at x = {} with J0(x) = {}", mid, j0_mid);
            break;
        }

        if j0_mid > 0.0 {
            low = mid;
        } else {
            high = mid;
        }

        iteration += 1;
    }

    if iteration == max_iterations {
        println!("Did not converge after {} iterations", max_iterations);
        println!(
            "Current best estimate: x = {}, J0(x) = {}",
            (low + high) / 2.0,
            j0((low + high) / 2.0)
        );
    }
}
