use scirs2_special::j1;

fn main() {
    let result = j1(2.0f64);
    println!("j1(2.0) = {}", result);
    println!("j1(2.0) formatted: {:.10}", result);
}