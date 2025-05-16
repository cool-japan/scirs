fn main() {
    use scirs2_special::j0;
    let x = 2.404825557695773f64;
    let j0_x: f64 = j0(x);
    println!("J0({}) = {}", x, j0_x);
    println!("Absolute value: {}", j0_x.abs());
}
