fn main() {
    use scirs2_special::j0;

    // Theoretical zero from literature
    let theoretical_zero = 2.404825557695773;
    let j0_val = j0(theoretical_zero);
    println!("J0({}) = {}", theoretical_zero, j0_val);

    // The zero found by our find_zero program
    let our_zero = 2.5;
    let j0_val2 = j0(our_zero);
    println!("J0({}) = {}", our_zero, j0_val2);

    // Try with a slightly different value
    let test_zero = 0.0;
    let j0_val3 = j0(test_zero);
    println!("J0({}) = {}", test_zero, j0_val3);
}
