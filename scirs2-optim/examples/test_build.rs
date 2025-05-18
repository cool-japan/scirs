use ndarray::array;
use scirs2_optim::optimizers::{Adam, Optimizer};

fn main() {
    let mut adam = Adam::new(0.001);
    adam.set_beta1(0.9);
    adam.set_beta2(0.999);

    let params = array![1.0, 2.0, 3.0];
    let gradients = array![0.1, 0.2, 0.3];

    match adam.step(&params, &gradients) {
        Ok(updated) => println!("Updated params: {}", updated),
        Err(e) => eprintln!("Error: {}", e),
    }
}
