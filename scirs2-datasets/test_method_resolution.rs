use ndarray::{Array1, ArrayView1};

fn main() {
    let data: Array1<f64> = Array1::from(vec![]);
    let view: ArrayView1<f64> = data.view();
    
    println!("Empty array view.len(): {}", view.len());
    
    // Test ndarray's built-in std method - let's see what parameters it takes
    println!("Calling view.std():");
    let builtin_std = view.std(0.0);
    println!("Built-in std result: {:?}", builtin_std);
    
    // Test with non-empty array
    let data2: Array1<f64> = Array1::from(vec![1.0, 2.0, 3.0]);
    let view2 = data2.view();
    let builtin_std2 = view2.std(0.0);
    println!("Non-empty array std result: {:?}", builtin_std2);
}