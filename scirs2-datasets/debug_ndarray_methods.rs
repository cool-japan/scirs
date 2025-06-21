use ndarray::{Array1, ArrayView1};

fn main() {
    let data: Array1<f64> = Array1::from(vec![]);
    let view: ArrayView1<f64> = data.view();
    
    // Let's see what happens when we call different std methods
    println!("Empty array view.len(): {}", view.len());
    
    // This should show us what std method is being called
    let result = view.std(0.0);
    println!("view.std(0.0) result: {:?}", result);
    
    // Check if there are other methods
    let data2: Array1<f64> = Array1::from(vec![1.0, 2.0, 3.0]);
    let view2 = data2.view();
    let result2 = view2.std(0.0);
    println!("non-empty view.std(0.0) result: {:?}", result2);
    
    // Test if ndarray has built-in statistical methods
    // We need to check what trait is providing the std method
    println!("Testing trait method resolution...");
}