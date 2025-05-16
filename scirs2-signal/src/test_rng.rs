use rand::{rngs::StdRng, Rng, SeedableRng};

fn main() {
    // Test from_entropy
    let _rng1 = StdRng::from_entropy();
    
    // Test from_seed
    let _rng2 = StdRng::from_seed([0u8; 32]);
    
    // Test thread_rng
    let mut _rng3 = rand::thread_rng();
    
    println!("Tests passed!");
}