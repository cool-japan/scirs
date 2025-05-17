use rand::{rngs::StdRng, Rng, SeedableRng};

fn main() {
    // Test from_rng
    let mut system_rng = rand::rng();
    let mut rng1 = StdRng::from_rng(&mut system_rng);
    println!("Random number from system rng: {}", rng1.random::<u64>());

    // Test from_seed
    let mut rng2 = StdRng::from_seed([0u8; 32]);
    println!("Random number from seed: {}", rng2.random::<u64>());

    // Test rng (renamed from thread_rng)
    let mut rng3 = rand::rng();
    println!("Random number from rng: {}", rng3.random::<u64>());

    println!("Tests passed!");
}
