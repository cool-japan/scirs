use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

#[allow(dead_code)]
fn main() {
    // Test from_rng
    let mut system_rng = scirs2_core::random::rng();
    let mut rng1 = StdRng::from_rng(&mut system_rng);
    println!("Random number from system rng: {}", rng1.random::<u64>());

    // Test from_seed
    let mut rng2 = StdRng::seed_from_u64(0u64);
    println!("Random number from seed: {}", rng2.random::<u64>());

    // Test rng (renamed from rng)
    let mut rng3 = scirs2_core::random::rng();
    println!("Random number from rng: {}", rng3.random::<u64>());

    println!("Tests passed!");
}
