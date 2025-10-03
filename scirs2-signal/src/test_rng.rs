use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::SeedableRng;

#[allow(unused_imports)]
#[allow(dead_code)]
fn main() {
    // Test from_entropy
    let _rng1 = StdRng::from_entropy();

    // Test from_seed
    let _rng2 = StdRng::seed_from_u64([0u8; 32]);

    // Test rng
    let mut _rng3 = scirs2_core::random::rng();

    println!("Tests passed!");
}
