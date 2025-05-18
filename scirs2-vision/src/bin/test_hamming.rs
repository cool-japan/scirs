use scirs2_vision::feature::brief::hamming_distance;

fn main() {
    let desc1 = vec![0b11001100_11001100_11001100_11001100u32];
    let desc2 = vec![0b11001100_11001100_11001100_00000000u32];

    let distance = hamming_distance(&desc1, &desc2);
    println!("Hamming distance: {} (expected: 6)", distance);

    // Debug the bits
    let diff = desc1[0] ^ desc2[0];
    println!("XOR result: {:032b}", diff);
    println!("Ones count: {}", diff.count_ones());

    // Compare byte by byte
    for i in 0..4 {
        let byte1 = (desc1[0] >> (i * 8)) & 0xFF;
        let byte2 = (desc2[0] >> (i * 8)) & 0xFF;
        let byte_diff = byte1 ^ byte2;
        println!(
            "Byte {}: {:08b} XOR {:08b} = {:08b} (ones: {})",
            i,
            byte1,
            byte2,
            byte_diff,
            byte_diff.count_ones()
        );
    }
}
