use ndarray::Array1;
use scirs2_signal::wvd::{wigner_ville, WvdConfig};

fn main() {
    // Create a chirp signal
    let fs = 1000.0;
    let t = Array1::linspace(0.0, 1.0, 1000);
    let signal = t.mapv(|ti| (2.0 * std::f64::consts::PI * (10.0 * ti + 50.0 * ti * ti)).sin());

    // Configure the WVD
    let config = WvdConfig {
        fs,
        ..Default::default()
    };

    // Compute the WVD
    let wvd = wigner_ville(&signal, config).unwrap();
    println!("WVD shape: {:?}", wvd.shape());
}
