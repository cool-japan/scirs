// pathfinder/simd/build.rs
//
// Copyright © 2019 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate rustc_version;

use rustc_version::Channel;

fn main() {
    // Assert we haven't travelled back in time
    assert!(rustc_version::version().expect("failed to determine rustc version").major >= 1);

    // Declare the custom cfg name for check-cfg compliance
    println!("cargo:rustc-check-cfg=cfg(pf_rustc_nightly)");

    // Set cfg flags depending on release channel
    match rustc_version::version_meta().expect("failed to query rustc version metadata").channel {
        Channel::Nightly => {
            println!("cargo:rustc-cfg=pf_rustc_nightly");
        }
        _ => {}
    }
}
