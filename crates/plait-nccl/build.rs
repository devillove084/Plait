use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/persistent.cu");
    println!("cargo:rustc-check-cfg=cfg(plait_persistent_cubin)");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cubin = out.join("persistent.cubin");
    let nvcc = if Path::new("/opt/cuda129/bin/nvcc").exists() {
        "/opt/cuda129/bin/nvcc"
    } else {
        "/usr/local/cuda/bin/nvcc"
    };
    let status = Command::new(nvcc)
        .args(["-cubin", "-O3", "-arch=sm_86"])
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/persistent.cu"))
        .arg("-o")
        .arg(&cubin)
        .status();
    if let Ok(status) = status {
        if status.success() {
            println!("cargo:rustc-cfg=plait_persistent_cubin");
            println!("cargo:rustc-env=PLAIT_PERSISTENT_CUBIN={}", cubin.display());
            return;
        }
    }
    eprintln!("plait-nccl: persistent cubin compilation unavailable");
}
