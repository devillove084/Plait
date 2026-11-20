use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/collective.cu");
    println!("cargo:rerun-if-env-changed=PLAIT_NVCC");
    println!("cargo:rustc-check-cfg=cfg(plait_collective_cubin)");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let arch = detect_arch();
    let Some(nvcc) = select_nvcc(&arch) else {
        eprintln!("plait-collective: nvcc not found; using NVRTC fallback");
        return;
    };
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cubin = out.join("collective.cubin");
    let status = Command::new(&nvcc)
        .args(["-cubin", "-O3", &format!("-arch={arch}")])
        .arg(manifest.join("src/collective.cu"))
        .arg("-o")
        .arg(&cubin)
        .status();
    match status {
        Ok(status) if status.success() => {
            println!("cargo:rustc-cfg=plait_collective_cubin");
            println!("cargo:rustc-env=PLAIT_COLLECTIVE_CUBIN={}", cubin.display());
        }
        _ => eprintln!("plait-collective: cubin compilation failed; using NVRTC fallback"),
    }
}

fn detect_arch() -> String {
    if let Ok(value) = env::var("PLAIT_CUDA_ARCH") {
        return value;
    }
    if let Ok(output) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
    {
        if output.status.success() {
            if let Some(line) = String::from_utf8_lossy(&output.stdout).lines().next() {
                let cap = line.trim().replace('.', "");
                if !cap.is_empty() {
                    return format!("sm_{cap}");
                }
            }
        }
    }
    "sm_80".to_owned()
}

fn select_nvcc(arch: &str) -> Option<PathBuf> {
    if let Ok(path) = env::var("PLAIT_NVCC") {
        return Some(PathBuf::from(path));
    }
    let candidates: &[&str] = if arch.starts_with("sm_8") {
        &[
            "/opt/cuda129/bin/nvcc",
            "/usr/local/cuda/bin/nvcc",
            "/usr/local/cuda-13.2/bin/nvcc",
        ]
    } else {
        &[
            "/usr/local/cuda/bin/nvcc",
            "/usr/local/cuda-13.2/bin/nvcc",
            "/opt/cuda129/bin/nvcc",
        ]
    };
    for candidate in candidates {
        let path = Path::new(candidate);
        if path.exists() {
            return Some(path.to_path_buf());
        }
    }
    None
}
