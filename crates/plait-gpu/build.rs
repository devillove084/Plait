use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/cutlass_shim.cu");
    println!("cargo:rerun-if-env-changed=PLAIT_NVCC");
    println!("cargo:rerun-if-env-changed=PLAIT_CUTLASS_DIR");
    println!("cargo:rustc-check-cfg=cfg(plait_cutlass)");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cutlass = env::var("PLAIT_CUTLASS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest.join("../../target/cutlass"));
    let cutlass_include = cutlass.join("include");
    if !cutlass_include.join("cutlass/gemm/device/gemm.h").exists() {
        eprintln!(
            "plait-gpu: CUTLASS not found at {}; building without CUTLASS",
            cutlass.display()
        );
        return;
    }

    let arch = detect_arch();
    let nvcc = select_nvcc(&arch);
    let Some(nvcc) = nvcc else {
        eprintln!("plait-gpu: nvcc not found; building without CUTLASS");
        return;
    };
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let object = out.join("cutlass_shim.o");
    let archive = out.join("libplait_cutlass.a");
    let status = Command::new(&nvcc)
        .args([
            "-std=c++17",
            "-O3",
            "-Xcompiler",
            "-fPIC",
            &format!("-arch={arch}"),
            "-I",
        ])
        .arg(&cutlass_include)
        .arg("-c")
        .arg(manifest.join("src/cutlass_shim.cu"))
        .arg("-o")
        .arg(&object)
        .status()
        .expect("failed to run nvcc");
    if !status.success() {
        panic!("CUTLASS shim compilation failed with {nvcc:?} for {arch}");
    }
    let ar = Command::new("ar")
        .args(["rcs"])
        .arg(&archive)
        .arg(&object)
        .status()
        .expect("failed to run ar");
    if !ar.success() {
        panic!("failed to archive CUTLASS shim");
    }
    println!("cargo:rustc-cfg=plait_cutlass");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=plait_cutlass");
    if let Some(root) = cuda_root(&nvcc) {
        println!("cargo:rustc-link-search=native={}/lib64", root.display());
        println!("cargo:rustc-link-search=native={}/lib", root.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib64", root.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", root.display());
    }
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-arg=-lrt");
    println!("cargo:rustc-link-arg=-ldl");
    println!("cargo:rustc-link-arg=-lpthread");
}

fn detect_arch() -> String {
    if let Ok(value) = env::var("PLAIT_CUDA_ARCH") {
        return value;
    }
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output();
    if let Ok(output) = output {
        if output.status.success() {
            if let Some(line) = String::from_utf8_lossy(&output.stdout).lines().next() {
                let cap = line.trim().replace('.', "");
                if !cap.is_empty() {
                    return format!("sm_{cap}");
                }
            }
        }
    }
    "sm_86".to_owned()
}

fn select_nvcc(arch: &str) -> Option<PathBuf> {
    if let Ok(path) = env::var("PLAIT_NVCC") {
        return Some(PathBuf::from(path));
    }
    // Local NVIDIA Ampere driver predates CUDA 13.2; use the installed 12.9 nvcc
    // when targeting Ampere. Blackwell needs the 13.x toolkit.
    if arch.starts_with("sm_8") {
        let old = Path::new("/opt/cuda129/bin/nvcc");
        if old.exists() {
            return Some(old.to_path_buf());
        }
    }
    for candidate in [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-13.2/bin/nvcc",
        "/opt/cuda129/bin/nvcc",
    ] {
        let path = Path::new(candidate);
        if path.exists() {
            return Some(path.to_path_buf());
        }
    }
    None
}

fn cuda_root(nvcc: &Path) -> Option<PathBuf> {
    let bin = nvcc.parent()?;
    let root = bin.parent()?;
    if root.join("lib64").exists() || root.join("lib").exists() {
        Some(root.to_path_buf())
    } else {
        None
    }
}
