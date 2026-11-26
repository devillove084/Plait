"""Check the CUDA W4A4 integer obligations and inspect a compiled PTX entry."""

import argparse
import hashlib
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "specs/kernels/PlaitCudaW4A4.smt2"
ENTRY = re.compile(
    r"^\.visible\s+\.entry\s+\S*plait_integer_mma10mma_kernelILj4E\S*\(", re.M
)
REQUIRED_OPCODES = (
    "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32",
    "dp4a.s32.s32",
    "shfl.sync.bfly.b32",
    "fma.rn.f32",
    "cvt.rn.f16.f32",
)


def check_obligations(z3: str) -> None:
    labels = re.findall(r'^\(echo "([^"]+)"\)$', CONTRACT.read_text(), re.M)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("The contract must name unique proof obligations")
    result = subprocess.run(
        [z3, "-smt2", str(CONTRACT)], capture_output=True, text=True
    )
    expected = [line for name in labels for line in (name, "unsat")]
    if result.returncode or result.stdout.splitlines() != expected:
        raise ValueError(f"Unexpected Z3 result:\n{result.stdout}{result.stderr}")
    print(f"{len(labels)} integer and indexing obligations passed")


def check_ptx(path: Path) -> None:
    raw = path.read_bytes()
    assembly = re.sub(r"//[^\n]*", "", raw.decode())
    entry = ENTRY.search(assembly)
    if entry is None:
        raise ValueError("The PTX has no native W4A4 MMA kernel entry")
    body, end, _ = assembly[entry.end() :].partition("\n}")
    if not end:
        raise ValueError("The native W4A4 MMA kernel entry is incomplete")
    for opcode in REQUIRED_OPCODES:
        if not re.search(r"\b" + re.escape(opcode) + r"\s", body):
            raise ValueError(f"The W4A4 entry is missing {opcode}")
    print(f"W4A4 PTX entry checked (SHA-256 {hashlib.sha256(raw).hexdigest()})")


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--z3", default="z3")
    cli.add_argument(
        "--ptx",
        type=Path,
        help="PTX built from kernels/w4a16_cuda.cu with native integer MMA enabled",
    )
    args = cli.parse_args()
    check_obligations(args.z3)
    if args.ptx is not None:
        check_ptx(args.ptx)


if __name__ == "__main__":
    main()
