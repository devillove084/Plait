#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command="${1:-check}"
preset="${2:-${PLAIT_PRESET:-dev}}"

usage() {
  printf '%s\n' \
    'Usage: ./build.sh [command] [preset]' \
    '' \
    'Commands: configure, fresh, build, test, check, format, format-check, tidy' \
    'Presets:  dev, release, tsan' \
    '' \
    'Defaults to: ./build.sh check dev'
}

if [[ "${command}" == "help" || "${command}" == "--help" || "${command}" == "-h" ]]; then
  usage
  exit 0
fi

case "${preset}" in
  dev | release | tsan) ;;
  *)
    printf 'Unknown preset: %s\n' "${preset}" >&2
    usage >&2
    exit 2
    ;;
esac

config_root="${XDG_CONFIG_HOME:-${HOME}/.config}"
environment_file="${PLAIT_ENV_FILE:-${config_root}/plait/toolchain.env}"
caller_llvm_root="${PLAIT_LLVM_ROOT:-}"
caller_cmake_root="${PLAIT_CMAKE_ROOT:-}"

if [[ -f "${environment_file}" ]]; then
  # Host provisioning is intentionally external to the repository. Variables
  # sourced here remain scoped to this script and the processes it launches.
  # shellcheck source=/dev/null
  source "${environment_file}"
fi

# Explicit caller variables take precedence over values from the optional host
# configuration file.
if [[ -n "${caller_llvm_root}" ]]; then
  PLAIT_LLVM_ROOT="${caller_llvm_root}"
fi
if [[ -n "${caller_cmake_root}" ]]; then
  PLAIT_CMAKE_ROOT="${caller_cmake_root}"
fi

warn() {
  printf 'warning: %s\n' "$*" >&2
}

path_tool() {
  command -v "$1" 2>/dev/null || true
}

llvm_root="${PLAIT_LLVM_ROOT:-${LLVM_ROOT:-}}"
cmake_root="${PLAIT_CMAKE_ROOT:-}"

if [[ -n "${llvm_root}" ]]; then
  cc="${llvm_root}/bin/clang"
  cxx="${llvm_root}/bin/clang++"
  clang_format="${llvm_root}/bin/clang-format"
  clang_tidy="${llvm_root}/bin/clang-tidy"
  export PATH="${llvm_root}/bin:${PATH}"
else
  cc="${CC:-${CLANG_PATH:-$(path_tool clang)}}"
  cxx="${CXX:-${CLANGXX_PATH:-$(path_tool clang++)}}"
  clang_format="${CLANG_FORMAT_PATH:-$(path_tool clang-format)}"
  clang_tidy="${CLANG_TIDY_PATH:-$(path_tool clang-tidy)}"
  warn 'PLAIT_LLVM_ROOT is not set; using the caller toolchain from CC/CXX/PATH.'
  warn 'Compiler, formatter, tidy, sanitizer, and libc++ behavior may differ across hosts; align them to one current stable LLVM release for reproducible builds.'
fi

if [[ -n "${cmake_root}" ]]; then
  cmake="${cmake_root}/bin/cmake"
  ctest="${cmake_root}/bin/ctest"
  export PATH="${cmake_root}/bin:${PATH}"
else
  cmake="${CMAKE_COMMAND:-$(path_tool cmake)}"
  if [[ -n "${cmake}" && -x "$(dirname -- "${cmake}")/ctest" ]]; then
    ctest="$(dirname -- "${cmake}")/ctest"
  else
    ctest="${CTEST_COMMAND:-$(path_tool ctest)}"
  fi
  warn 'PLAIT_CMAKE_ROOT is not set; using CMake and CTest from PATH.'
  warn 'CMake behavior may differ across hosts; use one current stable CMake release for reproducible builds.'
fi

for required_tool in \
  "${cc}" \
  "${cxx}" \
  "${clang_format}" \
  "${clang_tidy}" \
  "${cmake}" \
  "${ctest}"; do
  if [[ -z "${required_tool}" || ! -x "${required_tool}" ]]; then
    printf 'Required Plait tool is missing or not executable: %s\n' \
      "${required_tool:-<not found>}" >&2
    exit 1
  fi
done

export PLAIT_PROJECT_ROOT="${project_root}"
export CC="${cc}"
export CXX="${cxx}"
export CLANG_PATH="${cc}"
export CLANGXX_PATH="${cxx}"

if [[ -n "${llvm_root}" ]]; then
  export PLAIT_LLVM_ROOT="${llvm_root}"
  export LLVM_ROOT="${llvm_root}"
  export LLVM_DIR="${llvm_root}/lib/cmake/llvm"
  export LLVM_CONFIG_PATH="${llvm_root}/bin/llvm-config"
  export CLANG_ROOT="${llvm_root}"
  export Clang_DIR="${llvm_root}/lib/cmake/clang"
  export CMAKE_PREFIX_PATH="${llvm_root}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

  # A pinned Plait LLVM installation includes its matching libc++, runtimes,
  # and LLD. Do not impose these flags on an unpinned user toolchain.
  export CXXFLAGS="${CXXFLAGS:+${CXXFLAGS} }-stdlib=libc++"
  export LDFLAGS="${LDFLAGS:+${LDFLAGS} }-fuse-ld=lld -L${llvm_root}/lib -Wl,-rpath,${llvm_root}/lib"
fi

compiler_version="$("${cxx}" --version | sed -n '1p')"
format_version="$("${clang_format}" --version | sed -n '1p')"
tidy_version="$("${clang_tidy}" --version | sed -n '1,2p' | tr '\n' ' ')"
cmake_version="$("${cmake}" --version | sed -n '1p')"

printf 'Plait preset: %s\n' "${preset}"
printf 'C++ compiler: %s\n' "${compiler_version}"
printf 'clang-format: %s\n' "${format_version}"
printf 'clang-tidy: %s\n' "${tidy_version}"
printf 'CMake: %s\n' "${cmake_version}"

compiler_major="$(printf '%s\n' "${compiler_version}" | sed -n 's/.*clang version \([0-9][0-9]*\).*/\1/p')"
format_major="$(printf '%s\n' "${format_version}" | sed -n 's/.*clang-format version \([0-9][0-9]*\).*/\1/p')"
tidy_major="$(printf '%s\n' "${tidy_version}" | sed -n 's/.*LLVM version \([0-9][0-9]*\).*/\1/p')"

if [[ -z "${compiler_major}" ]]; then
  warn 'The selected C++ compiler is not identifiable as Clang; Plait sanitizer and tidy behavior is validated primarily with Clang.'
elif [[ -n "${format_major}" && "${format_major}" != "${compiler_major}" ]]; then
  warn "clang-format major ${format_major} does not match compiler major ${compiler_major}."
fi
if [[ -n "${compiler_major}" && -n "${tidy_major}" && "${tidy_major}" != "${compiler_major}" ]]; then
  warn "clang-tidy major ${tidy_major} does not match compiler major ${compiler_major}."
fi

publish_compile_database() {
  local source_database="${project_root}/build/${preset}/compile_commands.json"
  local published_database="${project_root}/compile_commands.json"

  if [[ ! -f "${source_database}" ]]; then
    printf 'CMake did not generate %s\n' "${source_database}" >&2
    exit 1
  fi
  "${cmake}" -E copy_if_different "${source_database}" "${published_database}"
  printf 'Compilation database: %s\n' "${published_database}"
}

configure() {
  "${cmake}" --preset "${preset}" -S "${project_root}" \
    -DPLAIT_CLANG_FORMAT_PROGRAM="${clang_format}" \
    -DPLAIT_CLANG_TIDY_PROGRAM="${clang_tidy}"
  publish_compile_database
}

build() {
  configure
  "${cmake}" --build --preset "${preset}" --parallel
}

case "${command}" in
  configure)
    configure
    ;;
  fresh)
    "${cmake}" --fresh --preset "${preset}" -S "${project_root}" \
      -DPLAIT_CLANG_FORMAT_PROGRAM="${clang_format}" \
      -DPLAIT_CLANG_TIDY_PROGRAM="${clang_tidy}"
    publish_compile_database
    ;;
  build)
    build
    ;;
  test)
    build
    "${ctest}" --preset "${preset}"
    ;;
  check)
    build
    "${cmake}" --build --preset "${preset}" --target format-check
    "${cmake}" --build --preset "${preset}" --target tidy
    "${ctest}" --preset "${preset}"
    ;;
  format)
    configure
    "${cmake}" --build --preset "${preset}" --target format
    ;;
  format-check)
    configure
    "${cmake}" --build --preset "${preset}" --target format-check
    ;;
  tidy)
    build
    "${cmake}" --build --preset "${preset}" --target tidy
    ;;
  *)
    printf 'Unknown command: %s\n' "${command}" >&2
    usage >&2
    exit 2
    ;;
esac
