# Dependency policy and lock

Plait does not use Git submodules for ordinary upstream dependencies. CMake
first-class packages and dependency providers remain usable, while the standard
presets fetch immutable, hash-verified source snapshots.

| Dependency | Snapshot | SHA-256 | Role |
| --- | --- | --- | --- |
| Abseil | `20260526.0` / `5650e9cf76d3be4318d5fa3af38ee483ddfd5e4a` | `bac569e31429f85d7bdd9f90e4767442faf9c8fa1def08a66a7d5d78e0fba204` | Core implementation utilities |
| GoogleTest | `v1.17.0` / `52eb8108c5bdec04579160ae17225d66034bd723` | `745c55415660044610f7fcd3af7a6420d5de16a7dbb9ebfe2e131275676232be` | Tests and mocks |
| YalantingLibs | `0.6.1` | `2ef2089a49a08f764c558e9caf46e1d37697b111e04a48e5c5156f57f3afff24` | Internal `struct_pack` wire codec |
| OpenCL-Headers | `v2026.05.29` | `d9e6c48357de5002da11ce45de600e0c3ffe6ab4f628a3b9fe2b38603161658a` | OpenCL 3.1.1 API surface |
| OpenCL-ICD-Loader | `v2026.05.29` | `48fd0c5181db7cd046f4f731d5955694892e10998d49d09ee0d997e7e04fd939` | Portable OpenCL dispatch loader |

The authoritative executable lock is `cmake/Dependencies.cmake`. Updating a
dependency requires changing both its immutable source location and hash,
running every build preset, and updating this table in the same change.

Git submodules remain reserved for an upstream that Plait must patch long-term,
or for a separately audited source tree whose exact checkout must be available
without CMake configuration. Neither case applies to the initial dependencies.
