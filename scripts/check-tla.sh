#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tlc -deadlock -config specs/PlaitExecutionGraphSmall.cfg specs/PlaitExecutionGraphSmall.tla
tlc -deadlock -config specs/PlaitExecutionGraphFive.cfg specs/PlaitExecutionGraphFive.tla
tlc -deadlock -config specs/PlaitOrderingSmall.cfg specs/PlaitOrderingSmall.tla
tlc -deadlock -config specs/PlaitOrderingTiny.cfg specs/PlaitOrderingTiny.tla
tlc -deadlock -config specs/PlaitUnifiedSmall.cfg specs/PlaitUnifiedSmall.tla
tlc -deadlock -config specs/PlaitUnifiedTiny.cfg specs/PlaitUnifiedTiny.tla
