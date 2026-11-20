#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tlc -deadlock -config specs/PlaitExecutionGraphSmall.cfg specs/PlaitExecutionGraphSmall.tla
tlc -deadlock -config specs/PlaitExecutionGraphFive.cfg specs/PlaitExecutionGraphFive.tla
tlc -deadlock -config specs/PlaitEPaxos.cfg specs/PlaitEPaxos.tla
tlc -deadlock -config specs/PlaitUnifiedSmall.cfg specs/PlaitUnifiedSmall.tla
