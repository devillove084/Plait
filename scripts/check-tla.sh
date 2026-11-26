#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tlc -deadlock -config specs/PlaitExecutionGraphSmall.cfg specs/PlaitExecutionGraphSmall.tla
tlc -deadlock -config specs/PlaitExecutionGraphFive.cfg specs/PlaitExecutionGraphFive.tla
tlc -deadlock -config specs/PlaitOrderingSmall.cfg specs/PlaitOrderingSmall.tla
tlc -deadlock -config specs/PlaitOrderingTiny.cfg specs/PlaitOrderingTiny.tla
tlc -deadlock -config specs/PlaitUnifiedSmall.cfg specs/PlaitUnifiedSmall.tla
tlc -deadlock -config specs/PlaitUnifiedTiny.cfg specs/PlaitUnifiedTiny.tla
tlc -deadlock -config specs/PlaitOrderingLive.cfg specs/PlaitOrderingSmall.tla
tlc -deadlock -config specs/PlaitOrderingNetwork.cfg specs/PlaitOrderingNetwork.tla
tlc -deadlock -config specs/PlaitAsyncReduction.cfg specs/PlaitAsyncReduction.tla
tlc -deadlock -config specs/PlaitAsyncReductionStalled.cfg specs/PlaitAsyncReduction.tla
tlc -deadlock -config specs/PlaitAsyncReductionRevoked.cfg specs/PlaitAsyncReduction.tla
tlc -deadlock -config specs/PlaitAsyncReductionSharedWorker.cfg specs/PlaitAsyncReduction.tla
tlc -deadlock -config specs/PlaitAsyncReductionHeldPull.cfg specs/PlaitAsyncReduction.tla
tlc -deadlock -config specs/PlaitAsyncReductionExpired.cfg specs/PlaitAsyncReduction.tla
tlc -deadlock -config specs/PlaitAsyncReductionFailed.cfg specs/PlaitAsyncReduction.tla
tlc -deadlock -config specs/PlaitResidentReclaim.cfg specs/PlaitResidentReclaim.tla
tlc -deadlock -config specs/PlaitResidentReclaimVerified.cfg specs/PlaitResidentReclaim.tla
tlc -deadlock -config specs/PlaitEpochAdmission.cfg specs/PlaitEpochAdmission.tla
tlc -deadlock -config specs/PlaitHostLaunch.cfg specs/PlaitHostLaunch.tla
tlc -deadlock -config specs/PlaitServingIngress.cfg specs/PlaitServingIngress.tla
tlc -deadlock -config specs/PlaitCompletionReplay.cfg specs/PlaitCompletionReplay.tla
tlc -deadlock -config specs/PlaitKvPublication.cfg specs/PlaitKvPublication.tla
tlc -deadlock -config specs/PlaitKernelPrimitivesCheck.cfg specs/PlaitKernelPrimitivesCheck.tla
tlc -deadlock -config specs/PlaitW4A4Check.cfg specs/PlaitW4A4Check.tla
tlc -deadlock -config specs/PlaitW4OperatorsCheck.cfg specs/PlaitW4OperatorsCheck.tla
tlc -deadlock -config specs/PlaitOrderedPublication.cfg specs/PlaitOrderedPublication.tla
tlc -deadlock -config specs/PlaitPeerCollective.cfg specs/PlaitPeerCollective.tla
tlc -deadlock -config specs/PlaitPeerCollectiveOdd.cfg specs/PlaitPeerCollective.tla
tlc -deadlock -config specs/PlaitPeerCollectiveReplicatedLoss.cfg specs/PlaitPeerCollective.tla
