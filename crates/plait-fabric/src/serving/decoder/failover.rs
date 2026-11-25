use std::collections::{BTreeMap, BTreeSet};

use plait_core::serving::{
    DecoderFailover, DecoderStandby, GroupPhase, GroupVersion, PdRoute, RouteSetId, ServingError,
    ServingOutcome, ServingResult, ServingRole,
};

use super::{RouteSet, ServingState};

#[derive(Clone)]
pub(super) struct Standby {
    alternatives: Vec<PdRoute>,
    fingerprint: [u8; 32],
}

impl Standby {
    pub(super) fn has_decode_alternative(&self, path: PdRoute) -> bool {
        self.alternatives
            .iter()
            .any(|candidate| candidate.prefill == path.prefill && candidate.decode != path.decode)
    }

    pub(super) fn has_prefill_alternative(&self, path: PdRoute) -> bool {
        self.alternatives
            .iter()
            .any(|candidate| candidate.prefill != path.prefill)
    }
}

#[cfg(test)]
mod tests {
    use super::super::RecoveryPolicy;
    use super::*;
    use plait_core::serving::{
        DecoderEndpoint, DecoderEntryRetry, DecoderKvSource, DecoderOffer, DecoderPreparation,
        DecoderRouteWeight, DecoderRouteWork, DecoderStep, DecoderSubmission, GroupId, GroupPlan,
        HostMemory, ResidentOwner, WorkerDemand, WorkerGrant,
    };
    use plait_core::{DeviceId, NodeId, OpId};

    fn prepare(
        state: &mut ServingState,
        ordinal: u32,
        role: ServingRole,
        fingerprint: [u8; 32],
    ) -> (WorkerGrant, ResidentOwner) {
        let (grant, owner) = prepare_pending(state, ordinal, role);
        ready(state, grant, fingerprint);
        (grant, owner)
    }

    fn prepare_pending(
        state: &mut ServingState,
        ordinal: u32,
        role: ServingRole,
    ) -> (WorkerGrant, ResidentOwner) {
        let host = NodeId::derive("standby-test");
        let device = DeviceId::new(host, ordinal);
        let group = GroupId(NodeId::derive(&format!("standby-{ordinal}")));
        let version = GroupVersion { group, epoch: 1 };
        let memory = HostMemory {
            bytes: 1,
            pinned_bytes: 0,
        };
        state
            .set_host_memory(
                host,
                HostMemory {
                    bytes: 8,
                    pinned_bytes: 0,
                },
            )
            .unwrap();
        state.register_worker(device, 1, 32).unwrap();
        state
            .prepare(
                group,
                0,
                &GroupPlan {
                    role,
                    model: "model".into(),
                    tensor_parallel: 1,
                    pipeline_parallel: 1,
                    launch_digest: [1; 32],
                    workers: BTreeMap::from([(
                        device,
                        WorkerDemand {
                            incarnation: 1,
                            device_bytes: 1,
                        },
                    )]),
                    hosts: BTreeMap::from([(host, memory)]),
                },
            )
            .unwrap();
        let grant = state.group(version).unwrap().grants[&device];
        let owner = ResidentOwner {
            session: NodeId::derive("standby-owner"),
            supervisor_lock: [0; 2],
            device_lock: [0; 2],
        };
        state.claim_resident(grant, owner, [1; 32]).unwrap();
        (grant, owner)
    }

    fn ready(state: &mut ServingState, grant: WorkerGrant, fingerprint: [u8; 32]) {
        state
            .decoder_ready(
                DecoderEndpoint {
                    grant,
                    addr: "127.0.0.1:9100".parse().unwrap(),
                    fingerprint,
                },
                [1; 32],
                true,
            )
            .unwrap();
    }

    #[test]
    fn routed_admission_weights_per_path_and_rejects_an_incomplete_manifest() {
        let mut state = ServingState::default();
        let (p, _) = prepare(&mut state, 0, ServingRole::Prefill, [9; 32]);
        let (d, _) = prepare(&mut state, 1, ServingRole::Decode, [9; 32]);
        let (fast, _) = prepare(&mut state, 2, ServingRole::Decode, [9; 32]);
        for version in [p.version, d.version, fast.version] {
            state.activate(version).unwrap();
        }
        let a = PdRoute {
            prefill: p.version,
            decode: d.version,
        };
        let b = PdRoute {
            prefill: p.version,
            decode: fast.version,
        };
        let key = RouteSetId::for_model("model");
        let id = |n| OpId::new(NodeId::derive("weighted-native-admissions"), n);
        state
            .publish_decoder_routes("model", 0, &[a, b], None, &[])
            .unwrap();
        state
            .configure_decoder_weights(
                "model",
                1,
                &[
                    DecoderRouteWeight { path: a, weight: 1 },
                    DecoderRouteWeight { path: b, weight: 3 },
                ],
            )
            .unwrap();
        let routes = state.decoder_admission_routes(key);
        let mut incomplete = routes.clone();
        incomplete.paths.pop();
        assert_eq!(
            state.admit_decoder_routed("model", &incomplete, id(1), 1, None),
            Err(ServingError::InvalidDirectory)
        );
        for (index, expected) in [b, b, a, b].into_iter().enumerate() {
            let outcome = state
                .admit_decoder_routed("model", &routes, id(index as u64 + 2), 1, None)
                .unwrap();
            let ServingOutcome::DecoderAdmitted { route, .. } = outcome else {
                panic!("wrong admission")
            };
            assert_eq!(route.path(), expected);
        }
        state.quiesce(fast.version, GroupPhase::Revoked).unwrap();
        let ServingOutcome::DecoderAdmitted { route, .. } = state
            .admit_decoder_routed("model", &routes, id(6), 2, None)
            .unwrap()
        else {
            panic!("wrong admission")
        };
        assert_eq!(route.path(), a);
    }

    #[test]
    fn expansion_publishes_staged_weights_and_keeps_a_zero_weight_path_active() {
        let mut state = ServingState::default();
        let (p, _) = prepare(&mut state, 0, ServingRole::Prefill, [9; 32]);
        let (d, _) = prepare(&mut state, 1, ServingRole::Decode, [9; 32]);
        let (added, _) = prepare_pending(&mut state, 2, ServingRole::Decode);
        for version in [p.version, d.version] {
            state.activate(version).unwrap();
        }
        let old_path = PdRoute {
            prefill: p.version,
            decode: d.version,
        };
        let new_path = PdRoute {
            prefill: p.version,
            decode: added.version,
        };
        let paths = [old_path, new_path];
        let weights = [
            DecoderRouteWeight {
                path: old_path,
                weight: 0,
            },
            DecoderRouteWeight {
                path: new_path,
                weight: 5,
            },
        ];
        let key = RouteSetId::for_model("model");
        let id = |n| OpId::new(NodeId::derive("weighted-expansion"), n);
        state
            .publish_decoder_routes("model", 0, &[old_path], None, &[])
            .unwrap();
        assert_eq!(
            state.prepare_decoder_switch("model", 1, &[], &paths, &[], id(1), Some(&weights[..1])),
            Err(ServingError::InvalidDirectory),
        );
        assert!(state.decoder_switch(key).is_none());
        state
            .prepare_decoder_switch("model", 1, &[], &paths, &[], id(2), Some(&weights))
            .unwrap();
        assert_eq!(
            state.prepare_decoder_switch("model", 1, &[], &paths, &[], id(3), Some(&weights)),
            Err(ServingError::DecoderSwitchInProgress { staged: id(2) }),
        );
        assert!(state.decoder_route_work(key).is_none());
        assert_eq!(state.decoder_admission_routes(key).paths, [old_path]);
        ready(&mut state, added, [9; 32]);
        let Some(DecoderRouteWork::Switch(intent)) = state.decoder_route_work(key) else {
            panic!("expansion should be ready");
        };
        state.commit_decoder_switch("model", &intent).unwrap();
        assert_eq!(state.group(d.version).unwrap().phase, GroupPhase::Active);
        let routes = state.decoder_admission_routes(key);
        assert_eq!(routes.paths, paths);
        assert_eq!(routes.revision, 2);
        assert_eq!(
            state.configure_decoder_weights("model", 2, &weights),
            Err(ServingError::InvalidDirectory),
        );
        state.quiesce(added.version, GroupPhase::Revoked).unwrap();
        assert_eq!(
            state.admit_decoder_routed("model", &routes, id(4), 1, None),
            Err(ServingError::InvalidDirectory),
        );
    }

    #[test]
    fn preparing_decoder_layout_reserves_both_roles_or_neither() {
        let mut state = ServingState::default();
        let (p, _) = prepare(&mut state, 0, ServingRole::Prefill, [9; 32]);
        let (d, _) = prepare(&mut state, 1, ServingRole::Decode, [9; 32]);
        state.activate(p.version).unwrap();
        state.activate(d.version).unwrap();
        let old = PdRoute {
            prefill: p.version,
            decode: d.version,
        };
        state
            .publish_decoder_routes("model", 0, &[old], None, &[])
            .unwrap();
        let make = |state: &mut ServingState, incumbent: WorkerGrant, ordinal| {
            let device = DeviceId::new(incumbent.device.node, ordinal);
            state.register_worker(device, 1, 32).unwrap();
            let mut plan = state.group(incumbent.version).unwrap().plan.clone();
            plan.workers = BTreeMap::from([(
                device,
                WorkerDemand {
                    incarnation: 1,
                    device_bytes: 1,
                },
            )]);
            plan.hosts.get_mut(&device.node).unwrap().bytes = 4;
            DecoderPreparation {
                group: incumbent.version.group,
                expected_epoch: 1,
                expected_active: Some(1),
                plan,
            }
        };
        let mut placements = [make(&mut state, p, 2), make(&mut state, d, 3)];
        let new = PdRoute {
            prefill: GroupVersion {
                epoch: 2,
                ..p.version
            },
            decode: GroupVersion {
                epoch: 2,
                ..d.version
            },
        };
        let key = RouteSetId::for_model("model");
        let id = OpId::new(NodeId::derive("prepare-layout"), 1);
        assert_eq!(
            state.prepare_decoder_switch("model", 1, &placements, &[new], &[], id, None),
            Err(ServingError::HostCapacityExceeded(p.device.node)),
        );
        assert!(state.group(new.prefill).is_none() && state.group(new.decode).is_none());
        assert!(state.decoder_switch(key).is_none());
        for placement in &mut placements {
            placement.plan.hosts.get_mut(&p.device.node).unwrap().bytes = 2;
        }
        assert_eq!(
            state.prepare_decoder_switch("model", 0, &placements, &[new], &[], id, None),
            Err(ServingError::StaleDirectory { current: 1 })
        );
        assert!(state.group(new.prefill).is_none() && state.group(new.decode).is_none());
        assert_eq!(
            state.prepare_decoder_switch("model", 1, &placements, &[new], &[], id, None),
            Ok(ServingOutcome::DecoderSwitchStaged { revision: 1 })
        );
        assert_eq!(
            state.group(new.prefill).unwrap().phase,
            GroupPhase::Preparing
        );
        assert_eq!(
            state.group(new.decode).unwrap().phase,
            GroupPhase::Preparing
        );
        assert_eq!(state.decoder_switch(key).unwrap().paths, [new]);
    }

    #[test]
    fn decoder_pair_waits_for_both_roles_and_drains_only_removed_groups() {
        let mut state = ServingState::default();
        let (p, _) = prepare(&mut state, 0, ServingRole::Prefill, [9; 32]);
        let (d, _) = prepare(&mut state, 1, ServingRole::Decode, [9; 32]);
        let (new_p, _) = prepare(&mut state, 2, ServingRole::Prefill, [9; 32]);
        let (new_d, _) = prepare_pending(&mut state, 3, ServingRole::Decode);
        state.activate(p.version).unwrap();
        state.activate(d.version).unwrap();
        let old = PdRoute {
            prefill: p.version,
            decode: d.version,
        };
        let new = PdRoute {
            prefill: new_p.version,
            decode: new_d.version,
        };
        let key = RouteSetId::for_model("model");
        let op = |counter| OpId::new(NodeId::derive("paired-decoder-switch"), counter);
        state
            .publish_decoder_routes("model", 0, &[old], None, &[])
            .unwrap();
        state
            .admit_decoder("model", 1, old, op(1), 1, RecoveryPolicy::None)
            .unwrap();
        state
            .stage_decoder_switch("model", 1, &[new], &[], op(2))
            .unwrap();
        let intent = state.decoder_switch(key).unwrap();
        assert!(state.decoder_route_work(key).is_none());
        assert_eq!(
            state.commit_decoder_switch("model", &intent),
            Err(ServingError::InvalidPhase(GroupPhase::Preparing))
        );
        assert_eq!(state.group(new_p.version).unwrap().phase, GroupPhase::Ready);
        assert_eq!(state.decoder_routes(key).unwrap().paths[0].path(), old);
        ready(&mut state, new_d, [9; 32]);
        assert_eq!(
            state.decoder_route_work(key),
            Some(DecoderRouteWork::Switch(intent.clone()))
        );
        assert_eq!(
            state.commit_decoder_switch("model", &intent),
            Ok(ServingOutcome::RoutesPublished { revision: 2 })
        );
        assert!(state.decoder_switch(key).is_none());
        for version in [old.prefill, old.decode] {
            assert_eq!(state.group(version).unwrap().phase, GroupPhase::Draining);
        }
        for version in [new.prefill, new.decode] {
            assert_eq!(state.group(version).unwrap().phase, GroupPhase::Active);
        }
        assert_eq!(state.decoder_routes(key).unwrap().paths[0].path(), new);
        state
            .advance_decoder("model", op(1), old, p, 2, DecoderStep::StartPrefill, op(3))
            .unwrap();
        state
            .advance_decoder(
                "model",
                op(1),
                old,
                p,
                2,
                DecoderStep::Finish { owner: op(3) },
                op(4),
            )
            .unwrap();
        assert_eq!(state.group(old.decode).unwrap().phase, GroupPhase::Revoked);
        assert_eq!(
            state.admit_decoder("model", 1, old, op(5), 3, RecoveryPolicy::None),
            Err(ServingError::StaleDirectory { current: 2 })
        );
        let (wrong_p, _) = prepare(&mut state, 4, ServingRole::Prefill, [8; 32]);
        let (wrong_d, _) = prepare(&mut state, 5, ServingRole::Decode, [8; 32]);
        let wrong = PdRoute {
            prefill: wrong_p.version,
            decode: wrong_d.version,
        };
        state
            .stage_decoder_switch("model", 2, &[wrong], &[], op(6))
            .unwrap();
        assert!(state.decoder_route_work(key).is_none());
        let invalid = state.decoder_switch(key).unwrap();
        assert_eq!(
            state.commit_decoder_switch("model", &invalid),
            Err(ServingError::InvalidRoute(wrong))
        );
        assert_eq!(
            state.group(wrong_p.version).unwrap().phase,
            GroupPhase::Ready
        );
        assert_eq!(
            state.cancel_decoder_switch("model", op(6)),
            Ok(ServingOutcome::DecoderSwitchCancelled)
        );
    }

    #[test]
    fn stable_submission_keeps_one_work_and_never_recycles_an_expired_id() {
        let mut state = ServingState::default();
        let (p, _) = prepare(&mut state, 0, ServingRole::Prefill, [9; 32]);
        let (d, _) = prepare(&mut state, 1, ServingRole::Decode, [9; 32]);
        state.activate(p.version).unwrap();
        state.activate(d.version).unwrap();
        let path = PdRoute {
            prefill: p.version,
            decode: d.version,
        };
        state
            .publish_decoder_routes("model", 0, &[path], None, &[])
            .unwrap();
        let id = NodeId::derive("stable-submission");
        let origin = NodeId::derive("submission-operations");
        let work = DecoderSubmission::work(id);
        let entry = OpId::new(origin, 3);
        let request = DecoderSubmission {
            id,
            sequence: 2,
            offset: 0,
            max_tokens: 1,
            input_sha256: [2; 32],
        };
        let submit = |state: &mut ServingState, counter, now, request| {
            state.submit_decoder(
                "model",
                OpId::new(origin, counter),
                1,
                Some(path),
                None,
                now,
                request,
            )
        };
        assert!(submit(&mut state, 1, 1, request).is_ok());
        assert_eq!(
            submit(&mut state, 2, 2, request),
            Err(ServingError::StaleDecoderEntry)
        );
        let next = DecoderSubmission {
            sequence: 3,
            ..request
        };
        assert!(matches!(
            submit(&mut state, 3, 3, next),
            Ok(ServingOutcome::DecoderTakenOver { committed: 0, .. })
        ));
        let owner = OpId::new(origin, 4);
        state
            .advance_decoder(
                "model",
                work,
                path,
                p,
                4,
                DecoderStep::StartPrefillEntry { entry },
                owner,
            )
            .unwrap();
        state
            .advance_decoder(
                "model",
                work,
                path,
                p,
                5,
                DecoderStep::FinishPrefillToken { owner, token: 7 },
                OpId::new(origin, 5),
            )
            .unwrap();
        assert_eq!(
            state.submit_decoder("model", OpId::new(origin, 6), 0, None, None, 6, next),
            Ok(ServingOutcome::DecoderSubmissionComplete {
                revision: 1,
                path,
                count: 1
            }),
        );
        assert_eq!(
            submit(
                &mut state,
                7,
                7,
                DecoderSubmission {
                    input_sha256: [3; 32],
                    ..next
                }
            ),
            Err(ServingError::InvalidDecoderOutput),
        );
        assert_eq!(
            state.submit_decoder(
                "another-model",
                OpId::new(origin, 8),
                1,
                None,
                None,
                8,
                next
            ),
            Err(ServingError::OperationIdReused),
        );
        let later = 300_002;
        let other = DecoderSubmission {
            id: NodeId::derive("another-request"),
            ..next
        };
        assert!(submit(&mut state, 9, later, other).is_ok());
        assert!(
            state
                .decoder_replay(RouteSetId::for_model("model"), work, 0, later)
                .is_none()
        );
        assert_eq!(
            submit(&mut state, 10, later, next),
            Err(ServingError::DecoderNotAdmitted)
        );
    }

    #[test]
    fn takeover_retry_orders_lost_receipts_without_reviving_an_old_entry() {
        let mut state = ServingState::default();
        let (p, _) = prepare(&mut state, 0, ServingRole::Prefill, [9; 32]);
        let (d, _) = prepare(&mut state, 1, ServingRole::Decode, [9; 32]);
        state.activate(p.version).unwrap();
        state.activate(d.version).unwrap();
        let path = PdRoute {
            prefill: p.version,
            decode: d.version,
        };
        state
            .publish_decoder_routes("model", 0, &[path], None, &[])
            .unwrap();
        let origin = NodeId::derive("takeover-test");
        let work = OpId::new(origin, 1);
        state
            .admit_decoder_replay("model", 1, path, work, 1, 4, Some([4; 32]))
            .unwrap();
        let series = NodeId::derive("takeover-series");
        let old_ref = DecoderEntryRetry {
            series,
            sequence: 1,
        };
        let recover = |state: &mut ServingState, counter, expected, retry| {
            state.takeover_decoder(
                "model",
                work,
                expected,
                OpId::new(origin, counter),
                1,
                path,
                path,
                2,
                0,
                [4; 32],
                retry,
            )
        };

        let mut before_commit = state.clone();
        let latest = DecoderEntryRetry {
            series,
            sequence: 3,
        };
        assert!(recover(&mut before_commit, 3, work, Some(latest)).is_ok());
        assert_eq!(
            recover(&mut before_commit, 2, work, Some(old_ref)),
            Err(ServingError::StaleDecoderEntry)
        );

        assert!(recover(&mut state, 2, work, Some(old_ref)).is_ok());
        let different = DecoderEntryRetry {
            series: NodeId::derive("another-series"),
            ..latest
        };
        assert_eq!(
            recover(&mut state, 4, work, Some(different)),
            Err(ServingError::StaleDecoderEntry)
        );
        assert_eq!(
            recover(&mut state, 5, work, Some(old_ref)),
            Err(ServingError::StaleDecoderEntry)
        );
        assert!(recover(&mut state, 3, work, Some(latest)).is_ok());
        assert_eq!(
            state.check_decoder_entry("model", work, Some(OpId::new(origin, 2))),
            Err(ServingError::StaleDecoderEntry)
        );
        assert!(recover(&mut state, 6, OpId::new(origin, 3), None).is_ok());
        let delayed = DecoderEntryRetry {
            sequence: 4,
            ..latest
        };
        assert_eq!(
            recover(&mut state, 7, work, Some(delayed)),
            Err(ServingError::StaleDecoderEntry)
        );
    }

    #[test]
    fn failover_requires_worker_loss_and_preserves_the_primary_fingerprint() {
        let mut state = ServingState::default();
        let (p, _) = prepare(&mut state, 0, ServingRole::Prefill, [9; 32]);
        let (d, owner) = prepare(&mut state, 1, ServingRole::Decode, [9; 32]);
        let (wrong, _) = prepare(&mut state, 2, ServingRole::Decode, [8; 32]);
        let (spare, _) = prepare(&mut state, 3, ServingRole::Decode, [9; 32]);
        state.activate(p.version).unwrap();
        state.activate(d.version).unwrap();
        let path = |decode| PdRoute {
            prefill: p.version,
            decode,
        };
        let primary = path(d.version);
        let standby = DecoderStandby {
            primary,
            alternatives: vec![path(wrong.version), path(spare.version)],
        };
        let key = RouteSetId::for_model("model");
        state
            .publish_decoder_routes("model", 0, &[primary], None, std::slice::from_ref(&standby))
            .unwrap();
        let mut manually_revoked = state.clone();
        manually_revoked
            .quiesce(d.version, GroupPhase::Revoked)
            .unwrap();
        assert!(manually_revoked.decoder_failover(key).is_none());
        state.lose_resident(d, owner).unwrap();
        state.worker_stopped(d).unwrap();
        state
            .retire(d.version, &BTreeSet::from([d.device]))
            .unwrap();
        let intent = state.decoder_failover(key).unwrap();
        assert_eq!(
            intent,
            DecoderFailover {
                revision: 1,
                standby
            }
        );
        assert_eq!(
            state.failover_decoder("model", &intent),
            Ok(ServingOutcome::RoutesPublished { revision: 2 })
        );
        assert_eq!(state.group(wrong.version).unwrap().phase, GroupPhase::Ready);
        assert_eq!(state.group(p.version).unwrap().phase, GroupPhase::Active);
        assert_eq!(
            state.decoder_routes(key).unwrap().paths[0].path(),
            path(spare.version)
        );
        assert_eq!(
            state.failover_decoder("model", &intent),
            Err(ServingError::StaleDirectory { current: 2 })
        );
    }

    #[test]
    fn decoding_claim_can_be_rebound_after_fault_using_the_original_offer() {
        let mut state = ServingState::default();
        let (p, _) = prepare(&mut state, 0, ServingRole::Prefill, [9; 32]);
        let (d, owner) = prepare(&mut state, 1, ServingRole::Decode, [9; 32]);
        let (spare, _) = prepare(&mut state, 2, ServingRole::Decode, [9; 32]);
        state.activate(p.version).unwrap();
        state.activate(d.version).unwrap();
        let primary = PdRoute {
            prefill: p.version,
            decode: d.version,
        };
        let replacement = PdRoute {
            prefill: p.version,
            decode: spare.version,
        };
        let key = RouteSetId::for_model("model");
        state
            .publish_decoder_routes(
                "model",
                0,
                &[primary],
                None,
                &[DecoderStandby {
                    primary,
                    alternatives: vec![replacement],
                }],
            )
            .unwrap();
        let id = |counter| OpId::new(NodeId::derive("recover-test"), counter);
        let [work, prefill, old_owner, new_owner] = [1, 2, 3, 4].map(id);
        let offer = DecoderOffer {
            source: DecoderKvSource {
                addr: "127.0.0.1:9000".parse().unwrap(),
                transfer: NodeId::derive("recover-transfer"),
                bytes: 4,
                sha256: [8; 32],
            },
            first_token: 3,
            prefix_tokens: 2,
        };
        assert!(matches!(
            state.admit_decoder("model", 1, primary, work, 1, RecoveryPolicy::Decode),
            Ok(ServingOutcome::DecoderRecoveryAdmitted { .. })
        ));
        let mut advance = |path, grant, step, operation| {
            state.advance_decoder("model", work, path, grant, 2, step, operation)
        };
        advance(primary, p, DecoderStep::StartPrefill, prefill).unwrap();
        advance(
            primary,
            p,
            DecoderStep::Offer {
                owner: prefill,
                kv: offer,
            },
            id(5),
        )
        .unwrap();
        advance(
            primary,
            d,
            DecoderStep::ClaimDecode {
                transfer: offer.source.transfer,
            },
            old_owner,
        )
        .unwrap();
        assert_eq!(state.decoder_drain(p.version).unwrap().pending, 1);
        assert_eq!(
            state.recover_decoder_work("model", work, 1, primary, replacement, 2, false),
            Err(ServingError::InvalidDecoderStage)
        );
        state.lose_resident(d, owner).unwrap();
        let intent = state.decoder_failover(key).unwrap();
        state.failover_decoder("model", &intent).unwrap();
        assert_eq!(
            state.recover_decoder_work("model", work, 2, primary, replacement, 120_001, false),
            Err(ServingError::DecoderNotAdmitted)
        );
        state
            .recover_decoder_work("model", work, 2, primary, replacement, 2, false)
            .unwrap();
        assert_eq!(
            state.advance_decoder(
                "model",
                work,
                primary,
                d,
                2,
                DecoderStep::Finish { owner: old_owner },
                id(6)
            ),
            Err(ServingError::DecoderNotAdmitted)
        );
        assert_eq!(
            state.advance_decoder(
                "model",
                work,
                replacement,
                spare,
                2,
                DecoderStep::ReadOffer,
                id(7)
            ),
            Ok(ServingOutcome::DecoderOffered(offer))
        );
        state
            .advance_decoder(
                "model",
                work,
                replacement,
                spare,
                2,
                DecoderStep::ClaimDecode {
                    transfer: offer.source.transfer,
                },
                new_owner,
            )
            .unwrap();
        state.quiesce(p.version, GroupPhase::Draining).unwrap();
        assert_eq!(state.group(p.version).unwrap().phase, GroupPhase::Draining);
        state
            .advance_decoder(
                "model",
                work,
                replacement,
                spare,
                2,
                DecoderStep::Finish { owner: new_owner },
                id(8),
            )
            .unwrap();
        assert_eq!(state.group(p.version).unwrap().phase, GroupPhase::Revoked);
    }

    #[test]
    fn prefill_loss_restarts_uninstalled_work_and_preserves_a_live_decode() {
        let mut state = ServingState::default();
        let (p, old_owner) = prepare(&mut state, 0, ServingRole::Prefill, [9; 32]);
        let (d, _) = prepare(&mut state, 1, ServingRole::Decode, [9; 32]);
        let (spare, _) = prepare(&mut state, 2, ServingRole::Prefill, [9; 32]);
        state.activate(p.version).unwrap();
        state.activate(d.version).unwrap();
        let primary = PdRoute {
            prefill: p.version,
            decode: d.version,
        };
        let replacement = PdRoute {
            prefill: spare.version,
            decode: d.version,
        };
        let key = RouteSetId::for_model("model");
        state
            .publish_decoder_routes(
                "model",
                0,
                &[primary],
                None,
                &[DecoderStandby {
                    primary,
                    alternatives: vec![replacement],
                }],
            )
            .unwrap();

        let id = |counter| OpId::new(NodeId::derive("prefill-loss-test"), counter);
        let staged = id(1);
        let claimed = id(2);
        let offer = DecoderOffer {
            source: DecoderKvSource {
                addr: "127.0.0.1:9000".parse().unwrap(),
                transfer: NodeId::derive("lost-prefill-kv"),
                bytes: 4,
                sha256: [8; 32],
            },
            first_token: 3,
            prefix_tokens: 2,
        };
        for (work, owner) in [(staged, id(3)), (claimed, id(4))] {
            assert!(matches!(
                state.admit_decoder("model", 1, primary, work, 1, RecoveryPolicy::Pipeline),
                Ok(ServingOutcome::DecoderRecoveryAdmitted { .. })
            ));
            state
                .advance_decoder(
                    "model",
                    work,
                    primary,
                    p,
                    2,
                    DecoderStep::StartPrefill,
                    owner,
                )
                .unwrap();
            state
                .advance_decoder(
                    "model",
                    work,
                    primary,
                    p,
                    2,
                    DecoderStep::Offer { owner, kv: offer },
                    id(5),
                )
                .unwrap();
        }
        state
            .advance_decoder(
                "model",
                claimed,
                primary,
                d,
                2,
                DecoderStep::ClaimDecode {
                    transfer: offer.source.transfer,
                },
                id(6),
            )
            .unwrap();
        state.lose_resident(p, old_owner).unwrap();
        state
            .failover_decoder("model", &state.decoder_failover(key).unwrap())
            .unwrap();
        assert_eq!(
            state.recover_decoder_work("model", claimed, 2, primary, replacement, 3, true),
            Err(ServingError::InvalidDecoderStage)
        );
        state
            .advance_decoder(
                "model",
                claimed,
                primary,
                d,
                3,
                DecoderStep::Finish { owner: id(6) },
                id(7),
            )
            .unwrap();

        state
            .recover_decoder_work("model", staged, 2, primary, replacement, 3, true)
            .unwrap();
        assert_eq!(
            state.advance_decoder(
                "model",
                staged,
                primary,
                d,
                3,
                DecoderStep::ClaimDecode {
                    transfer: offer.source.transfer
                },
                id(8)
            ),
            Err(ServingError::DecoderNotAdmitted)
        );
        assert_eq!(
            state.advance_decoder(
                "model",
                staged,
                replacement,
                d,
                3,
                DecoderStep::ClaimDecode {
                    transfer: offer.source.transfer
                },
                id(9)
            ),
            Err(ServingError::InvalidDecoderStage)
        );
        state
            .advance_decoder(
                "model",
                staged,
                replacement,
                spare,
                3,
                DecoderStep::StartPrefill,
                id(10),
            )
            .unwrap();
        assert_eq!(state.decoder_claims(spare.version)[0].work, staged);
    }
}

struct Replacement {
    path: PdRoute,
    activate: Vec<GroupVersion>,
}

impl ServingState {
    pub(super) fn decoder_standbys(
        &self,
        model: &str,
        fingerprints: &BTreeMap<PdRoute, [u8; 32]>,
        rules: &[DecoderStandby],
    ) -> Result<BTreeMap<PdRoute, Standby>, ServingError> {
        Self::decoder_standbys_using(fingerprints, rules, |path| self.valid_standby(model, path))
    }

    pub(super) fn decoder_standbys_using(
        fingerprints: &BTreeMap<PdRoute, [u8; 32]>,
        rules: &[DecoderStandby],
        valid: impl Fn(PdRoute) -> bool,
    ) -> Result<BTreeMap<PdRoute, Standby>, ServingError> {
        let mut standbys = BTreeMap::new();
        for rule in rules {
            let &fingerprint = fingerprints
                .get(&rule.primary)
                .ok_or(ServingError::InvalidRoute(rule.primary))?;
            if rule.alternatives.is_empty() || standbys.contains_key(&rule.primary) {
                return Err(ServingError::InvalidDirectory);
            }
            let mut unique = BTreeSet::new();
            for &path in &rule.alternatives {
                if path == rule.primary || !unique.insert(path) || !valid(path) {
                    return Err(ServingError::InvalidRoute(path));
                }
            }
            standbys.insert(
                rule.primary,
                Standby {
                    alternatives: rule.alternatives.clone(),
                    fingerprint,
                },
            );
        }
        Ok(standbys)
    }

    pub(super) fn valid_standby(&self, model: &str, path: PdRoute) -> bool {
        Self::valid_decoder_groups(model, path, |version| self.group(version))
    }

    pub(super) fn valid_decoder_groups<'a>(
        model: &str,
        path: PdRoute,
        group: impl Fn(GroupVersion) -> Option<&'a plait_core::serving::GroupView>,
    ) -> bool {
        path.prefill.group != path.decode.group
            && [
                (path.prefill, ServingRole::Prefill),
                (path.decode, ServingRole::Decode),
            ]
            .into_iter()
            .all(|(version, role)| {
                group(version).is_some_and(|view| {
                    view.plan.model == model
                        && view.plan.role == role
                        && view.grants.len() == 1
                        && matches!(
                            view.phase,
                            GroupPhase::Preparing | GroupPhase::Ready | GroupPhase::Active
                        )
                })
            })
    }

    pub(crate) fn decoder_failover(&self, id: RouteSetId) -> Option<DecoderFailover> {
        let directory = self.decoder.routes.get(&id)?;
        for &primary in &directory.paths {
            if let Some(standby) = directory.standbys.get(&primary)
                && self
                    .decoder_replacement(directory, primary, standby)
                    .is_some()
            {
                return Some(DecoderFailover {
                    revision: directory.revision,
                    standby: DecoderStandby {
                        primary,
                        alternatives: standby.alternatives.clone(),
                    },
                });
            }
        }
        None
    }

    pub(in crate::serving) fn failover_decoder(
        &mut self,
        model: &str,
        intent: &DecoderFailover,
    ) -> ServingResult {
        let key = RouteSetId::for_model(model);
        let directory = self
            .decoder
            .routes
            .get(&key)
            .filter(|directory| directory.model == model)
            .ok_or(ServingError::InvalidDirectory)?;
        if directory.revision != intent.revision {
            return Err(ServingError::StaleDirectory {
                current: directory.revision,
            });
        }
        let revision = directory
            .revision
            .checked_add(1)
            .ok_or(ServingError::SequenceExhausted)?;
        let primary = intent.standby.primary;
        let standby = directory
            .standbys
            .get(&primary)
            .filter(|standby| standby.alternatives == intent.standby.alternatives)
            .ok_or(ServingError::InvalidDirectory)?;
        let replacement = self
            .decoder_replacement(directory, primary, standby)
            .ok_or(ServingError::InvalidRoute(primary))?;
        for version in replacement.activate {
            self.activate(version)
                .expect("all replacement groups checked and distinct");
        }
        let directory = self
            .decoder
            .routes
            .get_mut(&key)
            .expect("known decoder directory");
        let mut standby = directory.standbys.remove(&primary).expect("known standby");
        standby
            .alternatives
            .retain(|&path| path != replacement.path);
        if !standby.alternatives.is_empty() {
            directory
                .standbys
                .entry(replacement.path)
                .or_insert(standby);
        }
        if directory.paths.contains(&replacement.path) {
            directory.paths.retain(|&path| path != primary);
        } else {
            let path = directory
                .paths
                .iter_mut()
                .find(|path| **path == primary)
                .expect("standby belongs to a declared path");
            *path = replacement.path;
        }
        directory.revision = revision;
        directory.admission_weights.clear();
        Ok(ServingOutcome::RoutesPublished { revision })
    }

    fn decoder_replacement(
        &self,
        directory: &RouteSet,
        primary: PdRoute,
        standby: &Standby,
    ) -> Option<Replacement> {
        let prefill_failed = self.decoder.faulted.contains(&primary.prefill);
        let decode_failed = self.decoder.faulted.contains(&primary.decode);
        if !prefill_failed && !decode_failed {
            return None;
        }
        standby.alternatives.iter().find_map(|&path| {
            if (primary.prefill != path.prefill) != prefill_failed
                || (primary.decode != path.decode) != decode_failed
            {
                return None;
            }
            let route = self.decoder_endpoints(&directory.model, path)?;
            if route.prefill.fingerprint != standby.fingerprint {
                return None;
            }
            let mut activate = Vec::new();
            for (new, faulted) in [(path.prefill, prefill_failed), (path.decode, decode_failed)] {
                match self.group(new)?.phase {
                    GroupPhase::Active => {}
                    GroupPhase::Ready
                        if faulted && self.groups.get(&new.group)?.active.is_none() =>
                    {
                        self.check_activation(new).ok()?;
                        activate.push(new);
                    }
                    _ => return None,
                }
            }
            Some(Replacement { path, activate })
        })
    }
}
