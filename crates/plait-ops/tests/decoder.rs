use std::cell::Cell;
use std::rc::Rc;

use half::f16;
use plait_ops::OperatorError;
use plait_ops::decoder::{DecoderBlock, DecoderConfig, DecoderStack, LinearProjection, TokenHead};

struct Projection {
    shape: (u32, u32),
    weights: Vec<f32>,
    fail: bool,
    inject_failure: Rc<Cell<bool>>,
}

impl LinearProjection for Projection {
    fn shape(&self) -> (u32, u32) {
        self.shape
    }

    fn forward(&self, input: &[u16]) -> Result<Vec<f32>, OperatorError> {
        if self.fail && self.inject_failure.get() {
            return Err(OperatorError::Backend("projection failed".into()));
        }
        Ok(input
            .chunks_exact(self.shape.0 as usize)
            .flat_map(|row| {
                self.weights.chunks_exact(row.len()).map(move |weights| {
                    row.iter()
                        .zip(weights)
                        .map(|(&input, &weight)| f16::from_bits(input).to_f32() * weight)
                        .sum()
                })
            })
            .collect())
    }
}

fn block(layer: [u8; 32], failure: Rc<Cell<bool>>) -> DecoderBlock<Projection> {
    let config = DecoderConfig {
        hidden_size: 8,
        intermediate_size: 12,
        attention_heads: 4,
        kv_heads: 2,
        rope_theta: 10_000.0,
        rms_epsilon: 1e-6,
    };
    let shapes = config.projection_shapes().unwrap();
    let projections = std::array::from_fn(|index| {
        let (k, n) = shapes[index];
        let weights = (0..k * n)
            .map(|i| ((i * 19 + (index as u32) * 11) % 23) as f32 / 128.0 - 0.085)
            .collect();
        Projection {
            shape: (k, n),
            weights,
            fail: index == 6,
            inject_failure: failure.clone(),
        }
    });
    let norm = [f16::ONE.to_bits(); 8];
    DecoderBlock::new(layer, config, &norm, &norm, projections).unwrap()
}

#[test]
fn prefix_kv_can_move_to_a_new_executor_and_preserves_causal_results() {
    let failure = Rc::new(Cell::new(false));
    let prefill = block([7; 32], failure.clone());
    let decode = block([7; 32], failure);
    let data: Vec<_> = (0..32)
        .map(|i| ((i * 31 % 29) - 14) as f32 / 32.0)
        .collect();
    let mut reference = prefill.empty_cache(4).unwrap();
    let expected = prefill.forward(&data, &mut reference).unwrap();
    let mut prefix = prefill.empty_cache(4).unwrap();
    let mut actual = prefill.forward(&data[..16], &mut prefix).unwrap();
    let mut resumed = decode.restore_cache(prefix.snapshot().clone(), 4).unwrap();
    for token in data[16..].chunks_exact(8) {
        actual.extend(decode.forward(token, &mut resumed).unwrap());
    }
    assert_eq!(expected, actual);
    assert_eq!(reference.snapshot(), resumed.snapshot());
}

#[test]
fn failed_projections_and_invalid_snapshots_cannot_advance_kv() {
    let failure = Rc::new(Cell::new(false));
    let block = block([7; 32], failure.clone());
    let mut cache = block.empty_cache(2).unwrap();
    block.forward(&[0.5; 8], &mut cache).unwrap();
    let snapshot = cache.snapshot().clone();
    failure.set(true);
    assert!(block.forward(&[0.25; 8], &mut cache).is_err());
    assert_eq!(&snapshot, cache.snapshot());
    failure.set(false);
    assert!(block.forward(&[0.25; 16], &mut cache).is_err());
    assert_eq!(&snapshot, cache.snapshot());
    let mut invalid = snapshot.clone();
    invalid.layer = [8; 32];
    assert!(block.restore_cache(invalid, 2).is_err());
    let mut invalid = snapshot;
    invalid.values_f16[0] = f16::NAN.to_bits();
    assert!(block.restore_cache(invalid, 2).is_err());
}

#[test]
fn stack_rolls_back_earlier_layers_and_rejects_misaligned_snapshots() {
    let failure = Rc::new(Cell::new(false));
    let stack = DecoderStack::new(vec![
        block([7; 32], Rc::new(Cell::new(false))),
        block([8; 32], failure.clone()),
    ])
    .unwrap();
    let mut cache = stack.empty_cache(4).unwrap();
    stack.forward(&[0.5; 8], &mut cache).unwrap();
    let prefix: Vec<_> = cache.snapshots().cloned().collect();
    failure.set(true);
    assert!(stack.forward(&[0.25; 8], &mut cache).is_err());
    assert_eq!(cache.tokens(), 1);
    assert!(cache.snapshots().eq(prefix.iter()));
    failure.set(false);
    let mut restored = stack.restore_cache(prefix.clone(), 4).unwrap();
    assert_eq!(
        stack.forward(&[0.25; 8], &mut cache).unwrap(),
        stack.forward(&[0.25; 8], &mut restored).unwrap()
    );
    let mut reordered = prefix.clone();
    reordered.swap(0, 1);
    assert!(stack.restore_cache(reordered, 4).is_err());
    let mut misaligned = prefix;
    misaligned[1].keys_f16.clear();
    misaligned[1].values_f16.clear();
    assert!(stack.restore_cache(misaligned, 4).is_err());
}

#[test]
fn generation_leaves_the_last_token_pending_and_rolls_back_head_failure() {
    let stack = DecoderStack::new(vec![block([7; 32], Rc::new(Cell::new(false)))]).unwrap();
    let fail = Rc::new(Cell::new(false));
    let half = f16::ONE.to_bits();
    let embeddings = [half; 24].into_iter().flat_map(u16::to_le_bytes).collect();
    let projection = Projection {
        shape: (8, 3),
        weights: vec![0.0; 24],
        fail: true,
        inject_failure: fail.clone(),
    };
    let head = TokenHead::new(embeddings, &[half; 8], projection, 1e-6).unwrap();
    let mut cache = stack.empty_cache(3).unwrap();
    let output = head.generate(&stack, &mut cache, &[1, 2], 2, &[0]).unwrap();
    assert_eq!(output, [0]);
    assert_eq!(cache.tokens(), 2);
    let prior: Vec<_> = cache.snapshots().cloned().collect();
    fail.set(true);
    assert!(head.generate(&stack, &mut cache, &output, 1, &[]).is_err());
    fail.set(false);
    assert!(head.generate(&stack, &mut cache, &output, 2, &[]).is_err());
    assert_eq!(cache.tokens(), 2);
    assert!(cache.snapshots().eq(prior.iter()));
    assert_eq!(
        head.generate(&stack, &mut cache, &output, 1, &[]).unwrap(),
        [0]
    );
    assert_eq!(cache.tokens(), 3);
}
