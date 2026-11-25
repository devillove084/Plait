use half::f16;
use plait_ops::{LinearOutput, OutputType, W4A16, W4Weights};

fn half(value: f32) -> u16 {
    f16::from_f32(value).to_bits()
}

#[test]
fn odd_k_preserves_group_zero_points_and_bias() {
    let packed = [0xe1, 0xf4];
    let scales = [half(0.5), half(1.5)];
    let zero = [5, 3];
    let weights = W4Weights::new(3, 1, 2, &packed, &scales, Some(&zero)).unwrap();
    let activation = [half(1.0), half(2.0), half(3.0)];
    let bias = [half(0.25)];
    for (kind, expected) in [
        (OutputType::F16, LinearOutput::F16(vec![half(11.75)])),
        (OutputType::F32, LinearOutput::F32(vec![11.75])),
    ] {
        let call = W4A16::new(weights, &activation, Some(&bias), kind).unwrap();
        assert_eq!(call.run_cpu(), expected);
    }
    assert!(W4Weights::new(3, 1, 2, &packed, &scales, Some(&[5, 16])).is_err());
    assert!(W4A16::new(weights, &activation[..2], None, OutputType::F16).is_err());
}

#[cfg(feature = "native")]
#[test]
fn native_c_matches_rust_for_asymmetric_and_symmetric_groups() {
    use plait_ops::native::NativeW4A16;
    use std::path::Path;

    let Some(path) = std::env::var_os("PLAIT_NATIVE_W4A16_TEST_LIBRARY") else {
        return;
    };
    let ordinals = std::env::var("PLAIT_W4A16_TEST_DEVICES").unwrap_or_else(|_| "0".to_owned());
    // The test runner supplies a library compiled from kernels/w4a16*.{c,cu}.
    let backend = unsafe { NativeW4A16::load(Path::new(&path)) }.unwrap();
    for ordinal in ordinals.split(',').map(|text| text.parse::<i32>().unwrap()) {
        for (n, k, group_size) in [(3, 7, 3), (33, 257, 16), (65, 386, 128)] {
            let groups = u32::div_ceil(k, group_size);
            let packed: Vec<_> = (0..n * k.div_ceil(2))
                .map(|i| (i * 59 + 27) as u8)
                .collect();
            let scales: Vec<_> = (0..n * groups)
                .map(|i| half(((i * 13) % 29 + 1) as f32 / 128.0))
                .collect();
            let zeros: Vec<_> = (0..n * groups)
                .map(|i| ((i * 11 + i / 19) % 16) as u8)
                .collect();
            let input: Vec<_> = (0..5 * k)
                .map(|i| half(((i * 13) as i32 % 37 - 18) as f32 / 32.0))
                .collect();
            let bias: Vec<_> = (0..n).map(|i| half(i as f32 / 32.0)).collect();
            for zero in [None, Some(zeros.as_slice())] {
                let weights = W4Weights::new(k, n, group_size, &packed, &scales, zero).unwrap();
                for mode in [OutputType::F16, OutputType::F32] {
                    let call = W4A16::new(weights, &input, Some(&bias), mode).unwrap();
                    assert_eq!(backend.run(ordinal, call).unwrap(), call.run_cpu());
                    let empty = W4A16::new(weights, &[], Some(&bias), mode).unwrap();
                    assert_eq!(backend.run(ordinal, empty).unwrap(), empty.run_cpu());
                }
            }
        }
        println!("native W4A16 device {ordinal}: 12 layout/dtype comparisons passed");
    }
}

#[cfg(feature = "cuda-marlin")]
#[test]
fn marlin_runs_rust_packed_weights_with_cuda_only() {
    use plait_ops::marlin::MarlinDevice;
    use std::path::Path;

    let Some(path) = std::env::var_os("PLAIT_MARLIN_TEST_LIBRARY") else {
        return;
    };
    let ordinals = std::env::var("PLAIT_MARLIN_TEST_DEVICES").unwrap_or_else(|_| "0".to_owned());
    let expect_unsupported = std::env::var_os("PLAIT_MARLIN_EXPECT_UNSUPPORTED").is_some();
    let (n, k) = (512, 768);
    let mut state = 70280u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state as u32
    };
    let packed: Vec<_> = (0..n * k / 2).map(|_| next() as u8).collect();
    let decode = |bits: u16| f16::from_bits(bits).to_f32();
    for ordinal in ordinals.split(',').map(|text| text.parse::<i32>().unwrap()) {
        // The test runner supplies the native library from kernels/marlin.
        let backend = unsafe { MarlinDevice::load(ordinal, Path::new(&path)) };
        if expect_unsupported {
            assert!(matches!(
                backend,
                Err(plait_ops::OperatorError::Unsupported(_))
            ));
            println!(
                "legacy Marlin device {ordinal}: rejected unsupported architecture before loading"
            );
            continue;
        }
        let backend = backend.unwrap();
        for group_size in [128, k] {
            let scales: Vec<_> = (0..n * k / group_size)
                .map(|_| half((1 + next() % 13) as f32 / 128.0))
                .collect();
            let weights = W4Weights::new(k, n, group_size, &packed, &scales, None).unwrap();
            let linear = backend.prepare(weights).unwrap();
            for rows in [1, 16, 32, 65] {
                let input: Vec<_> = (0..rows * k)
                    .map(|_| half(((next() % 93) as i32 - 46) as f32 / 64.0))
                    .collect();
                let reference = W4A16::new(weights, &input, None, OutputType::F16)
                    .unwrap()
                    .run_cpu();
                let actual = linear.run(&input, None, OutputType::F16).unwrap();
                let (LinearOutput::F16(expected), LinearOutput::F16(actual)) = (reference, actual)
                else {
                    panic!("expected FP16 outputs");
                };
                let mut maximum = 0.0f32;
                for (index, (expected, actual)) in expected.into_iter().zip(actual).enumerate() {
                    let (expected, actual) = (decode(expected), decode(actual));
                    let error = (expected - actual).abs();
                    maximum = maximum.max(error);
                    assert!(
                        error <= 0.015 + 0.004 * expected.abs(),
                        "device {ordinal} M={rows} G={group_size} at {index}: {actual} vs {expected}"
                    );
                }
                println!(
                    "Marlin device {ordinal}: M={rows} N={n} K={k} G={group_size} max_abs_error={maximum}"
                );
            }
            assert!(linear.run(&[], None, OutputType::F32).is_err());
            let zero = vec![7; scales.len()];
            let asymmetric =
                W4Weights::new(k, n, group_size, &packed, &scales, Some(&zero)).unwrap();
            assert!(backend.prepare(asymmetric).is_err());
        }
    }
}

#[cfg(feature = "native")]
#[test]
fn native_chains_device_tensors_and_owns_the_queue() {
    use plait_ops::OperatorError;
    use plait_ops::native::NativeW4A16;
    use std::path::Path;

    let Some(path) = std::env::var_os("PLAIT_W4A16_RESIDENT_TEST_LIBRARY") else {
        return;
    };
    let ordinals = std::env::var("PLAIT_W4A16_TEST_DEVICES").unwrap_or_else(|_| "0".to_owned());
    let backend = unsafe { NativeW4A16::load(Path::new(&path)) }.unwrap();

    let scales_a = [
        half(0.5),
        half(1.5),
        half(0.25),
        half(2.0),
        half(0.75),
        half(2.0),
    ];
    let scales_b = [half(0.125), half(0.5)];
    let weights_a = W4Weights::new(
        3,
        3,
        2,
        &[0xe1, 0xf4, 0x56, 0xfa, 0x98, 0xf7],
        &scales_a,
        Some(&[5, 3, 7, 4, 9, 7]),
    )
    .unwrap();
    let weights_b = W4Weights::new(3, 2, 3, &[0x98, 0xff, 0x65, 0xf7], &scales_b, None).unwrap();
    let activation = [
        half(1.0),
        half(2.0),
        half(3.0),
        half(-1.0),
        half(-0.5),
        half(2.0),
    ];
    let bias = [half(0.25), half(-0.5), half(1.0)];
    let LinearOutput::F16(middle) =
        W4A16::new(weights_a, &activation, Some(&bias), OutputType::F16)
            .unwrap()
            .run_cpu()
    else {
        panic!("expected FP16 reference");
    };
    let expected = W4A16::new(weights_b, &middle, None, OutputType::F32)
        .unwrap()
        .run_cpu();

    for ordinal in ordinals.split(',').map(|text| text.parse::<i32>().unwrap()) {
        let device = backend.open_device(ordinal).unwrap();
        let a = device.prepare(weights_a).unwrap();
        let b = device.prepare(weights_b).unwrap();
        let input = device.upload_f16(2, 3, &activation).unwrap();
        let bias = device.upload_f16(1, 3, &bias).unwrap();

        let other = backend.open_device(ordinal).unwrap();
        let foreign = other.upload_f16(2, 3, &activation).unwrap();
        assert!(matches!(
            a.submit(&foreign, None, OutputType::F16),
            Err(OperatorError::Shape(
                "tensor belongs to another device session"
            ))
        ));
        let empty = device.upload_f16(0, 3, &[]).unwrap();
        assert_eq!(
            a.submit(&empty, None, OutputType::F32)
                .unwrap()
                .read()
                .unwrap(),
            LinearOutput::F32(vec![])
        );

        let first = a.submit(&input, Some(&bias), OutputType::F16).unwrap();
        let second = b.submit(&first, None, OutputType::F32).unwrap();
        assert!(matches!(
            a.submit(&second, None, OutputType::F16),
            Err(OperatorError::Shape(_))
        ));
        let first_again = a.submit(&input, Some(&bias), OutputType::F16).unwrap();
        let second_again = b.submit(&first_again, None, OutputType::F32).unwrap();

        // The two outputs outlive their inputs, weights and original session.
        drop((device, a, b, input, bias, first, first_again, empty));
        assert_eq!(second.read().unwrap(), expected);
        assert_eq!(second_again.read().unwrap(), expected);
        assert_eq!(second.read().unwrap(), expected);
        println!("resident W4A16 device {ordinal}: two chained submissions match Rust CPU");
    }
}

#[cfg(feature = "native")]
#[test]
fn native_chains_reference_and_marlin_on_one_queue() {
    use plait_ops::OperatorError;
    use plait_ops::native::NativeW4A16;
    use std::path::Path;

    let Some(path) = std::env::var_os("PLAIT_RESIDENT_MARLIN_TEST_LIBRARY") else {
        return;
    };
    let ordinals = std::env::var("PLAIT_W4A16_TEST_DEVICES").unwrap_or_else(|_| "0".to_owned());
    let expect_unsupported = std::env::var_os("PLAIT_RESIDENT_MARLIN_EXPECT_UNSUPPORTED").is_some();
    let backend = unsafe { NativeW4A16::load(Path::new(&path)) }.unwrap();
    let width = 256;
    let projection_columns = [0, 127, 255];
    let identity_weights = select_weights(width, 0..width);
    let projection_weights = select_weights(width, projection_columns);
    let unit_scale = vec![half(1.0); width as usize];
    let projection =
        W4Weights::new(width, 3, width, &projection_weights, &unit_scale[..3], None).unwrap();
    let identity =
        W4Weights::new(width, width, width, &identity_weights, &unit_scale, None).unwrap();
    let packed: Vec<_> = (0..width * width / 2)
        .map(|index| ((index * 37 + index / 29 + 5) % 256) as u8)
        .collect();

    for ordinal in ordinals.split(',').map(|text| text.parse::<i32>().unwrap()) {
        let session = backend.open_device(ordinal).unwrap();
        if expect_unsupported {
            assert!(matches!(
                session.prepare_marlin(identity),
                Err(OperatorError::Unsupported(_))
            ));
            println!(
                "resident Marlin device {ordinal}: rejected unsupported backend or architecture"
            );
            continue;
        }
        let before = session.prepare(identity).unwrap();
        let after = session.prepare(projection).unwrap();
        let bias = session.upload_f16(1, width, &unit_scale).unwrap();
        for group_size in [128, width] {
            let scales: Vec<_> = (0..width * width / group_size)
                .map(|index| half((1 + (index * 13 % 7)) as f32 / 128.0))
                .collect();
            let weights = W4Weights::new(width, width, group_size, &packed, &scales, None).unwrap();
            let linear = session.prepare_marlin(weights).unwrap();
            let foreign_session = backend.open_device(ordinal).unwrap();
            let empty = session.upload_f16(0, width, &[]).unwrap();
            let foreign = foreign_session.upload_f16(0, width, &[]).unwrap();
            assert!(matches!(
                linear.submit(&foreign, None, OutputType::F16),
                Err(OperatorError::Shape(_))
            ));
            assert!(matches!(
                linear.submit(&empty, None, OutputType::F32),
                Err(OperatorError::Unsupported(_))
            ));
            assert!(matches!(
                linear.submit(&empty, Some(&bias), OutputType::F16),
                Err(OperatorError::Unsupported(_))
            ));
            assert_eq!(
                linear
                    .submit(&empty, None, OutputType::F16)
                    .unwrap()
                    .read()
                    .unwrap(),
                LinearOutput::F16(vec![])
            );

            for rows in [1, 65] {
                let activation: Vec<_> = (0..rows * width)
                    .map(|index| half(((index * 47 % 63) as i32 - 31) as f32 / 32.0))
                    .collect();
                let LinearOutput::F16(expected) =
                    W4A16::new(weights, &activation, None, OutputType::F16)
                        .unwrap()
                        .run_cpu()
                else {
                    panic!("expected FP16 reference");
                };
                let input = session.upload_f16(rows, width, &activation).unwrap();
                let upstream = before.submit(&input, None, OutputType::F16).unwrap();
                let first = linear.submit(&upstream, None, OutputType::F16).unwrap();
                let first_final = after.submit(&first, None, OutputType::F32).unwrap();
                let second = linear.submit(&upstream, None, OutputType::F16).unwrap();
                let second_final = after.submit(&second, None, OutputType::F32).unwrap();
                drop((input, upstream));

                let LinearOutput::F16(actual) = first.read().unwrap() else {
                    panic!("expected FP16 Marlin result");
                };
                let mut max_abs_error = 0.0f32;
                for (&actual, expected) in actual.iter().zip(expected) {
                    let actual = f16::from_bits(actual).to_f32();
                    let expected = f16::from_bits(expected).to_f32();
                    let error = (actual - expected).abs();
                    assert!(
                        error <= 0.015 + 0.004 * expected.abs(),
                        "{actual} vs {expected}"
                    );
                    max_abs_error = max_abs_error.max(error);
                }
                let projected = actual
                    .chunks_exact(width as usize)
                    .flat_map(|row| {
                        projection_columns
                            .map(|column| f16::from_bits(row[column as usize]).to_f32())
                    })
                    .collect();
                let expected_final = LinearOutput::F32(projected);
                drop((first, second));
                assert_eq!(first_final.read().unwrap(), expected_final);
                assert_eq!(second_final.read().unwrap(), expected_final);
                println!(
                    "resident Marlin device {ordinal}: M={rows} N={width} K={width} G={group_size} max_abs_error={max_abs_error}"
                );
            }
        }
    }
}

#[cfg(feature = "native")]
fn select_weights(width: u32, columns: impl IntoIterator<Item = u32>) -> Vec<u8> {
    let mut weights = Vec::new();
    for column in columns {
        let start = weights.len();
        weights.resize(start + width as usize / 2, 0x88);
        weights[start + column as usize / 2] |= 1 << (4 * (column % 2));
    }
    weights
}

#[cfg(feature = "native")]
#[test]
fn native_humming_runs_groupwise_w4a16_and_chains_on_device() {
    use plait_ops::OperatorError;
    use plait_ops::native::NativeW4A16;
    use std::path::Path;

    let Some(path) = std::env::var_os("PLAIT_RESIDENT_HUMMING_TEST_LIBRARY") else {
        return;
    };
    let dimensions =
        std::env::var("PLAIT_HUMMING_TEST_NK").unwrap_or_else(|_| "256,512".to_owned());
    let (n, k) = dimensions.split_once(',').unwrap();
    let (n, k) = (n.parse::<u32>().unwrap(), k.parse::<u32>().unwrap());
    let group_size = 128;
    let ordinals = std::env::var("PLAIT_W4A16_TEST_DEVICES").unwrap_or_else(|_| "0".to_owned());
    let expect_unsupported =
        std::env::var_os("PLAIT_RESIDENT_HUMMING_EXPECT_UNSUPPORTED").is_some();
    let backend = unsafe { NativeW4A16::load(Path::new(&path)) }.unwrap();
    let packed: Vec<_> = (0..n * k / 2)
        .map(|i| ((i * 37 + i / 29 + i / 113 + 5) % 256) as u8)
        .collect();
    let scales: Vec<_> = (0..n * k / group_size)
        .map(|i| half((1 + i * 13 % 17) as f32 / 256.0))
        .collect();
    let zeros: Vec<_> = (0..n * k / group_size)
        .map(|i| ((i * 11 + i / 19) % 16) as u8)
        .collect();
    let projection_columns = [0, n / 2 - 1, n - 1];
    let projection_weights = select_weights(n, projection_columns);
    let unit_scales = [half(1.0); 3];
    let projection = W4Weights::new(n, 3, n, &projection_weights, &unit_scales, None).unwrap();
    let sample = W4Weights::new(k, n, group_size, &packed, &scales, Some(&zeros)).unwrap();

    for ordinal in ordinals.split(',').map(|text| text.parse::<i32>().unwrap()) {
        let session = backend.open_device(ordinal).unwrap();
        if expect_unsupported {
            assert!(matches!(
                session.prepare_humming(sample),
                Err(OperatorError::Unsupported(_))
            ));
            println!(
                "resident Humming device {ordinal}: rejected unsupported backend or architecture"
            );
            continue;
        }
        assert!(matches!(
            session.prepare_humming(projection),
            Err(OperatorError::Unsupported(_))
        ));
        let next = session.prepare(projection).unwrap();
        for zero in [None, Some(zeros.as_slice())] {
            let weights = W4Weights::new(k, n, group_size, &packed, &scales, zero).unwrap();
            let linear = session.prepare_humming(weights).unwrap();
            let empty = session.upload_f16(0, k, &[]).unwrap();
            assert!(matches!(
                linear.submit(&empty, None, OutputType::F32),
                Err(OperatorError::Unsupported(_))
            ));
            assert_eq!(
                linear
                    .submit(&empty, None, OutputType::F16)
                    .unwrap()
                    .read()
                    .unwrap(),
                LinearOutput::F16(vec![])
            );

            for rows in [1, 33, 65] {
                let activation: Vec<_> = (0..rows * k)
                    .map(|i| half(((i * 47 % 63) as i32 - 31) as f32 / 64.0))
                    .collect();
                let LinearOutput::F16(expected) =
                    W4A16::new(weights, &activation, None, OutputType::F16)
                        .unwrap()
                        .run_cpu()
                else {
                    panic!("expected FP16 CPU reference");
                };
                let input = session.upload_f16(rows, k, &activation).unwrap();
                let result = linear.submit(&input, None, OutputType::F16).unwrap();
                let downstream = next.submit(&result, None, OutputType::F32).unwrap();
                let LinearOutput::F16(actual) = result.read().unwrap() else {
                    panic!("expected FP16 Humming result");
                };
                assert_eq!(actual.len(), expected.len());
                let mut maximum = 0.0f32;
                for (&actual, expected) in actual.iter().zip(expected) {
                    let actual = f16::from_bits(actual).to_f32();
                    let expected = f16::from_bits(expected).to_f32();
                    let error = (actual - expected).abs();
                    assert!(
                        error <= 0.015 + 0.004 * expected.abs(),
                        "{actual} vs {expected}"
                    );
                    maximum = maximum.max(error);
                }
                let projected = actual
                    .chunks_exact(n as usize)
                    .flat_map(|row| {
                        projection_columns
                            .map(|column| f16::from_bits(row[column as usize]).to_f32())
                    })
                    .collect();
                drop((input, result));
                assert_eq!(downstream.read().unwrap(), LinearOutput::F32(projected));
                println!(
                    "resident Humming device {ordinal}: M={rows} N={n} K={k} G={group_size} explicit_zero={} max_abs_error={maximum}",
                    zero.is_some()
                );
            }
        }
    }
}
