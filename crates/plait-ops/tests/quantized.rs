use half::f16;
use plait_ops::{ActivationBits, LinearOutput, OutputType, QuantizedActivations, W4Weights};

fn half(value: f32) -> u16 {
    f16::from_f32(value).to_bits()
}

#[test]
fn quantization_packs_signed_codes_and_preserves_the_tail() {
    let values = [half(0.0), half(-7.0), half(7.0)];
    let four = QuantizedActivations::from_f16(ActivationBits::I4, 3, 2, &values).unwrap();
    assert_eq!(four.packed(), &[0x90, 0x07]);
    let eight = QuantizedActivations::from_f16(ActivationBits::I8, 3, 2, &values).unwrap();
    assert_eq!(eight.packed(), &[0, (-127i8) as u8, 127]);
    assert!(QuantizedActivations::from_f16(ActivationBits::I4, 3, 2, &values[..2]).is_err());
    assert!(
        QuantizedActivations::from_f16(ActivationBits::I8, 1, 1, &[half(f32::INFINITY)]).is_err()
    );

    let scales = [half(1.0), half(0.5)];
    let weights = W4Weights::new(3, 1, 2, &[0xf1, 3], &scales, Some(&[6, 1])).unwrap();
    assert_eq!(
        four.run_cpu(weights, None, OutputType::F32).unwrap(),
        LinearOutput::F32(vec![-56.0])
    );
    assert_eq!(
        eight.run_cpu(weights, None, OutputType::F32).unwrap(),
        LinearOutput::F32(vec![-56.0])
    );
}

#[cfg(feature = "native")]
#[test]
fn native_integer_activation_matches_groupwise_cpu() {
    use plait_ops::native::NativeW4A16;
    use std::path::Path;

    let Some(library) = std::env::var_os("PLAIT_QUANTIZED_TEST_LIBRARY") else {
        return;
    };
    let backend = unsafe { NativeW4A16::load(Path::new(&library)) }.unwrap();
    let indices = std::env::var("PLAIT_W4A16_TEST_DEVICES").unwrap_or_else(|_| "0".to_owned());
    for ordinal in indices
        .split(',')
        .map(|index| index.parse::<i32>().unwrap())
    {
        let device = backend.open_device(ordinal).unwrap();
        let shapes = [
            (3, 7, 19, 8),
            (2, 5, 37, 9),
            (3, 35, 260, 128),
            (17, 11, 145, 65),
            (33, 67, 181, 65),
            (2, 5, 519, 257),
            // Word-aligned groups that still require padding to a Cube tile.
            (2, 37, 264, 136),
            // Aligned packed loads with a partial quantization group and SIMD group.
            (3, 11, 544, 128),
        ];
        for (m, n, k, group_size) in shapes {
            let fixture = Fixture::new(m, n, k, group_size);
            let bias = device.upload_f16(1, n, &fixture.bias).unwrap();
            let full = device
                .supports_f16_quantization()
                .then(|| device.upload_f16(m, k, &fixture.activation).unwrap());
            for has_zero in [false, true] {
                let weights = fixture.weights(has_zero);
                let linear = device.prepare(weights).unwrap();
                for bits in [ActivationBits::I4, ActivationBits::I8] {
                    let input = fixture.quantize(bits);
                    let activation = match &full {
                        Some(full) => device.quantize_f16(full, bits, group_size).unwrap(),
                        None => device.upload_quantized(&input).unwrap(),
                    };
                    let pending = [OutputType::F32, OutputType::F16].map(|dtype| {
                        let output = linear
                            .submit_quantized(&activation, Some(&bias), dtype)
                            .unwrap();
                        (dtype, output)
                    });
                    for (dtype, output) in pending {
                        let expected = input.run_cpu(weights, Some(&fixture.bias), dtype).unwrap();
                        let actual = output.read().unwrap();
                        match (actual, expected) {
                            (LinearOutput::F32(actual), LinearOutput::F32(expected)) => {
                                assert_eq!(actual.len(), expected.len());
                                for (a, b) in actual.into_iter().zip(expected) {
                                    assert!(
                                        (a - b).abs() <= 0.000002 * (1.0 + b.abs()),
                                        "{a} vs {b}"
                                    );
                                }
                            }
                            (LinearOutput::F16(actual), LinearOutput::F16(expected)) => {
                                assert_eq!(actual, expected)
                            }
                            _ => panic!("unexpected output type"),
                        }
                    }
                }
            }
        }
        println!(
            "integer W4 device {ordinal}: {} CPU comparisons passed, device_quantization={}",
            shapes.len() * 8,
            device.supports_f16_quantization(),
        );
        if device.supports_f16_quantization() {
            check_device_quantization_chain(&device);
        }
        if std::env::var_os("PLAIT_QUANTIZED_BENCH").is_some() {
            compare_resident_execution(&device, ordinal);
        }
    }
}

#[cfg(feature = "native")]
fn check_device_quantization_chain(device: &plait_ops::native::DeviceSession<'_>) {
    let mut packed = [0; 8 * 4];
    for column in 0..8 {
        packed[column * 4 + column / 2] = 1 << (4 * (column % 2));
    }
    let scales = [half(1.0); 8];
    let zero = [0; 8];
    let weights = W4Weights::new(8, 8, 8, &packed, &scales, Some(&zero)).unwrap();
    let identity = device.prepare(weights).unwrap();
    for (bits, maximum) in [(ActivationBits::I4, 7.0), (ActivationBits::I8, 127.0)] {
        let values = [
            half(maximum),
            half(1.5),
            half(2.5),
            half(-1.5),
            half(-2.5),
            half(-maximum),
            1,
            half(-0.0),
        ];
        let input = device.upload_f16(1, 8, &values).unwrap();
        let computed = identity.submit(&input, None, OutputType::F16).unwrap();
        let quantized = device.quantize_f16(&computed, bits, 8).unwrap();
        let actual = identity
            .submit_quantized(&quantized, None, OutputType::F32)
            .unwrap()
            .read()
            .unwrap();
        let expected = QuantizedActivations::from_f16(bits, 8, 8, &values)
            .unwrap()
            .run_cpu(weights, None, OutputType::F32)
            .unwrap();
        assert_eq!(actual, expected);
    }
    for nonfinite in [f16::NEG_INFINITY, f16::NAN] {
        let input = device.upload_f16(1, 8, &[nonfinite.to_bits(); 8]).unwrap();
        let computed = identity.submit(&input, None, OutputType::F16).unwrap();
        let quantized = device
            .quantize_f16(&computed, ActivationBits::I4, 8)
            .unwrap();
        if device.supports_completion_query() {
            let _ = quantized.is_ready().unwrap();
        }
        let output = identity
            .submit_quantized(&quantized, None, OutputType::F32)
            .unwrap();
        drop(quantized);
        drop(computed);
        assert!(matches!(
            output.read(),
            Err(plait_ops::OperatorError::Shape(
                "quantized activations must be finite"
            ))
        ));
        if device.supports_completion_query() {
            assert!(output.is_ready().unwrap());
        }
    }
    println!("device quantization chain: round-to-even, subnormal and non-finite inputs verified");
}

#[cfg(feature = "native")]
fn compare_resident_execution(device: &plait_ops::native::DeviceSession<'_>, ordinal: i32) {
    use std::time::Instant;

    fn median(mut run: impl FnMut()) -> u128 {
        for _ in 0..4 {
            run();
        }
        let mut times: Vec<_> = (0..17)
            .map(|_| {
                let start = Instant::now();
                run();
                start.elapsed().as_nanos()
            })
            .collect();
        times.sort_unstable();
        times[times.len() / 2]
    }

    for (m, n, k) in [(1, 1024, 4096), (16, 512, 4096)] {
        let fixture = Fixture::new(m, n, k, 128);
        let linear = device.prepare(fixture.weights(true)).unwrap();
        let full = device.upload_f16(m, k, &fixture.activation).unwrap();
        let submit_full = || linear.submit(&full, None, OutputType::F32).unwrap();
        let elapsed_full = median(|| submit_full().wait().unwrap());
        let LinearOutput::F32(baseline) = submit_full().read().unwrap() else {
            panic!("expected FP32 reference");
        };
        println!(
            "integer W4 benchmark device {ordinal}: M={m} N={n} K={k} A=16 median_ns={elapsed_full}"
        );
        for bits in [ActivationBits::I4, ActivationBits::I8] {
            let start = Instant::now();
            let input = fixture.quantize(bits);
            let quantize_ns = start.elapsed().as_nanos();
            let input = device.upload_quantized(&input).unwrap();
            let submit = || {
                linear
                    .submit_quantized(&input, None, OutputType::F32)
                    .unwrap()
            };
            let elapsed = median(|| submit().wait().unwrap());
            let LinearOutput::F32(values) = submit().read().unwrap() else {
                panic!("expected FP32 output");
            };
            let squared_error: f64 = values
                .iter()
                .zip(&baseline)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum();
            let squared_signal: f64 = baseline.iter().map(|x| f64::from(*x).powi(2)).sum();
            let relative_l2 = (squared_error / squared_signal).sqrt();
            println!(
                "integer W4 benchmark device {ordinal}: M={m} N={n} K={k} A={} median_ns={elapsed} host_quantize_ns={quantize_ns} relative_l2={relative_l2}",
                bits as u32
            );
            if device.supports_f16_quantization() {
                let finish = |input: &plait_ops::native::DeviceQuantizedActivations<'_>| {
                    linear
                        .submit_quantized(input, None, OutputType::F32)
                        .unwrap()
                        .read()
                        .unwrap()
                };
                let host = || {
                    let input = device.upload_quantized(&fixture.quantize(bits)).unwrap();
                    finish(&input)
                };
                let resident = |input: &plait_ops::native::DeviceTensor<'_>| {
                    let input = device.quantize_f16(input, bits, 128).unwrap();
                    finish(&input)
                };
                let device_upload = || {
                    let input = device.upload_f16(m, k, &fixture.activation).unwrap();
                    resident(&input)
                };
                assert_eq!(host(), device_upload());
                let host_ns = median(|| {
                    host();
                });
                let device_ns = median(|| {
                    device_upload();
                });
                let resident_ns = median(|| {
                    resident(&full);
                });
                println!(
                    "device quantization benchmark device {ordinal}: M={m} N={n} K={k} A={} host_path_ns={host_ns} device_path_ns={device_ns} resident_path_ns={resident_ns}",
                    bits as u32,
                );
            }
        }
    }
}

#[cfg(feature = "native")]
struct Fixture {
    n: u32,
    k: u32,
    group_size: u32,
    packed: Vec<u8>,
    scale: Vec<u16>,
    zero: Vec<u8>,
    activation: Vec<u16>,
    bias: Vec<u16>,
}

#[cfg(feature = "native")]
impl Fixture {
    fn new(m: u32, n: u32, k: u32, group_size: u32) -> Self {
        let groups = n * k.div_ceil(group_size);
        Self {
            n,
            k,
            group_size,
            packed: (0..n * k.div_ceil(2))
                .map(|i| (i * 29 + i / 13 + 7) as u8)
                .collect(),
            scale: (0..groups)
                .map(|i| half(((i * 13 % 19) + 1) as f32 / 128.0))
                .collect(),
            zero: (0..groups).map(|i| ((i * 7 + i / 13) % 16) as u8).collect(),
            activation: (0..m * k)
                .map(|i| half((i32::try_from(i * 47 % 137).unwrap() - 68) as f32 / 68.0))
                .collect(),
            bias: (0..n).map(|i| half(i as f32 / 16.0)).collect(),
        }
    }

    fn weights(&self, has_zero: bool) -> W4Weights<'_> {
        W4Weights::new(
            self.k,
            self.n,
            self.group_size,
            &self.packed,
            &self.scale,
            has_zero.then_some(self.zero.as_slice()),
        )
        .unwrap()
    }

    fn quantize(&self, bits: ActivationBits) -> QuantizedActivations {
        QuantizedActivations::from_f16(bits, self.k, self.group_size, &self.activation).unwrap()
    }
}
