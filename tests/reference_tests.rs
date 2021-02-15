// Tests can be downloaded from https://tech.ebu.ch/publications/ebu_loudness_test_set
// and should be put into tests/reference_files
//
// Expected results are in
//  - https://tech.ebu.ch/docs/tech/tech3341.pdf
//  - https://tech.ebu.ch/docs/tech/tech3342.pdf

macro_rules! read_samples {
    ($reader:expr, 16) => {
        $reader
            .into_samples::<i16>()
            .map(|s| s.expect("Failed to read samples from reference file"))
            .collect::<Vec<_>>()
    };
    ($reader:expr, 24) => {
        $reader
            .into_samples::<i32>()
            .map(|s| s.expect("Failed to read samples from reference file") << 8)
            .collect::<Vec<_>>()
    };
}

macro_rules! prepare_file {
    ($file_name:expr, $rate:expr, $bpp:tt, $mode:expr, $channel_map:expr) => {{
        let input_path = {
            let mut r = std::path::PathBuf::new();
            r.push(env!("CARGO_MANIFEST_DIR"));
            r.push("tests");
            r.push("reference_files");
            r.push($file_name);
            r
        };

        if !input_path.exists() {
            panic!("Reference file {} not found", input_path.display());
        }

        let mut e = ebur128::EbuR128::new($channel_map.len() as u32, $rate, $mode)
            .expect("Can't create EbuR128 instance");
        e.set_channel_map(&$channel_map).unwrap();

        let reader = hound::WavReader::open(&input_path).expect("Failed to read reference file");
        let samples = read_samples!(reader, $bpp);

        (e, samples)
    }};
}

macro_rules! add_samples {
    ($e:expr, $samples:expr, 16) => {
        $e.add_frames_i16(&*$samples).expect("Failed to analyze samples");
    };
    ($e:expr, $samples:expr, 24) => {
        $e.add_frames_i32(&*$samples).expect("Failed to analyze samples");
    };
}

macro_rules! test_global_loudness(
    ($file_name:expr, $rate:expr, $bpp:tt, $channel_map:expr, $expected_loudness:expr) => {
        {
            let (mut e, samples) = prepare_file!($file_name, $rate, $bpp, ebur128::Mode::I, $channel_map);

            add_samples!(e, samples, $bpp);

            float_eq::assert_float_eq!(e.loudness_global().expect("Failed to get global loudness"), $expected_loudness, abs <= 0.1);
        }

        {
            let (mut e, samples) = prepare_file!($file_name, $rate, $bpp, ebur128::Mode::I | ebur128::Mode::HISTOGRAM, $channel_map);

            add_samples!(e, samples, $bpp);

            float_eq::assert_float_eq!(e.loudness_global().expect("Failed to get global loudness"), $expected_loudness, abs <= 0.1);
        }
    };
);

macro_rules! test_true_peak(
    ($file_name:expr, $rate:expr, $bpp:tt, $channel_map:expr, $expected_true_peak:expr) => {
        let (mut e, samples) = prepare_file!($file_name, $rate, $bpp, ebur128::Mode::TRUE_PEAK, $channel_map);

        add_samples!(e, samples, $bpp);

        let mut max_true_peak = f64::MIN;
        for peak in (0..$channel_map.len())
            .map(|c| e.true_peak(c as u32).expect("Failed to get true peak")) {
            if peak > max_true_peak {
                max_true_peak = peak;
            }
        }
        let max_true_peak = 20.0 * f64::log10(max_true_peak);

        assert!(
            ($expected_true_peak - 0.4..=$expected_true_peak + 0.2).contains(&max_true_peak),
            "{} != {}",
            max_true_peak,
            $expected_true_peak,
        );
    };
);

macro_rules! test_loudness_range(
    ($file_name:expr, $rate:expr, $bpp:tt, $channel_map:expr, $expected_loudness_range:expr) => {
        {
            let (mut e, samples) = prepare_file!($file_name, $rate, $bpp, ebur128::Mode::LRA, $channel_map);

            add_samples!(e, samples, $bpp);

            let loudness_range = e.loudness_range().expect("Failed to get loudness range");
            float_eq::assert_float_eq!(loudness_range, $expected_loudness_range, abs <= 1.0, "queue mode");
        }

        {
            let (mut e, samples) = prepare_file!($file_name, $rate, $bpp, ebur128::Mode::LRA | ebur128::Mode::HISTOGRAM, $channel_map);

            add_samples!(e, samples, $bpp);

            let loudness_range = e.loudness_range().expect("Failed to get loudness range in histogram mode");
            float_eq::assert_float_eq!(loudness_range, $expected_loudness_range, abs <= 1.0, "histogram mode");
        }
    };
);

#[test]
fn seq_3341_1() {
    {
        test_global_loudness!(
            "seq-3341-1-16bit.wav",
            48_000,
            16,
            [ebur128::Channel::Left, ebur128::Channel::Right],
            -23.0
        );
    }

    {
        let (mut e, samples) = prepare_file!(
            "seq-3341-1-16bit.wav",
            48_000,
            16,
            ebur128::Mode::S,
            [ebur128::Channel::Left, ebur128::Channel::Right]
        );

        // 100ms chunks / 10Hz
        for (i, chunk) in samples.chunks(2 * 48_000 / 10).enumerate() {
            e.add_frames_i16(chunk).expect("Failed to analyze samples");

            // Constant after 3s
            if i >= 29 {
                float_eq::assert_float_eq!(
                    e.loudness_shortterm()
                        .expect("Failed to get shortterm loudness"),
                    -23.0,
                    abs <= 0.1
                );
            }
        }
    }

    {
        let (mut e, samples) = prepare_file!(
            "seq-3341-1-16bit.wav",
            48_000,
            16,
            ebur128::Mode::M,
            [ebur128::Channel::Left, ebur128::Channel::Right]
        );

        // 100ms chunks / 10Hz
        for (i, chunk) in samples.chunks(2 * 48_000 / 10).enumerate() {
            e.add_frames_i16(chunk).expect("Failed to analyze samples");

            // Constant after 1s
            if i >= 10 {
                float_eq::assert_float_eq!(
                    e.loudness_momentary()
                        .expect("Failed to get momentary loudness"),
                    -23.0,
                    abs <= 0.1
                );
            }
        }
    }
}

#[test]
fn seq_3341_2() {
    test_global_loudness!(
        "seq-3341-2-16bit.wav",
        48_000,
        16,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        -33.0
    );

    {
        let (mut e, samples) = prepare_file!(
            "seq-3341-2-16bit.wav",
            48_000,
            16,
            ebur128::Mode::S,
            [ebur128::Channel::Left, ebur128::Channel::Right]
        );

        // 100ms chunks / 10Hz
        for (i, chunk) in samples.chunks(2 * 48_000 / 10).enumerate() {
            e.add_frames_i16(chunk).expect("Failed to analyze samples");

            // Constant after 3s
            if i >= 29 {
                float_eq::assert_float_eq!(
                    e.loudness_shortterm()
                        .expect("Failed to get shortterm loudness"),
                    -33.0,
                    abs <= 0.1
                );
            }
        }
    }

    {
        let (mut e, samples) = prepare_file!(
            "seq-3341-2-16bit.wav",
            48_000,
            16,
            ebur128::Mode::M,
            [ebur128::Channel::Left, ebur128::Channel::Right]
        );

        // 100ms chunks / 10Hz
        for (i, chunk) in samples.chunks(2 * 48_000 / 10).enumerate() {
            e.add_frames_i16(chunk).expect("Failed to analyze samples");

            // Constant after 1s
            if i >= 10 {
                float_eq::assert_float_eq!(
                    e.loudness_momentary()
                        .expect("Failed to get momentary loudness"),
                    -33.0,
                    abs <= 0.1
                );
            }
        }
    }
}

#[test]
fn seq_3341_3() {
    test_global_loudness!(
        "seq-3341-3-16bit-v02.wav",
        48_000,
        16,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        -23.0
    );
}

#[test]
fn seq_3341_4() {
    test_global_loudness!(
        "seq-3341-4-16bit-v02.wav",
        48_000,
        16,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        -23.0
    );
}

#[test]
fn seq_3341_5() {
    test_global_loudness!(
        "seq-3341-5-16bit-v02.wav",
        48_000,
        16,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        -23.0
    );
}

#[test]
fn seq_3341_6() {
    test_global_loudness!(
        "seq-3341-6-5channels-16bit.wav",
        48_000,
        16,
        [
            ebur128::Channel::Left,
            ebur128::Channel::Right,
            ebur128::Channel::Center,
            ebur128::Channel::LeftSurround,
            ebur128::Channel::RightSurround
        ],
        -23.0
    );
}

#[test]
fn seq_3341_6_1() {
    test_global_loudness!(
        "seq-3341-6-6channels-WAVEEX-16bit.wav",
        48_000,
        16,
        [
            ebur128::Channel::Left,
            ebur128::Channel::Right,
            ebur128::Channel::Center,
            ebur128::Channel::Unused,
            ebur128::Channel::LeftSurround,
            ebur128::Channel::RightSurround
        ],
        -23.0
    );
}

#[test]
fn seq_3341_7() {
    test_global_loudness!(
        "seq-3341-7_seq-3342-5-24bit.wav",
        48_000,
        24,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        -23.0
    );
}

#[test]
fn seq_3341_8() {
    test_global_loudness!(
        "seq-3341-2011-8_seq-3342-6-24bit-v02.wav",
        48_000,
        24,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        -23.0
    );
}

#[test]
fn seq_3341_9() {
    let (mut e, samples) = prepare_file!(
        "seq-3341-9-24bit.wav",
        48_000,
        24,
        ebur128::Mode::S,
        [ebur128::Channel::Left, ebur128::Channel::Right]
    );

    // 100ms chunks / 10Hz
    for (i, chunk) in samples.chunks(2 * 48_000 / 10).enumerate() {
        e.add_frames_i32(chunk).expect("Failed to analyze samples");

        // Constant after 3s
        if i >= 29 {
            float_eq::assert_float_eq!(
                e.loudness_shortterm()
                    .expect("Failed to get shortterm loudness"),
                -23.0,
                abs <= 0.1
            );
        }
    }
}

#[test]
fn seq_3341_10() {
    for i in 1..=20 {
        let (mut e, samples) = prepare_file!(
            format!("seq-3341-10-{}-24bit.wav", i),
            48_000,
            24,
            ebur128::Mode::S,
            [ebur128::Channel::Left, ebur128::Channel::Right]
        );

        // 100ms chunks / 10Hz
        let mut max_loudness = f64::MIN;
        for chunk in samples.chunks(2 * 48_000 / 10) {
            e.add_frames_i32(chunk).expect("Failed to analyze samples");

            let loudness = e
                .loudness_shortterm()
                .expect("Failed to get shortterm loudness");
            if loudness > max_loudness {
                max_loudness = loudness;
            }
        }

        float_eq::assert_float_eq!(max_loudness, -23.0, abs <= 0.1, "file {}", i);
    }
}

#[test]
fn seq_3341_11() {
    let (mut e, samples) = prepare_file!(
        "seq-3341-11-24bit.wav",
        48_000,
        24,
        ebur128::Mode::S,
        [ebur128::Channel::Left, ebur128::Channel::Right]
    );

    // 100ms chunks / 10Hz
    let mut max_loudness = f64::MIN;
    for (i, chunk) in samples.chunks(2 * 48_000 / 10).enumerate() {
        e.add_frames_i32(chunk).expect("Failed to analyze samples");

        let loudness = e
            .loudness_shortterm()
            .expect("Failed to get shortterm loudness");

        if loudness > max_loudness {
            max_loudness = loudness;
        }
        if (i + 1) % 60 == 0 {
            let expected = -38.0 + ((i + 1) / 60 - 1) as f64;
            float_eq::assert_float_eq!(
                max_loudness,
                expected,
                abs <= 0.1,
                "chunk {}",
                (i + 1) / 60 - 1
            );
        }
    }
}

#[test]
fn seq_3341_12() {
    let (mut e, samples) = prepare_file!(
        "seq-3341-12-24bit.wav",
        48_000,
        24,
        ebur128::Mode::M,
        [ebur128::Channel::Left, ebur128::Channel::Right]
    );

    // 100ms chunks / 10Hz
    for (i, chunk) in samples.chunks(2 * 48_000 / 10).enumerate() {
        e.add_frames_i32(chunk).expect("Failed to analyze samples");

        // Constant after 1s
        if i >= 10 {
            float_eq::assert_float_eq!(
                e.loudness_momentary()
                    .expect("Failed to get momentary loudness"),
                -23.0,
                abs <= 0.1
            );
        }
    }
}

#[test]
fn seq_3341_13() {
    for i in 1..=20 {
        let (mut e, samples) = prepare_file!(
            format!(
                "seq-3341-13-{}-24bit.wav{}",
                i,
                if i > 2 { ".wav" } else { "" }
            ),
            48_000,
            24,
            ebur128::Mode::M,
            [ebur128::Channel::Left, ebur128::Channel::Right]
        );

        // 10ms chunks / 100Hz
        let mut max_loudness = f64::MIN;
        for chunk in samples.chunks(2 * 48_000 / 100) {
            e.add_frames_i32(chunk).expect("Failed to analyze samples");

            let loudness = e
                .loudness_momentary()
                .expect("Failed to get momentary loudness");
            if loudness > max_loudness {
                max_loudness = loudness;
            }
        }

        float_eq::assert_float_eq!(max_loudness, -23.0, abs <= 0.1, "file {}", i);
    }
}

#[test]
fn seq_3341_14() {
    let (mut e, samples) = prepare_file!(
        "seq-3341-14-24bit.wav.wav",
        48_000,
        24,
        ebur128::Mode::M,
        [ebur128::Channel::Left, ebur128::Channel::Right]
    );

    // 10ms chunks / 100Hz
    let mut max_loudness = f64::MIN;
    for (i, chunk) in samples.chunks(2 * 48_000 / 100).enumerate() {
        e.add_frames_i32(chunk).expect("Failed to analyze samples");

        let loudness = e
            .loudness_momentary()
            .expect("Failed to get momentary loudness");

        if loudness > max_loudness {
            max_loudness = loudness;
        }
        if (i + 1) % 80 == 0 {
            let expected = -38.0 + ((i + 1) / 80 - 1) as f64;
            float_eq::assert_float_eq!(
                max_loudness,
                expected,
                abs <= 0.1,
                "chunk {}",
                (i + 1) / 80 - 1
            );
        }
    }
}

#[test]
fn seq_3341_15() {
    test_true_peak!(
        "seq-3341-15-24bit.wav.wav",
        48_000,
        24,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        -6.0
    );
}

#[test]
fn seq_3341_16() {
    test_true_peak!(
        "seq-3341-16-24bit.wav.wav",
        48_000,
        24,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        -6.0
    );
}

#[test]
fn seq_3341_17() {
    test_true_peak!(
        "seq-3341-17-24bit.wav.wav",
        48_000,
        24,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        -6.0
    );
}

#[test]
fn seq_3341_18() {
    test_true_peak!(
        "seq-3341-18-24bit.wav.wav",
        48_000,
        24,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        -6.0
    );
}

#[test]
fn seq_3341_19() {
    test_true_peak!(
        "seq-3341-19-24bit.wav.wav",
        48_000,
        24,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        3.0
    );
}

#[test]
fn seq_3341_20() {
    test_true_peak!(
        "seq-3341-20-24bit.wav.wav",
        48_000,
        24,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        0.0
    );
}

#[test]
fn seq_3341_21() {
    test_true_peak!(
        "seq-3341-21-24bit.wav.wav",
        48_000,
        24,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        0.0
    );
}

#[test]
fn seq_3341_22() {
    test_true_peak!(
        "seq-3341-22-24bit.wav.wav",
        48_000,
        24,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        0.0
    );
}

#[test]
fn seq_3341_23() {
    test_true_peak!(
        "seq-3341-23-24bit.wav.wav",
        48_000,
        24,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        0.0
    );
}

#[test]
fn seq_3342_1() {
    test_loudness_range!(
        "seq-3342-1-16bit.wav",
        48_000,
        16,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        10.0
    );
}

#[test]
fn seq_3342_2() {
    test_loudness_range!(
        "seq-3342-2-16bit.wav",
        48_000,
        16,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        5.0
    );
}

#[test]
fn seq_3342_3() {
    test_loudness_range!(
        "seq-3342-3-16bit.wav",
        48_000,
        16,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        20.0
    );
}

#[test]
fn seq_3342_4() {
    test_loudness_range!(
        "seq-3342-4-16bit.wav",
        48_000,
        16,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        15.0
    );
}

#[test]
fn seq_3342_5() {
    test_loudness_range!(
        "seq-3341-7_seq-3342-5-24bit.wav",
        48_000,
        24,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        5.0
    );
}

#[test]
fn seq_3342_6() {
    test_loudness_range!(
        "seq-3341-2011-8_seq-3342-6-24bit-v02.wav",
        48_000,
        24,
        [ebur128::Channel::Left, ebur128::Channel::Right],
        15.0
    );
}
