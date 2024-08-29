/*
wget https://github.com/thewh1teagle/vibe/raw/main/samples/multi.wav
cargo run --example normalize multi.wav normalized.wav
*/
use ebur128::{EbuR128, Mode};
use hound::{WavReader, WavSpec, WavWriter};

fn main() {
    let input_path = std::env::args()
        .nth(1)
        .expect("Please specify input wav path");
    let output_path = std::env::args()
        .nth(2)
        .expect("Please specify output wav path");
    let target_loudness = -23.0; // EBU R128 standard target loudness

    let mut reader = WavReader::open(&input_path).expect("Failed to open WAV file");

    let spec = reader.spec();
    let channels = spec.channels as usize;
    let rate = spec.sample_rate;
    let mut ebur128 =
        EbuR128::new(channels as u32, rate, Mode::all()).expect("Failed to create ebur128");

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();
    let chunk_size = rate; // 1s

    // Compute loudness
    for chunk in samples.chunks(chunk_size as usize * channels) {
        ebur128.add_frames_f32(chunk).expect("Failed to add frames");
        ebur128
            .loudness_global()
            .expect("Failed to get global loudness");
    }

    let global_loudness = ebur128
        .loudness_global()
        .expect("Failed to get global loudness");

    // Convert dB difference to linear gain
    let gain = 10f32.powf(((target_loudness - global_loudness) / 20.0) as f32);

    let mut writer = WavWriter::create(
        output_path,
        WavSpec {
            channels: spec.channels,
            sample_rate: spec.sample_rate,
            bits_per_sample: spec.bits_per_sample,
            sample_format: spec.sample_format,
        },
    )
    .expect("Failed to create WAV writer");

    for sample in samples {
        let normalized_sample = (sample * gain).clamp(-1.0, 1.0);
        writer
            .write_sample((normalized_sample * i16::MAX as f32) as i16)
            .expect("Failed to write sample");
    }
}
