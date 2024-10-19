use ebur128::{energy_to_loudness, EbuR128, Mode};
use hound::WavReader;
use std::path::Path;

/// ReplayGain 2.0 Reference Gain
///
/// See the [ReplayGain 2.0 specification][rg2spec] for details.
///
/// [rg2spec]: https://wiki.hydrogenaud.io/index.php?title=ReplayGain_2.0_specification#Reference_level
const REPLAYGAIN2_REFERENCE_LUFS: f64 = -18.0;

struct TrackData {
    pub loudness: f64,
    pub peak: f64,
    pub gating_block_count: u64,
    pub energy: f64,
}

fn main() {
    let input_path = std::env::args()
        .nth(1)
        .expect("Please specify input wav directory path");

    let mut album_peak: f64 = 0.0;
    let mut album_gating_block_count: u64 = 0;
    let mut album_energy: f64 = 0.0;

    for dir_entry in std::fs::read_dir(&input_path).expect("Failed to read directory path") {
        let dir_entry = dir_entry.expect("Failed to read dir entry");
        let metadata = dir_entry.metadata().expect("Failed to read metadata");
        if !metadata.is_file() {
            continue;
        }

        let track_data = analyze_file(&dir_entry.path());
        let track_gain = REPLAYGAIN2_REFERENCE_LUFS - track_data.loudness;
        let track_peak = track_data.peak;

        println!("TRACK_PATH={}", dir_entry.path().display());

        // ReplayGain 2.0 Track Gain, formatted according to "Table 3: Metadata keys and value
        // formatting" in the ["Metadata format" section in the ReplayGain 2.0 specification][rgmeta].
        //
        // [rgmeta]: https://wiki.hydrogenaud.io/index.php?title=ReplayGain_2.0_specification#Metadata_format
        println!("REPLAYGAIN_TRACK_GAIN={track_gain:.2} dB");

        // ReplayGain 2.0 Track Peak, formatted according to "Table 3: Metadata keys and value
        // formatting" in the ["Metadata format" section in the ReplayGain 2.0 specification][rgmeta].
        //
        // [rgmeta]: https://wiki.hydrogenaud.io/index.php?title=ReplayGain_2.0_specification#Metadata_format
        println!("REPLAYGAIN_TRACK_PEAK={track_peak:.6}");
        println!();

        // Album peak is just the maximum peak on the album.
        album_peak = album_peak.max(track_peak);
        album_gating_block_count += track_data.gating_block_count;
        album_energy += track_data.energy;
    }

    let album_gain = REPLAYGAIN2_REFERENCE_LUFS
        - energy_to_loudness(album_energy / album_gating_block_count as f64);

    println!("REPLAYGAIN_ALBUM_GAIN={album_gain:.2} dB");
    println!("REPLAYGAIN_ALBUM_PEAK={album_peak:.6}");
    println!("REPLAYGAIN_REFERNCE_LOUDNESS={REPLAYGAIN2_REFERENCE_LUFS:.2} LUFS");
}

fn analyze_file(input_path: &Path) -> TrackData {
    let mut reader = WavReader::open(input_path).expect("Failed to open WAV file");

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

    let loudness = ebur128
        .loudness_global()
        .expect("Failed to get global loudness");
    let peak = (0..channels)
        .map(|channel_index| ebur128.sample_peak(channel_index as u32))
        .try_fold(0.0f64, |a, b| b.map(|b| a.max(b)))
        .expect("Failed to determine peak");
    let (gating_block_count, energy) = ebur128
        .gating_block_count_and_energy()
        .expect("failed to get gating block count and loudness");

    TrackData {
        loudness,
        peak,
        gating_block_count,
        energy,
    }
}
