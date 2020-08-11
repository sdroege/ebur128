// Copyright (c) 2011 Jan Kokemüller
// Copyright (c) 2020 Sebastian Dröge <sebastian@centricular.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

use crate::energy_to_loudness;

use bitflags::bitflags;

use std::error;
use std::fmt;

/// Error values for [`EbuR128`](struct.EbuR128.html) functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Not enough memory
    NoMem,
    /// Invalid mode selected
    InvalidMode,
    /// Invalid channel index passed
    InvalidChannelIndex,
}

impl error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::NoMem => write!(f, "NoMem"),
            Error::InvalidMode => write!(f, "Invalid Mode"),
            Error::InvalidChannelIndex => write!(f, "Invalid Channel Index"),
        }
    }
}

bitflags! {
    /// Processing mode.
    ///
    /// Use these values in [`EbuR128::new`](struct.EbuR128.html#method.new). Try to use the lowest
    /// possible modes that suit your needs, as performance will be better.
    pub struct Mode: u8 {
        /// can call [`EbuR128::loudness_momentary`](struct.EbuR128.html#method.loudness_momentary)
        const M = 0b00000001;
        /// can call [`EbuR128::loudness_shortterm`](struct.EbuR128.html#method.loudness_shortterm)
        const S = 0b00000010 | Mode::M.bits;
        /// can call [`EbuR128::loudness_global`](struct.EbuR128.html#method.loudness_global) and
        /// [`EbuR128::relative_threshold`](struct.EbuR128.html#method.relative_threshold)
        const I = 0b00000100 | Mode::M.bits;
        /// can call [`EbuR128::loudness_range`](struct.EbuR128.html#method.loudness_range)
        const LRA = 0b00001000 | Mode::S.bits;
        /// can call [`EbuR128::sample_peak`](struct.EbuR128.html#method.sample_peak)
        const SAMPLE_PEAK = 0b00010000 | Mode::M.bits;
        /// can call [`EbuR128::true_peak`](struct.EbuR128.html#method.true_peak)
        const TRUE_PEAK = 0b00110001;
        /// uses histogram algorithm to calculate loudness
        const HISTOGRAM = 0b01000000;
    }
}

/// Channel position.
///
/// Use these values when setting the channel map with
/// [`EbuR128::set_channel`](struct.EbuR128.html#method.set_channel).
/// See definitions in ITU R-REC-BS 1770-4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Channel {
    /// unused channel (for example LFE channel)
    Unused,
    /// Left or itu M+030
    Left,
    /// Right or itu M-030
    Right,
    /// Center or itu M+000
    Center,
    /// Left surround or itu M+110
    LeftSurround,
    /// Right surround or itu M-110
    RightSurround,
    /// a channel that is counted twice
    DualMono,
    /// itu M+SC
    MpSC,
    /// itu M-SC
    MmSC,
    /// itu M+060
    Mp060,
    /// itu M-060
    Mm060,
    /// itu M+090
    Mp090,
    /// itu M-090
    Mm090,
    /// itu M+135
    Mp135,
    /// itu M-135
    Mm135,
    /// itu M+180
    Mp180,
    /// itu U+000
    Up000,
    /// itu U+030
    Up030,
    /// itu U-030
    Um030,
    /// itu U+045
    Up045,
    /// itu U-030
    Um045,
    /// itu U+090
    Up090,
    /// itu U-090
    Um090,
    /// itu U+110
    Up110,
    /// itu U-110
    Um110,
    /// itu U+135
    Up135,
    /// itu U-135
    Um135,
    /// itu U+180
    Up180,
    /// itu T+000
    Tp000,
    /// itu B+000
    Bp000,
    /// itu B+045
    Bp045,
    /// itu B-045
    Bm045,
}

/// EBU R128 loudness analyzer.
pub struct EbuR128 {
    // The current mode.
    mode: Mode,
    // The sample rate.
    rate: u32,
    // The number of channels
    channels: u32,

    // Filtered audio data (used as ring buffer).
    audio_data: Vec<f64>,
    // Current index for audio_data.
    audio_data_index: usize,

    // How many frames are needed for a gating block. Will correspond to 400ms
    // of audio at initialization, and 100ms after the first block (75% overlap
    // as specified in the 2011 revision of BS1770).
    needed_frames: usize,

    // The channel map. Has as many elements as there are channels.
    channel_map: Vec<Channel>,

    // How many samples fit in 100ms (rounded).
    samples_in_100ms: usize,

    // Filter.
    filter: crate::filter::Filter,

    // Block energy history.
    block_energy_history: crate::history::History,

    // Short term block energy history.
    short_term_block_energy_history: crate::history::History,
    short_term_frame_counter: usize,

    // Maximum sample peak, one per channel.
    sample_peak: Vec<f64>,

    // Maximum true peak, one per channel.
    true_peak: Vec<f64>,

    // The maximum window duration in ms.
    window: usize,
    history: usize,
}

impl fmt::Debug for EbuR128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EbuR128")
            .field("mode", &self.mode)
            .field("rate", &self.rate)
            .field("channels", &self.channels)
            // Not audio data
            .field("audio_data_index", &self.audio_data_index)
            .field("needed_frames", &self.needed_frames)
            .field("channel_map", &self.channel_map)
            .field("samples_in_100ms", &self.samples_in_100ms)
            .field("filter", &self.filter)
            .field("block_energy_history", &self.block_energy_history)
            .field(
                "short_term_block_energy_history",
                &self.short_term_block_energy_history,
            )
            .field("short_term_frame_counter", &self.short_term_frame_counter)
            .field("sample_peak", &self.sample_peak)
            .field("true_peak", &self.true_peak)
            .field("window", &self.window)
            .field("history", &self.history)
            .finish()
    }
}

fn default_channel_map(channels: u32) -> Vec<Channel> {
    match channels {
        4 => vec![
            Channel::Left,
            Channel::Right,
            Channel::LeftSurround,
            Channel::RightSurround,
        ],
        5 => vec![
            Channel::Left,
            Channel::Right,
            Channel::Center,
            Channel::LeftSurround,
            Channel::RightSurround,
        ],
        _ => {
            let mut v = vec![Channel::Unused; channels as usize];

            let set_channels = std::cmp::min(channels as usize, 6);
            v[0..set_channels].copy_from_slice(
                &[
                    Channel::Left,
                    Channel::Right,
                    Channel::Center,
                    Channel::Unused,
                    Channel::LeftSurround,
                    Channel::RightSurround,
                ][..set_channels],
            );

            v
        }
    }
}

fn calc_gating_block(
    frames_per_block: usize,
    audio_data: &[f64],
    audio_data_index: usize,
    channel_map: &[Channel],
) -> f64 {
    let mut sum = 0.0;

    let channels = channel_map.len();
    assert!(audio_data_index <= audio_data.len());
    assert!(audio_data.len() % channels == 0);
    assert!(audio_data_index % channels == 0);

    for (c, channel) in channel_map.iter().enumerate() {
        if *channel == Channel::Unused {
            continue;
        }

        let mut channel_sum = 0.0;

        // XXX: Don't use channel_sum += sum() here because that gives slightly different
        // results than the C version because of rounding errors
        if audio_data_index < frames_per_block * channels {
            for frame in audio_data[..audio_data_index].chunks_exact(channels) {
                channel_sum += frame[c] * frame[c];
            }

            for frame in audio_data
                [(audio_data.len() - frames_per_block * channels + audio_data_index)..]
                .chunks_exact(channels)
            {
                channel_sum += frame[c] * frame[c];
            }
        } else {
            for frame in audio_data
                [(audio_data_index - frames_per_block * channels)..audio_data_index]
                .chunks_exact(channels)
            {
                channel_sum += frame[c] * frame[c];
            }
        }

        match channel {
            Channel::LeftSurround
            | Channel::RightSurround
            | Channel::Mp060
            | Channel::Mm060
            | Channel::Mp090
            | Channel::Mm090 => {
                channel_sum *= 1.41;
            }
            Channel::DualMono => {
                channel_sum *= 2.0;
            }
            _ => (),
        }

        sum += channel_sum;
    }

    sum /= frames_per_block as f64;

    sum
}

// For tests/benchmarks
#[cfg(feature = "internal-tests")]
pub fn calc_gating_block_internal(
    frames_per_block: usize,
    audio_data: &[f64],
    audio_data_index: usize,
    channel_map: &[Channel],
) -> f64 {
    calc_gating_block(frames_per_block, audio_data, audio_data_index, channel_map)
}

const MAX_RATE: u32 = 2822400;
const MAX_CHANNELS: u32 = 64;
const MAX_WINDOW: usize = ((3 as usize) << 30) / MAX_RATE as usize / MAX_CHANNELS as usize / 8;

impl EbuR128 {
    /// Create a new instance with the given configuration.
    pub fn new(channels: u32, rate: u32, mode: Mode) -> Result<Self, Error> {
        if channels == 0 || channels > MAX_CHANNELS {
            return Err(Error::NoMem);
        }

        if rate < 16 || rate > MAX_RATE {
            return Err(Error::NoMem);
        }

        let sample_peak = vec![0.0; channels as usize];
        let true_peak = vec![0.0; channels as usize];

        let history = usize::MAX;
        let samples_in_100ms = (rate as usize + 5) / 10;

        let window = if mode.contains(Mode::S) {
            3000
        } else if mode.contains(Mode::M) {
            400
        } else {
            return Err(Error::InvalidMode);
        };

        let mut audio_data_frames = rate as usize * window / 1000;
        if audio_data_frames % samples_in_100ms != 0 {
            // round up to multiple of samples_in_100ms
            audio_data_frames =
                (audio_data_frames + samples_in_100ms) - (audio_data_frames % samples_in_100ms);
        }

        let audio_data = vec![0.0; audio_data_frames * channels as usize];
        // start at the beginning of the buffer
        let audio_data_index = 0;

        let block_energy_history =
            crate::history::History::new(mode.contains(Mode::HISTOGRAM), history / 100);

        let short_term_block_energy_history =
            crate::history::History::new(mode.contains(Mode::HISTOGRAM), history / 3000);
        let short_term_frame_counter = 0;

        let filter = crate::filter::Filter::new(
            rate,
            channels,
            mode.contains(Mode::SAMPLE_PEAK),
            mode.contains(Mode::TRUE_PEAK),
        );

        let channel_map = default_channel_map(channels);

        // the first block needs 400ms of audio data
        let needed_frames = samples_in_100ms * 4;

        Ok(Self {
            mode,
            rate,
            channels,
            audio_data,
            audio_data_index,
            needed_frames,
            channel_map,
            samples_in_100ms,
            filter,
            block_energy_history,
            short_term_block_energy_history,
            short_term_frame_counter,
            sample_peak,
            true_peak,
            window,
            history,
        })
    }

    /// Get the configured mode.
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Get the configured number of channels.
    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// Get the configured sample rate.
    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// Get the configured channel types.
    pub fn channel_map(&self) -> &[Channel] {
        &*self.channel_map
    }

    /// Set channel type.
    ///
    /// The default is:
    ///
    /// * 0 \-> `Left`
    /// * 1 \-> `Right`
    /// * 2 \-> `Center`
    /// * 3 \-> `Unused`
    /// * 4 \-> `LeftSurround`
    /// * 5 \-> `RightSurround`
    pub fn set_channel(&mut self, channel_number: u32, value: Channel) -> Result<(), Error> {
        if channel_number >= self.channels {
            return Err(Error::InvalidChannelIndex);
        }

        if value == Channel::DualMono && (self.channels != 1 || channel_number != 0) {
            return Err(Error::InvalidChannelIndex);
        }

        self.channel_map[channel_number as usize] = value;
        Ok(())
    }

    /// Change library parameters.
    ///
    /// Note that the channel map will be reset when setting a different number of channels. The
    /// current unfinished block will be lost.
    pub fn change_parameters(&mut self, channels: u32, rate: u32) -> Result<(), Error> {
        if channels == 0 || channels > MAX_CHANNELS {
            return Err(Error::NoMem);
        }

        if rate < 16 || rate > MAX_RATE {
            return Err(Error::NoMem);
        }

        if self.rate == rate && self.channels == channels {
            return Ok(());
        }

        if self.channels != channels {
            self.channels = channels;
            self.channel_map = default_channel_map(channels);
            self.sample_peak = vec![0.0; channels as usize];
            self.true_peak = vec![0.0; channels as usize];
        }

        if self.rate != rate {
            self.rate = rate;
            self.samples_in_100ms = (rate as usize + 5) / 10;
        }

        self.filter = crate::filter::Filter::new(
            rate,
            channels,
            self.mode.contains(Mode::SAMPLE_PEAK),
            self.mode.contains(Mode::TRUE_PEAK),
        );

        let mut audio_data_frames = rate as usize * self.window / 1000;
        if audio_data_frames % self.samples_in_100ms != 0 {
            // round up to multiple of samples_in_100ms
            audio_data_frames = (audio_data_frames + self.samples_in_100ms)
                - (audio_data_frames % self.samples_in_100ms);
        }

        self.audio_data = vec![0.0; audio_data_frames * channels as usize];

        // the first block needs 400ms of audio data
        self.needed_frames = self.samples_in_100ms * 4;
        // start at the beginning of the buffer
        self.audio_data_index = 0;
        // reset short term frame counter
        self.short_term_frame_counter = 0;

        Ok(())
    }

    /// Set the maximum window duration.
    ///
    /// Set the maximum duration in ms that will be used for
    /// [`EbuR128::loudness_window`](struct.EbuR128.html#method.loudness_window). Note that this
    /// destroys the current content of the audio buffer.
    pub fn set_max_window(&mut self, window: u32) -> Result<(), Error> {
        let window = if self.mode.contains(Mode::S) {
            std::cmp::max(window, 3000)
        } else if self.mode.contains(Mode::M) {
            std::cmp::max(window, 400)
        } else {
            window
        };

        if window as usize == self.window {
            return Ok(());
        }

        if window as usize >= MAX_WINDOW {
            return Err(Error::NoMem);
        }

        self.window = window as usize;

        let mut audio_data_frames = self.rate as usize * self.window / 1000;
        if audio_data_frames % self.samples_in_100ms != 0 {
            // round up to multiple of samples_in_100ms
            audio_data_frames = (audio_data_frames + self.samples_in_100ms)
                - (audio_data_frames % self.samples_in_100ms);
        }

        self.audio_data = vec![0.0; audio_data_frames * self.channels as usize];

        // the first block needs 400ms of audio data
        self.needed_frames = self.samples_in_100ms * 4;
        // start at the beginning of the buffer
        self.audio_data_index = 0;
        // reset short term frame counter
        self.short_term_frame_counter = 0;

        Ok(())
    }

    /// Set the maximum history.
    ///
    /// Set the maximum history in ms that will be stored for loudness integration. More history
    /// provides more accurate results, but requires more resources.
    ///
    /// Applies to [`EbuR128::loudness_range`](struct.EbuR128.html#method.loudness_range) and
    /// [`EbuR128::loudness_global`](struct.EbuR128.html#method.loudness_global) when
    /// `Mode::HISTOGRAM` is not set.
    ///
    /// Default is `ULONG_MAX` (at least ~50 days). Minimum is 3000ms for `Mode::LRA` and 400ms
    /// for `Mode::M`.
    pub fn set_max_history(&mut self, history: u32) -> Result<(), Error> {
        let history = if self.mode.contains(Mode::S) {
            std::cmp::max(history, 3000)
        } else if self.mode.contains(Mode::M) {
            std::cmp::max(history, 400)
        } else {
            history
        };

        if self.history == history as usize {
            return Ok(());
        }

        self.history = history as usize;

        self.block_energy_history.set_max_size(self.history / 100);
        self.short_term_block_energy_history
            .set_max_size(self.history / 3000);

        Ok(())
    }

    fn add_frames<T: crate::filter::AsF64>(&mut self, frames: &[T]) -> Result<(), Error> {
        if frames.is_empty() {
            return Ok(());
        }

        if self.channels == 0 || frames.len() % self.channels as usize != 0 {
            return Err(Error::NoMem);
        }

        self.filter.reset_peaks();

        let mut src = frames;
        while !src.is_empty() {
            let num_frames = src.len() / self.channels as usize;
            let needed_samples = self.needed_frames * self.channels as usize;

            let dest = &mut self.audio_data
                [self.audio_data_index..(self.audio_data_index + needed_samples)];

            if num_frames >= self.needed_frames {
                let (current_frame, next_src) = src.split_at(needed_samples);
                src = next_src;
                self.filter.process(current_frame, dest, &self.channel_map);

                self.audio_data_index += needed_samples;
                if self.mode.contains(Mode::I) {
                    let energy = calc_gating_block(
                        self.samples_in_100ms * 4,
                        &self.audio_data,
                        self.audio_data_index,
                        &self.channel_map,
                    );
                    self.block_energy_history.add(energy);
                }

                if self.mode.contains(Mode::LRA) {
                    self.short_term_frame_counter += self.needed_frames;
                    if self.short_term_frame_counter == self.samples_in_100ms * 30 {
                        let energy = self.energy_shortterm()?;
                        self.short_term_block_energy_history.add(energy);
                        self.short_term_frame_counter = self.samples_in_100ms * 20;
                    }
                }

                if self.audio_data_index == self.audio_data.len() {
                    self.audio_data_index = 0;
                }

                // 100ms are needed for all blocks besides the first one
                self.needed_frames = self.samples_in_100ms;
            } else {
                self.filter
                    .process(src, &mut dest[..src.len()], &self.channel_map);

                self.audio_data_index += src.len();
                if self.mode.contains(Mode::LRA) {
                    self.short_term_frame_counter += num_frames;
                }
                self.needed_frames -= num_frames;
                src = &[];
            }
        }

        let prev_sample_peak = self.filter.sample_peak();
        for (sample_peak, prev_sample_peak) in
            self.sample_peak.iter_mut().zip(prev_sample_peak.iter())
        {
            if *prev_sample_peak > *sample_peak {
                *sample_peak = *prev_sample_peak;
            }
        }

        let prev_true_peak = self.filter.true_peak();
        for (true_peak, prev_true_peak) in self.true_peak.iter_mut().zip(prev_true_peak.iter()) {
            if *prev_true_peak > *true_peak {
                *true_peak = *prev_true_peak;
            }
        }

        Ok(())
    }

    /// Add frames to be processed.
    pub fn add_frames_i16(&mut self, frames: &[i16]) -> Result<(), Error> {
        self.add_frames(frames)
    }

    /// Add frames to be processed.
    pub fn add_frames_i32(&mut self, frames: &[i32]) -> Result<(), Error> {
        self.add_frames(frames)
    }

    /// Add frames to be processed.
    pub fn add_frames_f32(&mut self, frames: &[f32]) -> Result<(), Error> {
        self.add_frames(frames)
    }

    /// Add frames to be processed.
    pub fn add_frames_f64(&mut self, frames: &[f64]) -> Result<(), Error> {
        self.add_frames(frames)
    }

    /// Get global integrated loudness in LUFS.
    pub fn loudness_global(&self) -> Result<f64, Error> {
        if !self.mode.contains(Mode::I) {
            return Err(Error::InvalidMode);
        }

        Ok(self.block_energy_history.gated_loudness())
    }

    fn energy_in_interval(&self, interval_frames: usize) -> Result<f64, Error> {
        if interval_frames > self.audio_data.len() / self.channels as usize {
            return Err(Error::InvalidMode);
        }

        Ok(calc_gating_block(
            interval_frames,
            &self.audio_data,
            self.audio_data_index,
            &self.channel_map,
        ))
    }

    /// Get momentary loudness (last 400ms) in LUFS.
    pub fn loudness_momentary(&self) -> Result<f64, Error> {
        let energy = self.energy_in_interval(self.samples_in_100ms * 4)?;

        if energy <= 0.0 {
            return Ok(f64::MIN);
        }

        Ok(energy_to_loudness(energy))
    }

    fn energy_shortterm(&self) -> Result<f64, Error> {
        self.energy_in_interval(self.samples_in_100ms * 30)
    }

    /// Get short-term loudness (last 3s) in LUFS.
    pub fn loudness_shortterm(&self) -> Result<f64, Error> {
        let energy = self.energy_shortterm()?;

        if energy <= 0.0 {
            return Ok(f64::MIN);
        }

        Ok(energy_to_loudness(energy))
    }

    /// Get loudness of the specified window in LUFS.
    ///
    /// window must not be larger than the current window. The current window can be changed by
    /// calling [`EbuR128::set_max_window`](struct.EbuR128.html#method.set_max_window).
    pub fn loudness_window(&self, window: u32) -> Result<f64, Error> {
        if window as usize >= MAX_WINDOW {
            return Err(Error::InvalidMode);
        }

        let interval_frames = self.rate as usize * window as usize / 1000;
        let energy = self.energy_in_interval(interval_frames)?;

        if energy <= 0.0 {
            return Ok(f64::MIN);
        }

        Ok(energy_to_loudness(energy))
    }

    /// Get loudness range (LRA) of programme in LU.
    ///
    /// Calculates loudness range according to EBU 3342.
    pub fn loudness_range(&self) -> Result<f64, Error> {
        if !self.mode.contains(Mode::LRA) {
            return Err(Error::InvalidMode);
        }

        Ok(self.short_term_block_energy_history.loudness_range())
    }

    /// Get maximum sample peak from all frames that have been processed.
    ///
    /// The equation to convert to dBFS is: 20 * log10(out)
    pub fn sample_peak(&self, channel_number: u32) -> Result<f64, Error> {
        if !self.mode.contains(Mode::SAMPLE_PEAK) {
            return Err(Error::InvalidMode);
        }

        if channel_number >= self.channels {
            return Err(Error::InvalidChannelIndex);
        }

        Ok(self.sample_peak[channel_number as usize])
    }

    /// Get maximum sample peak from the last call to
    /// [`EbuR128::add_frames`](struct.EbuR128.html#method.add_frames_i16).
    ///
    /// The equation to convert to dBFS is: 20 * log10(out)
    pub fn prev_sample_peak(&self, channel_number: u32) -> Result<f64, Error> {
        if !self.mode.contains(Mode::SAMPLE_PEAK) {
            return Err(Error::InvalidMode);
        }

        if channel_number >= self.channels {
            return Err(Error::InvalidChannelIndex);
        }

        Ok(self.filter.sample_peak()[channel_number as usize])
    }

    /// Get maximum true peak from all frames that have been processed.
    ///
    /// Uses an implementation defined algorithm to calculate the true peak. Do not try to compare
    /// resulting values across different versions of the library, as the algorithm may change.
    ///
    /// The current implementation uses a custom polyphase FIR interpolator to calculate true peak.
    /// Will oversample 4x for sample rates < 96000 Hz, 2x for sample rates < 192000 Hz and leave
    /// the signal unchanged for 192000 Hz.
    ///
    /// The equation to convert to dBTP is: 20 * log10(out)
    pub fn true_peak(&self, channel_number: u32) -> Result<f64, Error> {
        if !self.mode.contains(Mode::TRUE_PEAK) {
            return Err(Error::InvalidMode);
        }

        if channel_number >= self.channels {
            return Err(Error::InvalidChannelIndex);
        }

        if self.sample_peak[channel_number as usize] > self.true_peak[channel_number as usize] {
            Ok(self.sample_peak[channel_number as usize])
        } else {
            Ok(self.true_peak[channel_number as usize])
        }
    }

    /// Get maximum true peak from the last call to
    /// [`EbuR128::add_frames`](struct.EbuR128.html#method.add_frames_i16).
    ///
    /// Uses an implementation defined algorithm to calculate the true peak. Do not try to compare
    /// resulting values across different versions of the library, as the algorithm may change.
    ///
    /// The current implementation uses a custom polyphase FIR interpolator to calculate true peak.
    /// Will oversample 4x for sample rates < 96000 Hz, 2x for sample rates < 192000 Hz and leave
    /// the signal unchanged for 192000 Hz.
    ///
    /// The equation to convert to dBTP is: 20 * log10(out)
    pub fn prev_true_peak(&self, channel_number: u32) -> Result<f64, Error> {
        if !self.mode.contains(Mode::TRUE_PEAK) {
            return Err(Error::InvalidMode);
        }

        if channel_number >= self.channels {
            return Err(Error::InvalidChannelIndex);
        }

        let sample_peak = self.filter.sample_peak();
        let true_peak = self.filter.true_peak();

        if sample_peak[channel_number as usize] > true_peak[channel_number as usize] {
            Ok(sample_peak[channel_number as usize])
        } else {
            Ok(true_peak[channel_number as usize])
        }
    }

    /// Get relative threshold in LUFS.
    pub fn relative_threshold(&self) -> Result<f64, Error> {
        if !self.mode.contains(Mode::I) {
            return Err(Error::InvalidMode);
        }

        Ok(self.block_energy_history.relative_threshold())
    }
}

#[cfg(feature = "c-tests")]
extern "C" {
    pub fn calc_gating_block_c(
        frames_per_block: usize,
        audio_data: *const f64,
        audio_data_frames: usize,
        audio_data_index: usize,
        channel_map: *const u32,
        channels: usize,
    ) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "c-tests")]
    use crate::tests::Signal;
    #[cfg(feature = "c-tests")]
    use quickcheck_macros::quickcheck;

    macro_rules! assert_eq_f64(
        ($a:expr, $b:expr) => {
            assert!(
                float_cmp::approx_eq!(f64, $a, $b, ulps = 2),
                "{} != {}",
                $a,
                $b,
            )
        }
    );

    #[test]
    fn sine_stereo_i16() {
        let mut data = vec![0i16; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator) * (std::i16::MAX - 1) as f32;
            out[0] = val as i16;
            out[1] = val as i16;
            accumulator += step;
        }

        let mut ebu = EbuR128::new(2, 48_000, Mode::all()).unwrap();
        ebu.add_frames_i16(&data).unwrap();

        assert_eq_f64!(ebu.loudness_global().unwrap(), -0.6500000000000054);
        assert_eq_f64!(ebu.loudness_momentary().unwrap(), -0.6820309226891973);
        assert_eq_f64!(ebu.loudness_shortterm().unwrap(), -0.6834583474398446);
        assert_eq_f64!(ebu.loudness_window(1).unwrap(), -0.875007988101488);
        assert_eq_f64!(ebu.loudness_range().unwrap(), 0.0);

        assert_eq_f64!(ebu.sample_peak(0).unwrap(), 0.99993896484375);
        assert_eq_f64!(ebu.sample_peak(1).unwrap(), 0.99993896484375);
        assert_eq_f64!(ebu.prev_sample_peak(0).unwrap(), 0.99993896484375);
        assert_eq_f64!(ebu.prev_sample_peak(1).unwrap(), 0.99993896484375);

        assert_eq_f64!(ebu.true_peak(0).unwrap(), 1.0007814168930054);
        assert_eq_f64!(ebu.true_peak(1).unwrap(), 1.0007814168930054);
        assert_eq_f64!(ebu.prev_true_peak(0).unwrap(), 1.0007814168930054);
        assert_eq_f64!(ebu.prev_true_peak(1).unwrap(), 1.0007814168930054);

        assert_eq_f64!(ebu.relative_threshold().unwrap(), -10.650000000000006);
    }

    #[test]
    fn sine_stereo_i32() {
        let mut data = vec![0i32; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator) * (std::i32::MAX - 1) as f32;
            out[0] = val as i32;
            out[1] = val as i32;
            accumulator += step;
        }

        let mut ebu = EbuR128::new(2, 48_000, Mode::all()).unwrap();
        ebu.add_frames_i32(&data).unwrap();

        assert_eq_f64!(ebu.loudness_global().unwrap(), -0.6500000000000054);
        assert_eq_f64!(ebu.loudness_momentary().unwrap(), -0.6813325598274425);
        assert_eq_f64!(ebu.loudness_shortterm().unwrap(), -0.6827591715105212);
        assert_eq_f64!(ebu.loudness_window(1).unwrap(), -0.8742956620040943);
        assert_eq_f64!(ebu.loudness_range().unwrap(), 0.0);

        assert_eq_f64!(ebu.sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.sample_peak(1).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(1).unwrap(), 1.0);

        assert_eq_f64!(ebu.true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.true_peak(1).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(1).unwrap(), 1.0008491277694702);

        assert_eq_f64!(ebu.relative_threshold().unwrap(), -10.650000000000006);
    }

    #[test]
    fn sine_stereo_f32() {
        let mut data = vec![0.0f32; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator);
            out[0] = val;
            out[1] = val;
            accumulator += step;
        }

        let mut ebu = EbuR128::new(2, 48_000, Mode::all()).unwrap();
        ebu.add_frames_f32(&data).unwrap();

        assert_eq_f64!(ebu.loudness_global().unwrap(), -0.6500000000000054);
        assert_eq_f64!(ebu.loudness_momentary().unwrap(), -0.6813325598268921);
        assert_eq_f64!(ebu.loudness_shortterm().unwrap(), -0.6827591715100236);
        assert_eq_f64!(ebu.loudness_window(1).unwrap(), -0.8742956620008693);
        assert_eq_f64!(ebu.loudness_range().unwrap(), 0.0);

        assert_eq_f64!(ebu.sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.sample_peak(1).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(1).unwrap(), 1.0);

        assert_eq_f64!(ebu.true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.true_peak(1).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(1).unwrap(), 1.0008491277694702);

        assert_eq_f64!(ebu.relative_threshold().unwrap(), -10.650000000000006);
    }

    #[test]
    fn sine_stereo_f64() {
        let mut data = vec![0.0f64; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator);
            out[0] = val as f64;
            out[1] = val as f64;
            accumulator += step;
        }

        let mut ebu = EbuR128::new(2, 48_000, Mode::all()).unwrap();
        ebu.add_frames_f64(&data).unwrap();

        assert_eq_f64!(ebu.loudness_global().unwrap(), -0.6500000000000054);
        assert_eq_f64!(ebu.loudness_momentary().unwrap(), -0.6813325598268921);
        assert_eq_f64!(ebu.loudness_shortterm().unwrap(), -0.6827591715100236);
        assert_eq_f64!(ebu.loudness_window(1).unwrap(), -0.8742956620008693);
        assert_eq_f64!(ebu.loudness_range().unwrap(), 0.0);

        assert_eq_f64!(ebu.sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.sample_peak(1).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(1).unwrap(), 1.0);

        assert_eq_f64!(ebu.true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.true_peak(1).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(1).unwrap(), 1.0008491277694702);

        assert_eq_f64!(ebu.relative_threshold().unwrap(), -10.650000000000006);
    }

    #[test]
    fn sine_stereo_i16_no_histogram() {
        let mut data = vec![0i16; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator) * (std::i16::MAX - 1) as f32;
            out[0] = val as i16;
            out[1] = val as i16;
            accumulator += step;
        }

        let mut ebu = EbuR128::new(2, 48_000, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu.add_frames_i16(&data).unwrap();

        assert_eq_f64!(ebu.loudness_global().unwrap(), -0.683303243667768);
        assert_eq_f64!(ebu.loudness_momentary().unwrap(), -0.6820309226891973);
        assert_eq_f64!(ebu.loudness_shortterm().unwrap(), -0.6834583474398446);
        assert_eq_f64!(ebu.loudness_window(1).unwrap(), -0.875007988101488);
        assert_eq_f64!(ebu.loudness_range().unwrap(), 0.00006950793233284625);

        assert_eq_f64!(ebu.sample_peak(0).unwrap(), 0.99993896484375);
        assert_eq_f64!(ebu.sample_peak(1).unwrap(), 0.99993896484375);
        assert_eq_f64!(ebu.prev_sample_peak(0).unwrap(), 0.99993896484375);
        assert_eq_f64!(ebu.prev_sample_peak(1).unwrap(), 0.99993896484375);

        assert_eq_f64!(ebu.true_peak(0).unwrap(), 1.0007814168930054);
        assert_eq_f64!(ebu.true_peak(1).unwrap(), 1.0007814168930054);
        assert_eq_f64!(ebu.prev_true_peak(0).unwrap(), 1.0007814168930054);
        assert_eq_f64!(ebu.prev_true_peak(1).unwrap(), 1.0007814168930054);

        assert_eq_f64!(ebu.relative_threshold().unwrap(), -10.683303243667767);
    }

    #[test]
    fn sine_stereo_i32_no_histogram() {
        let mut data = vec![0i32; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator) * (std::i32::MAX - 1) as f32;
            out[0] = val as i32;
            out[1] = val as i32;
            accumulator += step;
        }

        let mut ebu = EbuR128::new(2, 48_000, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu.add_frames_i32(&data).unwrap();

        assert_eq_f64!(ebu.loudness_global().unwrap(), -0.6826039914171368);
        assert_eq_f64!(ebu.loudness_momentary().unwrap(), -0.6813325598274425);
        assert_eq_f64!(ebu.loudness_shortterm().unwrap(), -0.6827591715105212);
        assert_eq_f64!(ebu.loudness_window(1).unwrap(), -0.8742956620040943);
        assert_eq_f64!(ebu.loudness_range().unwrap(), 0.00006921150165073442);

        assert_eq_f64!(ebu.sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.sample_peak(1).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(1).unwrap(), 1.0);

        assert_eq_f64!(ebu.true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.true_peak(1).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(1).unwrap(), 1.0008491277694702);

        assert_eq_f64!(ebu.relative_threshold().unwrap(), -10.682603991417135);
    }

    #[test]
    fn sine_stereo_f32_no_histogram() {
        let mut data = vec![0.0f32; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator);
            out[0] = val;
            out[1] = val;
            accumulator += step;
        }

        let mut ebu = EbuR128::new(2, 48_000, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu.add_frames_f32(&data).unwrap();

        assert_eq_f64!(ebu.loudness_global().unwrap(), -0.6826039914165554);
        assert_eq_f64!(ebu.loudness_momentary().unwrap(), -0.6813325598268921);
        assert_eq_f64!(ebu.loudness_shortterm().unwrap(), -0.6827591715100236);
        assert_eq_f64!(ebu.loudness_window(1).unwrap(), -0.8742956620008693);
        assert_eq_f64!(ebu.loudness_range().unwrap(), 0.00006921150169403312);

        assert_eq_f64!(ebu.sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.sample_peak(1).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(1).unwrap(), 1.0);

        assert_eq_f64!(ebu.true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.true_peak(1).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(1).unwrap(), 1.0008491277694702);

        assert_eq_f64!(ebu.relative_threshold().unwrap(), -10.682603991416554);
    }

    #[test]
    fn sine_stereo_f64_no_histogram() {
        let mut data = vec![0.0f64; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator);
            out[0] = val as f64;
            out[1] = val as f64;
            accumulator += step;
        }

        let mut ebu = EbuR128::new(2, 48_000, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu.add_frames_f64(&data).unwrap();

        assert_eq_f64!(ebu.loudness_global().unwrap(), -0.6826039914165554);
        assert_eq_f64!(ebu.loudness_momentary().unwrap(), -0.6813325598268921);
        assert_eq_f64!(ebu.loudness_shortterm().unwrap(), -0.6827591715100236);
        assert_eq_f64!(ebu.loudness_window(1).unwrap(), -0.8742956620008693);
        assert_eq_f64!(ebu.loudness_range().unwrap(), 0.00006921150169403312);

        assert_eq_f64!(ebu.sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.sample_peak(1).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(1).unwrap(), 1.0);

        assert_eq_f64!(ebu.true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.true_peak(1).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(1).unwrap(), 1.0008491277694702);

        assert_eq_f64!(ebu.relative_threshold().unwrap(), -10.682603991416554);
    }

    #[cfg(feature = "c-tests")]
    #[quickcheck]
    fn compare_c_impl_i16(signal: Signal<i16>) {
        let mut ebu = EbuR128::new(signal.channels, signal.rate, Mode::all()).unwrap();
        ebu.add_frames_i16(&signal.data).unwrap();

        let mut ebu_c =
            ebur128_c::EbuR128::new(signal.channels, signal.rate, ebur128_c::Mode::all()).unwrap();
        ebu_c.add_frames_i16(&signal.data).unwrap();

        assert_eq_f64!(
            ebu.loudness_global().unwrap(),
            ebu_c.loudness_global().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_momentary().unwrap(),
            ebu_c.loudness_momentary().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_shortterm().unwrap(),
            ebu_c.loudness_shortterm().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_window(1).unwrap(),
            ebu_c.loudness_window(1).unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_range().unwrap(),
            ebu_c.loudness_range().unwrap()
        );

        for c in 0..signal.channels {
            assert_eq_f64!(ebu.sample_peak(c).unwrap(), ebu_c.sample_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_sample_peak(c).unwrap(),
                ebu_c.prev_sample_peak(c).unwrap()
            );

            assert_eq_f64!(ebu.true_peak(c).unwrap(), ebu_c.true_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_true_peak(c).unwrap(),
                ebu_c.prev_true_peak(c).unwrap()
            );
        }

        assert_eq_f64!(
            ebu.relative_threshold().unwrap(),
            ebu_c.relative_threshold().unwrap()
        );
    }

    #[cfg(feature = "c-tests")]
    #[quickcheck]
    fn compare_c_impl_i32(signal: Signal<i32>) {
        let mut ebu = EbuR128::new(signal.channels, signal.rate, Mode::all()).unwrap();
        ebu.add_frames_i32(&signal.data).unwrap();

        let mut ebu_c =
            ebur128_c::EbuR128::new(signal.channels, signal.rate, ebur128_c::Mode::all()).unwrap();
        ebu_c.add_frames_i32(&signal.data).unwrap();

        assert_eq_f64!(
            ebu.loudness_global().unwrap(),
            ebu_c.loudness_global().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_momentary().unwrap(),
            ebu_c.loudness_momentary().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_shortterm().unwrap(),
            ebu_c.loudness_shortterm().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_window(1).unwrap(),
            ebu_c.loudness_window(1).unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_range().unwrap(),
            ebu_c.loudness_range().unwrap()
        );

        for c in 0..signal.channels {
            assert_eq_f64!(ebu.sample_peak(c).unwrap(), ebu_c.sample_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_sample_peak(c).unwrap(),
                ebu_c.prev_sample_peak(c).unwrap()
            );

            assert_eq_f64!(ebu.true_peak(c).unwrap(), ebu_c.true_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_true_peak(c).unwrap(),
                ebu_c.prev_true_peak(c).unwrap()
            );
        }

        assert_eq_f64!(
            ebu.relative_threshold().unwrap(),
            ebu_c.relative_threshold().unwrap()
        );
    }

    #[cfg(feature = "c-tests")]
    #[quickcheck]
    fn compare_c_impl_f32(signal: Signal<f32>) {
        let mut ebu = EbuR128::new(signal.channels, signal.rate, Mode::all()).unwrap();
        ebu.add_frames_f32(&signal.data).unwrap();

        let mut ebu_c =
            ebur128_c::EbuR128::new(signal.channels, signal.rate, ebur128_c::Mode::all()).unwrap();
        ebu_c.add_frames_f32(&signal.data).unwrap();

        assert_eq_f64!(
            ebu.loudness_global().unwrap(),
            ebu_c.loudness_global().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_momentary().unwrap(),
            ebu_c.loudness_momentary().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_shortterm().unwrap(),
            ebu_c.loudness_shortterm().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_window(1).unwrap(),
            ebu_c.loudness_window(1).unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_range().unwrap(),
            ebu_c.loudness_range().unwrap()
        );

        for c in 0..signal.channels {
            assert_eq_f64!(ebu.sample_peak(c).unwrap(), ebu_c.sample_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_sample_peak(c).unwrap(),
                ebu_c.prev_sample_peak(c).unwrap()
            );

            assert_eq_f64!(ebu.true_peak(c).unwrap(), ebu_c.true_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_true_peak(c).unwrap(),
                ebu_c.prev_true_peak(c).unwrap()
            );
        }

        assert_eq_f64!(
            ebu.relative_threshold().unwrap(),
            ebu_c.relative_threshold().unwrap()
        );
    }

    #[cfg(feature = "c-tests")]
    #[quickcheck]
    fn compare_c_impl_f64(signal: Signal<f64>) {
        let mut ebu = EbuR128::new(signal.channels, signal.rate, Mode::all()).unwrap();
        ebu.add_frames_f64(&signal.data).unwrap();

        let mut ebu_c =
            ebur128_c::EbuR128::new(signal.channels, signal.rate, ebur128_c::Mode::all()).unwrap();
        ebu_c.add_frames_f64(&signal.data).unwrap();

        assert_eq_f64!(
            ebu.loudness_global().unwrap(),
            ebu_c.loudness_global().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_momentary().unwrap(),
            ebu_c.loudness_momentary().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_shortterm().unwrap(),
            ebu_c.loudness_shortterm().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_window(1).unwrap(),
            ebu_c.loudness_window(1).unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_range().unwrap(),
            ebu_c.loudness_range().unwrap()
        );

        for c in 0..signal.channels {
            assert_eq_f64!(ebu.sample_peak(c).unwrap(), ebu_c.sample_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_sample_peak(c).unwrap(),
                ebu_c.prev_sample_peak(c).unwrap()
            );

            assert_eq_f64!(ebu.true_peak(c).unwrap(), ebu_c.true_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_true_peak(c).unwrap(),
                ebu_c.prev_true_peak(c).unwrap()
            );
        }

        assert_eq_f64!(
            ebu.relative_threshold().unwrap(),
            ebu_c.relative_threshold().unwrap()
        );
    }

    #[cfg(feature = "c-tests")]
    #[quickcheck]
    fn compare_c_impl_i16_no_histogram(signal: Signal<i16>) {
        let mut ebu =
            EbuR128::new(signal.channels, signal.rate, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu.add_frames_i16(&signal.data).unwrap();

        let mut ebu_c = ebur128_c::EbuR128::new(
            signal.channels,
            signal.rate,
            ebur128_c::Mode::all() & !ebur128_c::Mode::HISTOGRAM,
        )
        .unwrap();
        ebu_c.add_frames_i16(&signal.data).unwrap();

        assert_eq_f64!(
            ebu.loudness_global().unwrap(),
            ebu_c.loudness_global().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_momentary().unwrap(),
            ebu_c.loudness_momentary().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_shortterm().unwrap(),
            ebu_c.loudness_shortterm().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_window(1).unwrap(),
            ebu_c.loudness_window(1).unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_range().unwrap(),
            ebu_c.loudness_range().unwrap()
        );

        for c in 0..signal.channels {
            assert_eq_f64!(ebu.sample_peak(c).unwrap(), ebu_c.sample_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_sample_peak(c).unwrap(),
                ebu_c.prev_sample_peak(c).unwrap()
            );

            assert_eq_f64!(ebu.true_peak(c).unwrap(), ebu_c.true_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_true_peak(c).unwrap(),
                ebu_c.prev_true_peak(c).unwrap()
            );
        }

        assert_eq_f64!(
            ebu.relative_threshold().unwrap(),
            ebu_c.relative_threshold().unwrap()
        );
    }

    #[cfg(feature = "c-tests")]
    #[quickcheck]
    fn compare_c_impl_i32_no_histogram(signal: Signal<i32>) {
        let mut ebu =
            EbuR128::new(signal.channels, signal.rate, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu.add_frames_i32(&signal.data).unwrap();

        let mut ebu_c = ebur128_c::EbuR128::new(
            signal.channels,
            signal.rate,
            ebur128_c::Mode::all() & !ebur128_c::Mode::HISTOGRAM,
        )
        .unwrap();
        ebu_c.add_frames_i32(&signal.data).unwrap();

        assert_eq_f64!(
            ebu.loudness_global().unwrap(),
            ebu_c.loudness_global().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_momentary().unwrap(),
            ebu_c.loudness_momentary().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_shortterm().unwrap(),
            ebu_c.loudness_shortterm().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_window(1).unwrap(),
            ebu_c.loudness_window(1).unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_range().unwrap(),
            ebu_c.loudness_range().unwrap()
        );

        for c in 0..signal.channels {
            assert_eq_f64!(ebu.sample_peak(c).unwrap(), ebu_c.sample_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_sample_peak(c).unwrap(),
                ebu_c.prev_sample_peak(c).unwrap()
            );

            assert_eq_f64!(ebu.true_peak(c).unwrap(), ebu_c.true_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_true_peak(c).unwrap(),
                ebu_c.prev_true_peak(c).unwrap()
            );
        }

        assert_eq_f64!(
            ebu.relative_threshold().unwrap(),
            ebu_c.relative_threshold().unwrap()
        );
    }

    #[cfg(feature = "c-tests")]
    #[quickcheck]
    fn compare_c_impl_f32_no_histogram(signal: Signal<f32>) {
        let mut ebu =
            EbuR128::new(signal.channels, signal.rate, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu.add_frames_f32(&signal.data).unwrap();

        let mut ebu_c = ebur128_c::EbuR128::new(
            signal.channels,
            signal.rate,
            ebur128_c::Mode::all() & !ebur128_c::Mode::HISTOGRAM,
        )
        .unwrap();
        ebu_c.add_frames_f32(&signal.data).unwrap();

        assert_eq_f64!(
            ebu.loudness_global().unwrap(),
            ebu_c.loudness_global().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_momentary().unwrap(),
            ebu_c.loudness_momentary().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_shortterm().unwrap(),
            ebu_c.loudness_shortterm().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_window(1).unwrap(),
            ebu_c.loudness_window(1).unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_range().unwrap(),
            ebu_c.loudness_range().unwrap()
        );

        for c in 0..signal.channels {
            assert_eq_f64!(ebu.sample_peak(c).unwrap(), ebu_c.sample_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_sample_peak(c).unwrap(),
                ebu_c.prev_sample_peak(c).unwrap()
            );

            assert_eq_f64!(ebu.true_peak(c).unwrap(), ebu_c.true_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_true_peak(c).unwrap(),
                ebu_c.prev_true_peak(c).unwrap()
            );
        }

        assert_eq_f64!(
            ebu.relative_threshold().unwrap(),
            ebu_c.relative_threshold().unwrap()
        );
    }

    #[cfg(feature = "c-tests")]
    #[quickcheck]
    fn compare_c_impl_f64_no_histogram(signal: Signal<f64>) {
        let mut ebu =
            EbuR128::new(signal.channels, signal.rate, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu.add_frames_f64(&signal.data).unwrap();

        let mut ebu_c = ebur128_c::EbuR128::new(
            signal.channels,
            signal.rate,
            ebur128_c::Mode::all() & !ebur128_c::Mode::HISTOGRAM,
        )
        .unwrap();
        ebu_c.add_frames_f64(&signal.data).unwrap();

        assert_eq_f64!(
            ebu.loudness_global().unwrap(),
            ebu_c.loudness_global().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_momentary().unwrap(),
            ebu_c.loudness_momentary().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_shortterm().unwrap(),
            ebu_c.loudness_shortterm().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_window(1).unwrap(),
            ebu_c.loudness_window(1).unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_range().unwrap(),
            ebu_c.loudness_range().unwrap()
        );

        for c in 0..signal.channels {
            assert_eq_f64!(ebu.sample_peak(c).unwrap(), ebu_c.sample_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_sample_peak(c).unwrap(),
                ebu_c.prev_sample_peak(c).unwrap()
            );

            assert_eq_f64!(ebu.true_peak(c).unwrap(), ebu_c.true_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_true_peak(c).unwrap(),
                ebu_c.prev_true_peak(c).unwrap()
            );
        }

        assert_eq_f64!(
            ebu.relative_threshold().unwrap(),
            ebu_c.relative_threshold().unwrap()
        );
    }

    #[cfg(feature = "c-tests")]
    #[derive(Clone, Debug)]
    struct GatingBlock {
        frames_per_block: usize,
        audio_data: Vec<f64>,
        audio_data_index: usize,
        channels: u32,
    }

    #[cfg(feature = "c-tests")]
    impl quickcheck::Arbitrary for GatingBlock {
        fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
            use rand::Rng;

            let channels = g.gen_range(1, 16);

            let rate = 48_000;

            let samples_in_100ms = (rate + 5) / 10;
            let (frames_per_block, window) = if g.gen() {
                (4 * samples_in_100ms, 400)
            } else {
                (30 * samples_in_100ms, 3000)
            };

            let mut audio_data_frames = rate * window / 1000;
            if audio_data_frames % samples_in_100ms != 0 {
                // round up to multiple of samples_in_100ms
                audio_data_frames =
                    (audio_data_frames + samples_in_100ms) - (audio_data_frames % samples_in_100ms);
            }

            let mut audio_data = vec![0.0; audio_data_frames * channels as usize];
            for v in &mut audio_data {
                *v = g.gen_range(-1.0, 1.0);
            }

            let audio_data_index = g.gen_range(0, audio_data_frames) * channels as usize;

            GatingBlock {
                frames_per_block,
                audio_data,
                audio_data_index,
                channels,
            }
        }
    }

    #[cfg(feature = "c-tests")]
    fn default_channel_map_c(channels: u32) -> Vec<u32> {
        match channels {
            4 => vec![1, 2, 4, 5],
            5 => vec![1, 2, 3, 4, 5],
            _ => {
                let mut v = vec![0; channels as usize];

                let set_channels = std::cmp::min(channels as usize, 6);
                v[0..set_channels].copy_from_slice(&[1, 2, 3, 0, 4, 5][..set_channels]);

                v
            }
        }
    }

    #[cfg(feature = "c-tests")]
    #[quickcheck]
    fn compare_c_impl_calc_gating_block(block: GatingBlock) {
        let channel_map = default_channel_map(block.channels);
        let channel_map_c = default_channel_map_c(block.channels);

        let energy = calc_gating_block(
            block.frames_per_block,
            &block.audio_data,
            block.audio_data_index,
            &channel_map,
        );
        let energy_c = unsafe {
            calc_gating_block_c(
                block.frames_per_block,
                block.audio_data.as_ptr(),
                block.audio_data.len() / block.channels as usize,
                block.audio_data_index,
                channel_map_c.as_ptr(),
                block.channels as usize,
            )
        };

        assert_eq_f64!(energy, energy_c);
    }
}
