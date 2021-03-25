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
use crate::utils::Sample;

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
/// See definitions in ITU R-REC-BS 1770-4 and ITU R-REC-BS 2051-2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Channel {
    /// unused channel (for example LFE channel)
    Unused,
    /// Left or ITU M+030
    Left,
    /// Right or ITU M-030
    Right,
    /// Center or ITU M+000
    Center,
    /// Left surround or ITU M+110
    LeftSurround,
    /// Right surround or ITU M-110
    RightSurround,
    /// a channel that is counted twice
    DualMono,
    /// ITU M+SC
    MpSC,
    /// ITU M-SC
    MmSC,
    /// ITU M+060
    Mp060,
    /// ITU M-060
    Mm060,
    /// ITU M+090
    Mp090,
    /// ITU M-090
    Mm090,
    /// ITU M+135
    Mp135,
    /// ITU M-135
    Mm135,
    /// ITU M+180
    Mp180,
    /// ITU U+000
    Up000,
    /// ITU U+030
    Up030,
    /// ITU U-030
    Um030,
    /// ITU U+045
    Up045,
    /// ITU U-030
    Um045,
    /// ITU U+090
    Up090,
    /// ITU U-090
    Um090,
    /// ITU U+110
    Up110,
    /// ITU U-110
    Um110,
    /// ITU U+135
    Up135,
    /// ITU U-135
    Um135,
    /// ITU U+180
    Up180,
    /// ITU T+000
    Tp000,
    /// ITU B+000
    Bp000,
    /// ITU B+045
    Bp045,
    /// ITU B-045
    Bm045,
}

/// EBU R128 loudness analyzer.
pub struct EbuR128 {
    /// The current mode.
    mode: Mode,
    /// The sample rate.
    rate: u32,
    /// The number of channels
    channels: u32,

    /// Filtered audio data (used as ring buffer).
    audio_data: Box<[f64]>,
    /// Current index for audio_data.
    audio_data_index: usize,

    /// How many frames are needed for a gating block. Will correspond to 400ms
    /// of audio at initialization, and 100ms after the first block (75% overlap
    /// as specified in the 2011 revision of BS1770).
    needed_frames: usize,

    /// The channel map. Has as many elements as there are channels.
    channel_map: Box<[Channel]>,

    /// How many samples fit in 100ms (rounded).
    samples_in_100ms: usize,

    /// Filter.
    filter: crate::filter::Filter,

    /// Block energy history.
    block_energy_history: crate::history::History,

    /// Short term block energy history.
    short_term_block_energy_history: crate::history::History,
    short_term_frame_counter: usize,

    /// Maximum sample peak, one per channel.
    sample_peak: Box<[f64]>,

    /// Maximum true peak, one per channel.
    true_peak: Box<[f64]>,

    /// The maximum window duration in ms.
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

pub(crate) fn default_channel_map(channels: u32) -> Vec<Channel> {
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

const MAX_RATE: u32 = 2822400;
const MAX_CHANNELS: u32 = 64;

impl EbuR128 {
    /// Allocate audio data buffer used by the filter and check if we can allocate enough memory
    /// for it.
    fn allocate_audio_data(channels: u32, rate: u32, window: usize) -> Result<Box<[f64]>, Error> {
        let samples_in_100ms = (rate as usize + 5) / 10;

        let mut audio_data_frames = (rate as usize).checked_mul(window).ok_or(Error::NoMem)? / 1000;
        if audio_data_frames % samples_in_100ms != 0 {
            // round up to multiple of samples_in_100ms
            audio_data_frames = audio_data_frames
                .checked_add(samples_in_100ms)
                .ok_or(Error::NoMem)?
                - (audio_data_frames % samples_in_100ms);
        }

        let audio_data = vec![
            0.0;
            audio_data_frames
                .checked_mul(channels as usize)
                .ok_or(Error::NoMem)?
        ]
        .into_boxed_slice();

        Ok(audio_data)
    }

    /// Create a new instance with the given configuration.
    pub fn new(channels: u32, rate: u32, mode: Mode) -> Result<Self, Error> {
        if channels == 0 || channels > MAX_CHANNELS {
            return Err(Error::NoMem);
        }

        if !(16..=MAX_RATE).contains(&rate) {
            return Err(Error::NoMem);
        }

        let sample_peak = vec![0.0; channels as usize];
        let true_peak = vec![0.0; channels as usize];

        let history = std::usize::MAX;
        let samples_in_100ms = (rate as usize + 5) / 10;

        let window = if mode.contains(Mode::S) {
            3000
        } else if mode.contains(Mode::M) {
            400
        } else {
            return Err(Error::InvalidMode);
        };

        let audio_data = Self::allocate_audio_data(channels, rate, window)?;
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
            channel_map: channel_map.into_boxed_slice(),
            samples_in_100ms,
            filter,
            block_energy_history,
            short_term_block_energy_history,
            short_term_frame_counter,
            sample_peak: sample_peak.into_boxed_slice(),
            true_peak: true_peak.into_boxed_slice(),
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

    /// Get the configured maximum window duration in ms.
    pub fn max_window(&self) -> usize {
        self.window
    }

    /// Get the configured maximum history in ms.
    pub fn max_history(&self) -> usize {
        self.history
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
    /// * _ \-> `Unused`
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

    /// Set channel types.
    ///
    /// The default is:
    ///
    /// * 0 \-> `Left`
    /// * 1 \-> `Right`
    /// * 2 \-> `Center`
    /// * 3 \-> `Unused`
    /// * 4 \-> `LeftSurround`
    /// * 5 \-> `RightSurround`
    /// * _ \-> `Unused`
    pub fn set_channel_map(&mut self, channel_map: &[Channel]) -> Result<(), Error> {
        if channel_map.len() != self.channels as usize {
            return Err(Error::InvalidChannelIndex);
        }

        for (channel_number, value) in channel_map.iter().enumerate() {
            if *value == Channel::DualMono && (self.channels != 1 || channel_number != 0) {
                return Err(Error::InvalidChannelIndex);
            }
        }

        self.channel_map.copy_from_slice(channel_map);
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

        if !(16..=MAX_RATE).contains(&rate) {
            return Err(Error::NoMem);
        }

        if self.rate == rate && self.channels == channels {
            return Ok(());
        }

        self.audio_data = Self::allocate_audio_data(channels, rate, self.window)?;

        if self.channels != channels {
            self.channels = channels;
            self.channel_map = default_channel_map(channels).into_boxed_slice();
            self.sample_peak = vec![0.0; channels as usize].into_boxed_slice();
            self.true_peak = vec![0.0; channels as usize].into_boxed_slice();
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

        self.audio_data = Self::allocate_audio_data(self.channels, self.rate, window as usize)?;
        self.window = window as usize;

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

    /// Resets the current state.
    pub fn reset(&mut self) {
        // TODO: Use slice::fill() once stabilized
        for v in &mut *self.audio_data {
            *v = 0.0;
        }

        // the first block needs 400ms of audio data
        self.needed_frames = self.samples_in_100ms * 4;
        // start at the beginning of the buffer
        self.audio_data_index = 0;
        // reset short term frame counter
        self.short_term_frame_counter = 0;

        // TODO: Use slice::fill() once stabilized
        for v in &mut *self.true_peak {
            *v = 0.0;
        }
        // TODO: Use slice::fill() once stabilized
        for v in &mut *self.sample_peak {
            *v = 0.0;
        }

        self.filter.reset();
        self.block_energy_history.reset();
        self.short_term_block_energy_history.reset();
    }

    /// Process frames. This is the generic variant of the different public add_frames() functions
    /// that are defined below.
    fn add_frames<'a, T: Sample + 'a, S: crate::Samples<'a, T>>(
        &mut self,
        mut src: S,
    ) -> Result<(), Error> {
        if src.frames() == 0 {
            return Ok(());
        }

        if self.channels == 0 {
            return Err(Error::NoMem);
        }

        self.filter.reset_peaks();

        while src.frames() > 0 {
            let num_frames = src.frames();

            if num_frames >= self.needed_frames {
                let (current, next) = src.split_at(self.needed_frames);

                self.filter.process(
                    current,
                    &mut *self.audio_data,
                    self.audio_data_index,
                    &self.channel_map,
                );

                src = next;
                self.audio_data_index += self.needed_frames;

                if self.mode.contains(Mode::I) {
                    let energy = crate::filter::Filter::calc_gating_block(
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

                if self.audio_data_index == self.audio_data.len() / self.channels as usize {
                    self.audio_data_index = 0;
                }

                // 100ms are needed for all blocks besides the first one
                self.needed_frames = self.samples_in_100ms;
            } else {
                let (current, next) = src.split_at(num_frames);

                self.filter.process(
                    current,
                    &mut *self.audio_data,
                    self.audio_data_index,
                    &self.channel_map,
                );

                self.audio_data_index += num_frames;
                if self.mode.contains(Mode::LRA) {
                    self.short_term_frame_counter += num_frames;
                }

                src = next;
                self.needed_frames -= num_frames;
            }
        }

        let prev_sample_peak = self.filter.sample_peak();
        for (sample_peak, prev_sample_peak) in
            Iterator::zip(self.sample_peak.iter_mut(), prev_sample_peak.iter())
        {
            if *prev_sample_peak > *sample_peak {
                *sample_peak = *prev_sample_peak;
            }
        }

        let prev_true_peak = self.filter.true_peak();
        for (true_peak, prev_true_peak) in
            Iterator::zip(self.true_peak.iter_mut(), prev_true_peak.iter())
        {
            if *prev_true_peak > *true_peak {
                *true_peak = *prev_true_peak;
            }
        }

        Ok(())
    }

    fn seed_frames<'a, T: Sample + 'a, S: crate::Samples<'a, T>>(&mut self, src: S) {
        self.filter.seed(src, &self.channel_map);
    }

    /// Add interleaved frames to be processed.
    pub fn add_frames_i16(&mut self, frames: &[i16]) -> Result<(), Error> {
        self.add_frames(crate::Interleaved::new(frames, self.channels as usize)?)
    }

    /// Add interleaved frames to be processed.
    pub fn add_frames_i32(&mut self, frames: &[i32]) -> Result<(), Error> {
        self.add_frames(crate::Interleaved::new(frames, self.channels as usize)?)
    }

    /// Add interleaved frames to be processed.
    pub fn add_frames_f32(&mut self, frames: &[f32]) -> Result<(), Error> {
        self.add_frames(crate::Interleaved::new(frames, self.channels as usize)?)
    }

    /// Add interleaved frames to be processed.
    pub fn add_frames_f64(&mut self, frames: &[f64]) -> Result<(), Error> {
        self.add_frames(crate::Interleaved::new(frames, self.channels as usize)?)
    }

    /// Add planar frames to be processed.
    pub fn add_frames_planar_i16(&mut self, frames: &[&[i16]]) -> Result<(), Error> {
        self.add_frames(crate::Planar::new(frames)?)
    }

    /// Add planar frames to be processed.
    pub fn add_frames_planar_i32(&mut self, frames: &[&[i32]]) -> Result<(), Error> {
        self.add_frames(crate::Planar::new(frames)?)
    }

    /// Add planar frames to be processed.
    pub fn add_frames_planar_f32(&mut self, frames: &[&[f32]]) -> Result<(), Error> {
        self.add_frames(crate::Planar::new(frames)?)
    }

    /// Add planar frames to be processed.
    pub fn add_frames_planar_f64(&mut self, frames: &[&[f64]]) -> Result<(), Error> {
        self.add_frames(crate::Planar::new(frames)?)
    }

    /// Add interleaved frames to warmup filters, but not be considered for measurements.
    /// See [`EbuR128::loudness_global_multiple`] for example usage.
    pub fn seed_frames_i16(&mut self, frames: &[i16]) -> Result<(), Error> {
        self.seed_frames(crate::Interleaved::new(frames, self.channels as usize)?);
        Ok(())
    }

    /// Add interleaved frames to warmup filters, but not be considered for measurements.
    /// See [`EbuR128::loudness_global_multiple`] for example usage.
    pub fn seed_frames_i32(&mut self, frames: &[i32]) -> Result<(), Error> {
        self.seed_frames(crate::Interleaved::new(frames, self.channels as usize)?);
        Ok(())
    }

    /// Add interleaved frames to warmup filters, but not be considered for measurements.
    /// See [`EbuR128::loudness_global_multiple`] for example usage.
    pub fn seed_frames_f32(&mut self, frames: &[f32]) -> Result<(), Error> {
        self.seed_frames(crate::Interleaved::new(frames, self.channels as usize)?);
        Ok(())
    }

    /// Add interleaved frames to warmup filters, but not be considered for measurements.
    /// See [`EbuR128::loudness_global_multiple`] for example usage.
    pub fn seed_frames_f64(&mut self, frames: &[f64]) -> Result<(), Error> {
        self.seed_frames(crate::Interleaved::new(frames, self.channels as usize)?);
        Ok(())
    }

    /// Add planar frames to warmup filters, but not be considered for measurements.
    /// See [`EbuR128::loudness_global_multiple`] for example usage.
    pub fn seed_frames_planar_i16(&mut self, frames: &[&[i16]]) -> Result<(), Error> {
        self.seed_frames(crate::Planar::new(frames)?);
        Ok(())
    }

    /// Add planar frames to warmup filters, but not be considered for measurements.
    /// See [`EbuR128::loudness_global_multiple`] for example usage.
    pub fn seed_frames_planar_i32(&mut self, frames: &[&[i32]]) -> Result<(), Error> {
        self.seed_frames(crate::Planar::new(frames)?);
        Ok(())
    }

    /// Add planar frames to warmup filters, but not be considered for measurements.
    /// See [`EbuR128::loudness_global_multiple`] for example usage.
    pub fn seed_frames_planar_f32(&mut self, frames: &[&[f32]]) -> Result<(), Error> {
        self.seed_frames(crate::Planar::new(frames)?);
        Ok(())
    }

    /// Add planar frames to warmup filters, but not be considered for measurements.
    /// See [`EbuR128::loudness_global_multiple`] for example usage.
    pub fn seed_frames_planar_f64(&mut self, frames: &[&[f64]]) -> Result<(), Error> {
        self.seed_frames(crate::Planar::new(frames)?);
        Ok(())
    }

    /// Get global integrated loudness in LUFS.
    pub fn loudness_global(&self) -> Result<f64, Error> {
        if !self.mode.contains(Mode::I) {
            return Err(Error::InvalidMode);
        }

        Ok(self.block_energy_history.gated_loudness())
    }

    /// Get global integrated loudness in LUFS across multiple instances.
    ///
    /// This can be used to allow parallel iteration of long signals, assuming some care is taken:
    ///  1. Divide input-signal up in "chunks" of even 100ms samples. Make chunks overlap by 400ms, for example (0-10s, 9.6-20s, 19.6-30s, ...)
    ///  2. The first chunk is processed as normal. Then in parallel, for each remaining chunk, create a new instance of `EbuR128`, and in parallel:
    ///     1. Feed the first 100ms of the chunk (these are samples overlapping with last chunk) through `seed_frames_*` function. This is sufficient to make filter-states in each instance what they would have been if a single analyzer would have reached this point.
    ///     2. Process the remaining samples of each chunk through the analyzer
    ///  3. Call [`EbuR128::loudness_global_multiple`] over all the chunks to get the global loudness
    // FIXME: Should maybe be IntoIterator? Maybe AsRef<Self>?
    pub fn loudness_global_multiple<'a>(
        iter: impl Iterator<Item = &'a Self>,
    ) -> Result<f64, Error> {
        use smallvec::SmallVec;

        let h = iter
            .map(|e| {
                if !e.mode.contains(Mode::I) {
                    Err(Error::InvalidMode)
                } else {
                    Ok(&e.block_energy_history)
                }
            })
            .collect::<Result<SmallVec<[_; 16]>, _>>()?;

        Ok(crate::history::History::gated_loudness_multiple(&*h))
    }

    fn energy_in_interval(&self, interval_frames: usize) -> Result<f64, Error> {
        if interval_frames > self.audio_data.len() / self.channels as usize {
            return Err(Error::InvalidMode);
        }

        Ok(crate::filter::Filter::calc_gating_block(
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
            return Ok(-std::f64::INFINITY);
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
            return Ok(-std::f64::INFINITY);
        }

        Ok(energy_to_loudness(energy))
    }

    /// Get loudness of the specified window in LUFS.
    ///
    /// window must not be larger than the current window. The current window can be changed by
    /// calling [`EbuR128::set_max_window`](struct.EbuR128.html#method.set_max_window).
    pub fn loudness_window(&self, window: u32) -> Result<f64, Error> {
        let interval_frames = (self.rate as usize)
            .checked_mul(window as usize)
            .ok_or(Error::InvalidMode)?
            / 1000;
        let energy = self.energy_in_interval(interval_frames)?;

        if energy <= 0.0 {
            return Ok(-std::f64::INFINITY);
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

    /// Get loudness range (LRA) of programme in LU across multiple instances.
    ///
    /// Calculates loudness range according to EBU 3342.
    // FIXME: Should maybe be IntoIterator? Maybe AsRef<Self>?
    pub fn loudness_range_multiple<'a>(
        iter: impl IntoIterator<Item = &'a Self>,
    ) -> Result<f64, Error> {
        use smallvec::SmallVec;

        let h = iter
            .into_iter()
            .map(|e| {
                if !e.mode.contains(Mode::LRA) {
                    Err(Error::InvalidMode)
                } else {
                    Ok(&e.short_term_block_energy_history)
                }
            })
            .collect::<Result<SmallVec<[_; 16]>, _>>()?;

        crate::history::History::loudness_range_multiple(&*h)
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

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "c-tests")]
    use crate::tests::Signal;
    use float_eq::assert_float_eq;
    #[cfg(feature = "c-tests")]
    use quickcheck_macros::quickcheck;

    fn f64_max(mut values: impl Iterator<Item = f64>) -> Option<f64> {
        let mut v = values.next()?;
        for candidate in values {
            if candidate > v {
                v = candidate
            }
        }
        Some(v)
    }

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

        assert_float_eq!(
            ebu.loudness_global().unwrap(),
            -0.6500000000000054,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_momentary().unwrap(),
            -0.6820309226891973,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_shortterm().unwrap(),
            -0.6834583474398446,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_window(1).unwrap(),
            -0.875007988101488,
            abs <= 0.000001
        );
        assert_float_eq!(ebu.loudness_range().unwrap(), 0.0, abs <= 0.000001);

        assert_float_eq!(
            ebu.sample_peak(0).unwrap(),
            0.99993896484375,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.sample_peak(1).unwrap(),
            0.99993896484375,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_sample_peak(0).unwrap(),
            0.99993896484375,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_sample_peak(1).unwrap(),
            0.99993896484375,
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu.true_peak(0).unwrap(),
            1.0007814168930054,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.true_peak(1).unwrap(),
            1.0007814168930054,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(0).unwrap(),
            1.0007814168930054,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(1).unwrap(),
            1.0007814168930054,
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu.relative_threshold().unwrap(),
            -10.650000000000006,
            abs <= 0.000001
        );

        ebu.reset();

        assert_float_eq!(
            ebu.loudness_global().unwrap(),
            -std::f64::INFINITY,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_momentary().unwrap(),
            -std::f64::INFINITY,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_shortterm().unwrap(),
            -std::f64::INFINITY,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_window(1).unwrap(),
            -std::f64::INFINITY,
            abs <= 0.000001
        );
        assert_float_eq!(ebu.loudness_range().unwrap(), 0.0, abs <= 0.000001);

        assert_float_eq!(ebu.sample_peak(0).unwrap(), 0.0, abs <= 0.000001);
        assert_float_eq!(ebu.sample_peak(1).unwrap(), 0.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_sample_peak(0).unwrap(), 0.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_sample_peak(1).unwrap(), 0.0, abs <= 0.000001);

        assert_float_eq!(ebu.true_peak(0).unwrap(), 0.0, abs <= 0.000001);
        assert_float_eq!(ebu.true_peak(1).unwrap(), 0.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_true_peak(0).unwrap(), 0.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_true_peak(1).unwrap(), 0.0, abs <= 0.000001);

        assert_float_eq!(ebu.relative_threshold().unwrap(), -70.0, abs <= 0.000001);
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

        assert_float_eq!(
            ebu.loudness_global().unwrap(),
            -0.6500000000000054,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_momentary().unwrap(),
            -0.6813325598274425,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_shortterm().unwrap(),
            -0.6827591715105212,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_window(1).unwrap(),
            -0.8742956620040943,
            abs <= 0.000001
        );
        assert_float_eq!(ebu.loudness_range().unwrap(), 0.0, abs <= 0.000001);

        assert_float_eq!(ebu.sample_peak(0).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.sample_peak(1).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_sample_peak(0).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_sample_peak(1).unwrap(), 1.0, abs <= 0.000001);

        assert_float_eq!(
            ebu.true_peak(0).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.true_peak(1).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(0).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(1).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu.relative_threshold().unwrap(),
            -10.650000000000006,
            abs <= 0.000001
        );
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

        assert_float_eq!(
            ebu.loudness_global().unwrap(),
            -0.6500000000000054,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_momentary().unwrap(),
            -0.6813325598268921,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_shortterm().unwrap(),
            -0.6827591715100236,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_window(1).unwrap(),
            -0.8742956620008693,
            abs <= 0.000001
        );
        assert_float_eq!(ebu.loudness_range().unwrap(), 0.0, abs <= 0.000001);

        assert_float_eq!(ebu.sample_peak(0).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.sample_peak(1).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_sample_peak(0).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_sample_peak(1).unwrap(), 1.0, abs <= 0.000001);

        assert_float_eq!(
            ebu.true_peak(0).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.true_peak(1).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(0).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(1).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu.relative_threshold().unwrap(),
            -10.650000000000006,
            abs <= 0.000001
        );
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

        assert_float_eq!(
            ebu.loudness_global().unwrap(),
            -0.6500000000000054,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_momentary().unwrap(),
            -0.6813325598268921,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_shortterm().unwrap(),
            -0.6827591715100236,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_window(1).unwrap(),
            -0.8742956620008693,
            abs <= 0.000001
        );
        assert_float_eq!(ebu.loudness_range().unwrap(), 0.0, abs <= 0.000001);

        assert_float_eq!(ebu.sample_peak(0).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.sample_peak(1).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_sample_peak(0).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_sample_peak(1).unwrap(), 1.0, abs <= 0.000001);

        assert_float_eq!(
            ebu.true_peak(0).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.true_peak(1).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(0).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(1).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu.relative_threshold().unwrap(),
            -10.650000000000006,
            abs <= 0.000001
        );
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

        assert_float_eq!(
            ebu.loudness_global().unwrap(),
            -0.683303243667768,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_momentary().unwrap(),
            -0.6820309226891973,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_shortterm().unwrap(),
            -0.6834583474398446,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_window(1).unwrap(),
            -0.875007988101488,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_range().unwrap(),
            0.00006950793233284625,
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu.sample_peak(0).unwrap(),
            0.99993896484375,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.sample_peak(1).unwrap(),
            0.99993896484375,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_sample_peak(0).unwrap(),
            0.99993896484375,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_sample_peak(1).unwrap(),
            0.99993896484375,
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu.true_peak(0).unwrap(),
            1.0007814168930054,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.true_peak(1).unwrap(),
            1.0007814168930054,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(0).unwrap(),
            1.0007814168930054,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(1).unwrap(),
            1.0007814168930054,
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu.relative_threshold().unwrap(),
            -10.683303243667767,
            abs <= 0.000001
        );
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

        assert_float_eq!(
            ebu.loudness_global().unwrap(),
            -0.6826039914171368,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_momentary().unwrap(),
            -0.6813325598274425,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_shortterm().unwrap(),
            -0.6827591715105212,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_window(1).unwrap(),
            -0.8742956620040943,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_range().unwrap(),
            0.00006921150165073442,
            abs <= 0.000001
        );

        assert_float_eq!(ebu.sample_peak(0).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.sample_peak(1).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_sample_peak(0).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_sample_peak(1).unwrap(), 1.0, abs <= 0.000001);

        assert_float_eq!(
            ebu.true_peak(0).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.true_peak(1).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(0).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(1).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu.relative_threshold().unwrap(),
            -10.682603991417135,
            abs <= 0.000001
        );
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

        assert_float_eq!(
            ebu.loudness_global().unwrap(),
            -0.6826039914165554,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_momentary().unwrap(),
            -0.6813325598268921,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_shortterm().unwrap(),
            -0.6827591715100236,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_window(1).unwrap(),
            -0.8742956620008693,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_range().unwrap(),
            0.00006921150169403312,
            abs <= 0.000001
        );

        assert_float_eq!(ebu.sample_peak(0).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.sample_peak(1).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_sample_peak(0).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_sample_peak(1).unwrap(), 1.0, abs <= 0.000001);

        assert_float_eq!(
            ebu.true_peak(0).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.true_peak(1).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(0).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(1).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu.relative_threshold().unwrap(),
            -10.682603991416554,
            abs <= 0.000001
        );
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

        assert_float_eq!(
            ebu.loudness_global().unwrap(),
            -0.6826039914165554,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_momentary().unwrap(),
            -0.6813325598268921,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_shortterm().unwrap(),
            -0.6827591715100236,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_window(1).unwrap(),
            -0.8742956620008693,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_range().unwrap(),
            0.00006921150169403312,
            abs <= 0.000001
        );

        assert_float_eq!(ebu.sample_peak(0).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.sample_peak(1).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_sample_peak(0).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_sample_peak(1).unwrap(), 1.0, abs <= 0.000001);

        assert_float_eq!(
            ebu.true_peak(0).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.true_peak(1).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(0).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(1).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu.relative_threshold().unwrap(),
            -10.682603991416554,
            abs <= 0.000001
        );
    }

    #[test]
    fn sine_stereo_i16_planar_no_histogram() {
        let mut data = vec![0i16; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        let (fst, snd) = data.split_at_mut(48_000 * 5);
        for (fst, snd) in Iterator::zip(fst.iter_mut(), snd.iter_mut()) {
            let val = f32::sin(accumulator) * (std::i16::MAX - 1) as f32;
            *fst = val as i16;
            *snd = val as i16;
            accumulator += step;
        }

        let mut ebu = EbuR128::new(2, 48_000, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu.add_frames_planar_i16(&[fst, snd]).unwrap();

        assert_float_eq!(
            ebu.loudness_global().unwrap(),
            -0.683303243667768,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_momentary().unwrap(),
            -0.6820309226891973,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_shortterm().unwrap(),
            -0.6834583474398446,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_window(1).unwrap(),
            -0.875007988101488,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_range().unwrap(),
            0.00006950793233284625,
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu.sample_peak(0).unwrap(),
            0.99993896484375,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.sample_peak(1).unwrap(),
            0.99993896484375,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_sample_peak(0).unwrap(),
            0.99993896484375,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_sample_peak(1).unwrap(),
            0.99993896484375,
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu.true_peak(0).unwrap(),
            1.0007814168930054,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.true_peak(1).unwrap(),
            1.0007814168930054,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(0).unwrap(),
            1.0007814168930054,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(1).unwrap(),
            1.0007814168930054,
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu.relative_threshold().unwrap(),
            -10.683303243667767,
            abs <= 0.000001
        );
    }

    #[test]
    fn sine_stereo_i32_planar_no_histogram() {
        let mut data = vec![0i32; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        let (fst, snd) = data.split_at_mut(48_000 * 5);
        for (fst, snd) in Iterator::zip(fst.iter_mut(), snd.iter_mut()) {
            let val = f32::sin(accumulator) * (std::i32::MAX - 1) as f32;
            *fst = val as i32;
            *snd = val as i32;
            accumulator += step;
        }

        let mut ebu = EbuR128::new(2, 48_000, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu.add_frames_planar_i32(&[fst, snd]).unwrap();

        assert_float_eq!(
            ebu.loudness_global().unwrap(),
            -0.6826039914171368,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_momentary().unwrap(),
            -0.6813325598274425,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_shortterm().unwrap(),
            -0.6827591715105212,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_window(1).unwrap(),
            -0.8742956620040943,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_range().unwrap(),
            0.00006921150165073442,
            abs <= 0.000001
        );

        assert_float_eq!(ebu.sample_peak(0).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.sample_peak(1).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_sample_peak(0).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_sample_peak(1).unwrap(), 1.0, abs <= 0.000001);

        assert_float_eq!(
            ebu.true_peak(0).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.true_peak(1).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(0).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(1).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu.relative_threshold().unwrap(),
            -10.682603991417135,
            abs <= 0.000001
        );
    }

    #[test]
    fn sine_stereo_f32_planar_no_histogram() {
        let mut data = vec![0.0f32; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        let (fst, snd) = data.split_at_mut(48_000 * 5);
        for (fst, snd) in Iterator::zip(fst.iter_mut(), snd.iter_mut()) {
            let val = f32::sin(accumulator);
            *fst = val;
            *snd = val;
            accumulator += step;
        }

        let mut ebu = EbuR128::new(2, 48_000, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu.add_frames_planar_f32(&[fst, snd]).unwrap();

        assert_float_eq!(
            ebu.loudness_global().unwrap(),
            -0.6826039914165554,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_momentary().unwrap(),
            -0.6813325598268921,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_shortterm().unwrap(),
            -0.6827591715100236,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_window(1).unwrap(),
            -0.8742956620008693,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_range().unwrap(),
            0.00006921150169403312,
            abs <= 0.000001
        );

        assert_float_eq!(ebu.sample_peak(0).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.sample_peak(1).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_sample_peak(0).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_sample_peak(1).unwrap(), 1.0, abs <= 0.000001);

        assert_float_eq!(
            ebu.true_peak(0).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.true_peak(1).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(0).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(1).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu.relative_threshold().unwrap(),
            -10.682603991416554,
            abs <= 0.000001
        );
    }

    #[test]
    fn sine_stereo_f64_planar_no_histogram() {
        let mut data = vec![0.0f64; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        let (fst, snd) = data.split_at_mut(48_000 * 5);
        for (fst, snd) in Iterator::zip(fst.iter_mut(), snd.iter_mut()) {
            let val = f32::sin(accumulator);
            *fst = val as f64;
            *snd = val as f64;
            accumulator += step;
        }

        let mut ebu = EbuR128::new(2, 48_000, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu.add_frames_planar_f64(&[fst, snd]).unwrap();

        assert_float_eq!(
            ebu.loudness_global().unwrap(),
            -0.6826039914165554,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_momentary().unwrap(),
            -0.6813325598268921,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_shortterm().unwrap(),
            -0.6827591715100236,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_window(1).unwrap(),
            -0.8742956620008693,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.loudness_range().unwrap(),
            0.00006921150169403312,
            abs <= 0.000001
        );

        assert_float_eq!(ebu.sample_peak(0).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.sample_peak(1).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_sample_peak(0).unwrap(), 1.0, abs <= 0.000001);
        assert_float_eq!(ebu.prev_sample_peak(1).unwrap(), 1.0, abs <= 0.000001);

        assert_float_eq!(
            ebu.true_peak(0).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.true_peak(1).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(0).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );
        assert_float_eq!(
            ebu.prev_true_peak(1).unwrap(),
            1.0008491277694702,
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu.relative_threshold().unwrap(),
            -10.682603991416554,
            abs <= 0.000001
        );
    }

    #[test]
    fn sine_stereo_f32_multiple() {
        let mut data = vec![0.0f32; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator);
            out[0] = val;
            out[1] = val;
            accumulator += step;
        }

        let mut ebu1 = EbuR128::new(2, 48_000, Mode::all()).unwrap();
        ebu1.add_frames_f32(&data).unwrap();

        let mut data = vec![0.0f32; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 880.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator);
            out[0] = 0.5 * val;
            out[1] = 0.5 * val;
            accumulator += step;
        }

        let mut ebu2 = EbuR128::new(2, 48_000, Mode::all()).unwrap();
        ebu2.add_frames_f32(&data).unwrap();

        assert_float_eq!(
            EbuR128::loudness_global_multiple([&ebu1, &ebu2].iter().copied()).unwrap(),
            -2.603757953612454,
            abs <= 0.000001
        );

        assert_float_eq!(
            EbuR128::loudness_range_multiple([&ebu1, &ebu2].iter().copied()).unwrap(),
            5.599999999999995,
            abs <= 0.000001
        );
    }

    #[test]
    fn sine_stereo_f32_no_histogram_multiple() {
        let mut data = vec![0.0f32; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator);
            out[0] = val;
            out[1] = val;
            accumulator += step;
        }

        let mut ebu1 = EbuR128::new(2, 48_000, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu1.add_frames_f32(&data).unwrap();

        let mut data = vec![0.0f32; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 880.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator);
            out[0] = 0.5 * val;
            out[1] = 0.5 * val;
            accumulator += step;
        }

        let mut ebu2 = EbuR128::new(2, 48_000, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu2.add_frames_f32(&data).unwrap();

        assert_float_eq!(
            EbuR128::loudness_global_multiple([&ebu1, &ebu2].iter().copied()).unwrap(),
            -2.6302830567858275,
            abs <= 0.000001
        );

        assert_float_eq!(
            EbuR128::loudness_range_multiple([&ebu1, &ebu2].iter().copied()).unwrap(),
            5.571749801957784,
            abs <= 0.000001
        );
    }

    #[test]
    fn chunks_queue_with_true_peak() {
        let mut data = vec![0.0f32; 48_000 * 3];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(1) {
            let val = f32::sin(accumulator);
            out[0] = val;
            accumulator += step;
        }

        let mut ebu1 = EbuR128::new(1, 48_000, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu1.add_frames_f32(&data).unwrap();

        let mut ebu_chunks = Vec::new();
        for i in 0..3usize {
            let mut ebu_chunk = EbuR128::new(1, 48_000, Mode::all() & !Mode::HISTOGRAM).unwrap();
            let start_index = std::cmp::max(i as isize * 48_000, 0) as usize;
            let stop_index = std::cmp::min(start_index + 48_000 + (48_00 * 3), data.len());
            if start_index > 0 {
                ebu_chunk
                    .seed_frames_f32(&data[start_index - 48_00..start_index])
                    .unwrap();
            }
            ebu_chunk
                .add_frames_f32(&data[start_index..stop_index])
                .unwrap();
            ebu_chunks.push(ebu_chunk);
        }

        assert_float_eq!(
            ebu1.sample_peak(0).unwrap(),
            f64_max(ebu_chunks.iter().map(|meter| meter.sample_peak(0).unwrap())).unwrap(),
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu1.true_peak(0).unwrap(),
            f64_max(ebu_chunks.iter().map(|meter| meter.true_peak(0).unwrap())).unwrap(),
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu1.loudness_global().unwrap(),
            EbuR128::loudness_global_multiple(ebu_chunks.iter()).unwrap(),
            abs <= 0.000001
        );
    }

    #[test]
    fn chunks_histogram_with_true_peak() {
        let mut data = vec![0.0f32; 48_000 * 3];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(1) {
            let val = f32::sin(accumulator);
            out[0] = val;
            accumulator += step;
        }

        let mut ebu1 = EbuR128::new(1, 48_000, Mode::all() | Mode::HISTOGRAM).unwrap();
        ebu1.add_frames_f32(&data).unwrap();

        let mut ebu_chunks = Vec::new();
        for i in 0..3usize {
            let mut ebu_chunk =
                EbuR128::new(1, 48_000, Mode::all() | Mode::HISTOGRAM & !Mode::HISTOGRAM).unwrap();
            let start_index = std::cmp::max(i as isize * 48_000, 0) as usize;
            let stop_index = std::cmp::min(start_index + 48_000 + (48_00 * 3), data.len());
            if start_index > 0 {
                ebu_chunk
                    .seed_frames_f32(&data[start_index - 48_00..start_index])
                    .unwrap();
            }
            ebu_chunk
                .add_frames_f32(&data[start_index..stop_index])
                .unwrap();
            ebu_chunks.push(ebu_chunk);
        }

        assert_float_eq!(
            ebu1.sample_peak(0).unwrap(),
            f64_max(ebu_chunks.iter().map(|meter| meter.sample_peak(0).unwrap())).unwrap(),
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu1.true_peak(0).unwrap(),
            f64_max(ebu_chunks.iter().map(|meter| meter.true_peak(0).unwrap())).unwrap(),
            abs <= 0.000001
        );

        assert_float_eq!(
            ebu1.loudness_global().unwrap(),
            EbuR128::loudness_global_multiple(ebu_chunks.iter()).unwrap(),
            abs <= 0.000001
        );
    }

    #[cfg(feature = "c-tests")]
    fn compare_results(ebu: &EbuR128, ebu_c: &ebur128_c::EbuR128, channels: u32) {
        assert_float_eq!(
            ebu.loudness_global().unwrap(),
            ebu_c.loudness_global().unwrap(),
            ulps <= 2
        );
        assert_float_eq!(
            ebu.loudness_momentary().unwrap(),
            ebu_c.loudness_momentary().unwrap(),
            ulps <= 2
        );
        assert_float_eq!(
            ebu.loudness_shortterm().unwrap(),
            ebu_c.loudness_shortterm().unwrap(),
            ulps <= 2
        );
        assert_float_eq!(
            ebu.loudness_window(1).unwrap(),
            ebu_c.loudness_window(1).unwrap(),
            ulps <= 2
        );
        assert_float_eq!(
            ebu.loudness_range().unwrap(),
            ebu_c.loudness_range().unwrap(),
            ulps <= 2
        );

        for c in 0..channels {
            assert_float_eq!(
                ebu.sample_peak(c).unwrap(),
                ebu_c.sample_peak(c).unwrap(),
                ulps <= 2
            );
            assert_float_eq!(
                ebu.prev_sample_peak(c).unwrap(),
                ebu_c.prev_sample_peak(c).unwrap(),
                ulps <= 2
            );

            assert_float_eq!(
                ebu.true_peak(c).unwrap(),
                ebu_c.true_peak(c).unwrap(),
                // For a performance-boost, filter is defined as f32, causing slightly lower precision
                abs <= 0.000004,
            );
            assert_float_eq!(
                ebu.prev_true_peak(c).unwrap(),
                ebu_c.prev_true_peak(c).unwrap(),
                // For a performance-boost, filter is defined as f32, causing slightly lower precision
                abs <= 0.000004,
            );
        }

        assert_float_eq!(
            ebu.relative_threshold().unwrap(),
            ebu_c.relative_threshold().unwrap(),
            ulps <= 2
        );
    }

    #[cfg(feature = "c-tests")]
    #[quickcheck]
    fn compare_c_impl_i16(signal: Signal<i16>) {
        let mut ebu = EbuR128::new(signal.channels, signal.rate, Mode::all()).unwrap();
        ebu.add_frames_i16(&signal.data).unwrap();

        let mut ebu_c =
            ebur128_c::EbuR128::new(signal.channels, signal.rate, ebur128_c::Mode::all()).unwrap();
        ebu_c.add_frames_i16(&signal.data).unwrap();

        compare_results(&ebu, &ebu_c, signal.channels);
    }

    #[cfg(feature = "c-tests")]
    #[quickcheck]
    fn compare_c_impl_i32(signal: Signal<i32>) {
        let mut ebu = EbuR128::new(signal.channels, signal.rate, Mode::all()).unwrap();
        ebu.add_frames_i32(&signal.data).unwrap();

        let mut ebu_c =
            ebur128_c::EbuR128::new(signal.channels, signal.rate, ebur128_c::Mode::all()).unwrap();
        ebu_c.add_frames_i32(&signal.data).unwrap();

        compare_results(&ebu, &ebu_c, signal.channels);
    }

    #[cfg(feature = "c-tests")]
    #[quickcheck]
    fn compare_c_impl_f32(signal: Signal<f32>) {
        let mut ebu = EbuR128::new(signal.channels, signal.rate, Mode::all()).unwrap();
        ebu.add_frames_f32(&signal.data).unwrap();

        let mut ebu_c =
            ebur128_c::EbuR128::new(signal.channels, signal.rate, ebur128_c::Mode::all()).unwrap();
        ebu_c.add_frames_f32(&signal.data).unwrap();

        compare_results(&ebu, &ebu_c, signal.channels);
    }

    #[cfg(feature = "c-tests")]
    #[quickcheck]
    fn compare_c_impl_f64(signal: Signal<f64>) {
        let mut ebu = EbuR128::new(signal.channels, signal.rate, Mode::all()).unwrap();
        ebu.add_frames_f64(&signal.data).unwrap();

        let mut ebu_c =
            ebur128_c::EbuR128::new(signal.channels, signal.rate, ebur128_c::Mode::all()).unwrap();
        ebu_c.add_frames_f64(&signal.data).unwrap();

        compare_results(&ebu, &ebu_c, signal.channels);
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

        compare_results(&ebu, &ebu_c, signal.channels);
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

        compare_results(&ebu, &ebu_c, signal.channels);
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

        compare_results(&ebu, &ebu_c, signal.channels);
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

        compare_results(&ebu, &ebu_c, signal.channels);
    }
}
