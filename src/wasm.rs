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

use wasm_bindgen::prelude::*;

use std::convert::TryFrom;

#[wasm_bindgen]
pub struct EbuR128(crate::EbuR128);

#[wasm_bindgen]
#[derive(Clone, Copy)]
#[repr(u32)]
pub enum Channel {
    Unused,
    Left,
    Right,
    Center,
    LeftSurround,
    RightSurround,
    DualMono,
    MpSC,
    MmSC,
    Mp060,
    Mm060,
    Mp090,
    Mm090,
    Mp135,
    Mm135,
    Mp180,
    Up000,
    Up030,
    Um030,
    Up045,
    Um045,
    Up090,
    Um090,
    Up110,
    Um110,
    Up135,
    Um135,
    Up180,
    Tp000,
    Bp000,
    Bp045,
    Bm045,
}

// FIXME: This sucks
impl From<Channel> for crate::Channel {
    fn from(v: Channel) -> crate::Channel {
        match v {
            Channel::Unused => crate::Channel::Unused,
            Channel::Left => crate::Channel::Left,
            Channel::Right => crate::Channel::Right,
            Channel::Center => crate::Channel::Center,
            Channel::LeftSurround => crate::Channel::LeftSurround,
            Channel::RightSurround => crate::Channel::RightSurround,
            Channel::DualMono => crate::Channel::DualMono,
            Channel::MpSC => crate::Channel::MpSC,
            Channel::MmSC => crate::Channel::MmSC,
            Channel::Mp060 => crate::Channel::Mp060,
            Channel::Mm060 => crate::Channel::Mm060,
            Channel::Mp090 => crate::Channel::Mp090,
            Channel::Mm090 => crate::Channel::Mm090,
            Channel::Mp135 => crate::Channel::Mp135,
            Channel::Mm135 => crate::Channel::Mm135,
            Channel::Mp180 => crate::Channel::Mp180,
            Channel::Up000 => crate::Channel::Up000,
            Channel::Up030 => crate::Channel::Up030,
            Channel::Um030 => crate::Channel::Um030,
            Channel::Up045 => crate::Channel::Up045,
            Channel::Um045 => crate::Channel::Um045,
            Channel::Up090 => crate::Channel::Up090,
            Channel::Um090 => crate::Channel::Um090,
            Channel::Up110 => crate::Channel::Up110,
            Channel::Um110 => crate::Channel::Um110,
            Channel::Up135 => crate::Channel::Up135,
            Channel::Um135 => crate::Channel::Um135,
            Channel::Up180 => crate::Channel::Up180,
            Channel::Tp000 => crate::Channel::Tp000,
            Channel::Bp000 => crate::Channel::Bp000,
            Channel::Bp045 => crate::Channel::Bp045,
            Channel::Bm045 => crate::Channel::Bm045,
        }
    }
}

// FIXME: This sucks
impl From<crate::Channel> for Channel {
    fn from(v: crate::Channel) -> Channel {
        match v {
            crate::Channel::Unused => Channel::Unused,
            crate::Channel::Left => Channel::Left,
            crate::Channel::Right => Channel::Right,
            crate::Channel::Center => Channel::Center,
            crate::Channel::LeftSurround => Channel::LeftSurround,
            crate::Channel::RightSurround => Channel::RightSurround,
            crate::Channel::DualMono => Channel::DualMono,
            crate::Channel::MpSC => Channel::MpSC,
            crate::Channel::MmSC => Channel::MmSC,
            crate::Channel::Mp060 => Channel::Mp060,
            crate::Channel::Mm060 => Channel::Mm060,
            crate::Channel::Mp090 => Channel::Mp090,
            crate::Channel::Mm090 => Channel::Mm090,
            crate::Channel::Mp135 => Channel::Mp135,
            crate::Channel::Mm135 => Channel::Mm135,
            crate::Channel::Mp180 => Channel::Mp180,
            crate::Channel::Up000 => Channel::Up000,
            crate::Channel::Up030 => Channel::Up030,
            crate::Channel::Um030 => Channel::Um030,
            crate::Channel::Up045 => Channel::Up045,
            crate::Channel::Um045 => Channel::Um045,
            crate::Channel::Up090 => Channel::Up090,
            crate::Channel::Um090 => Channel::Um090,
            crate::Channel::Up110 => Channel::Up110,
            crate::Channel::Um110 => Channel::Um110,
            crate::Channel::Up135 => Channel::Up135,
            crate::Channel::Um135 => Channel::Um135,
            crate::Channel::Up180 => Channel::Up180,
            crate::Channel::Tp000 => Channel::Tp000,
            crate::Channel::Bp000 => Channel::Bp000,
            crate::Channel::Bp045 => Channel::Bp045,
            crate::Channel::Bm045 => Channel::Bm045,
        }
    }
}

impl TryFrom<u32> for Channel {
    type Error = JsValue;

    fn try_from(v: u32) -> Result<Channel, JsValue> {
        match v {
            _ if v == Channel::Unused as u32 => Ok(Channel::Unused),
            _ if v == Channel::Left as u32 => Ok(Channel::Left),
            _ if v == Channel::Right as u32 => Ok(Channel::Right),
            _ if v == Channel::Center as u32 => Ok(Channel::Center),
            _ if v == Channel::LeftSurround as u32 => Ok(Channel::LeftSurround),
            _ if v == Channel::RightSurround as u32 => Ok(Channel::RightSurround),
            _ if v == Channel::DualMono as u32 => Ok(Channel::DualMono),
            _ if v == Channel::MpSC as u32 => Ok(Channel::MpSC),
            _ if v == Channel::MmSC as u32 => Ok(Channel::MmSC),
            _ if v == Channel::Mp060 as u32 => Ok(Channel::Mp060),
            _ if v == Channel::Mm060 as u32 => Ok(Channel::Mm060),
            _ if v == Channel::Mp090 as u32 => Ok(Channel::Mp090),
            _ if v == Channel::Mm090 as u32 => Ok(Channel::Mm090),
            _ if v == Channel::Mp135 as u32 => Ok(Channel::Mp135),
            _ if v == Channel::Mm135 as u32 => Ok(Channel::Mm135),
            _ if v == Channel::Mp180 as u32 => Ok(Channel::Mp180),
            _ if v == Channel::Up000 as u32 => Ok(Channel::Up000),
            _ if v == Channel::Up030 as u32 => Ok(Channel::Up030),
            _ if v == Channel::Um030 as u32 => Ok(Channel::Um030),
            _ if v == Channel::Up045 as u32 => Ok(Channel::Up045),
            _ if v == Channel::Um045 as u32 => Ok(Channel::Um045),
            _ if v == Channel::Up090 as u32 => Ok(Channel::Up090),
            _ if v == Channel::Um090 as u32 => Ok(Channel::Um090),
            _ if v == Channel::Up110 as u32 => Ok(Channel::Up110),
            _ if v == Channel::Um110 as u32 => Ok(Channel::Um110),
            _ if v == Channel::Up135 as u32 => Ok(Channel::Up135),
            _ if v == Channel::Um135 as u32 => Ok(Channel::Um135),
            _ if v == Channel::Up180 as u32 => Ok(Channel::Up180),
            _ if v == Channel::Tp000 as u32 => Ok(Channel::Tp000),
            _ if v == Channel::Bp000 as u32 => Ok(Channel::Bp000),
            _ if v == Channel::Bp045 as u32 => Ok(Channel::Bp045),
            _ if v == Channel::Bm045 as u32 => Ok(Channel::Bm045),
            _ => Err(JsValue::from("Invalid channel")),
        }
    }
}

impl From<crate::Error> for JsValue {
    fn from(err: crate::Error) -> JsValue {
        JsValue::from(err.to_string())
    }
}

#[wasm_bindgen]
impl EbuR128 {
    #[wasm_bindgen(constructor)]
    pub fn new(channels: u32, rate: u32, mode: JsValue) -> Result<EbuR128, JsValue> {
        let mode = if mode.is_null() || mode.is_undefined() {
            crate::Mode::M
                | crate::Mode::S
                | crate::Mode::I
                | crate::Mode::LRA
                | crate::Mode::SAMPLE_PEAK
                | crate::Mode::TRUE_PEAK
        } else {
            let get_bool = |key: &str| match js_sys::Reflect::get(&mode, &key.into()) {
                Ok(v) => v.as_bool(),
                Err(_) => None,
            };
            let mut mode = crate::Mode::M;

            if let Some(false) = get_bool("momentary") {
                mode &= !crate::Mode::M;
            }

            if let Some(true) = get_bool("short_term") {
                mode |= crate::Mode::S;
            }

            if let Some(true) = get_bool("integrated") {
                mode |= crate::Mode::I;
            }

            if let Some(true) = get_bool("loudness_range") {
                mode |= crate::Mode::LRA;
            }

            if let Some(true) = get_bool("sample_peak") {
                mode |= crate::Mode::LRA;
            }

            if let Some(true) = get_bool("true_peak") {
                mode |= crate::Mode::TRUE_PEAK;
            }

            if let Some(true) = get_bool("histogram") {
                mode |= crate::Mode::HISTOGRAM;
            }

            mode
        };

        let ebu = crate::EbuR128::new(channels, rate, mode)?;

        Ok(EbuR128(ebu))
    }

    #[wasm_bindgen]
    pub fn mode(&self) -> JsValue {
        let mode = self.0.mode();

        let map = js_sys::Map::new();
        let insert = |key: &str, value: bool| {
            map.set(&key.into(), &value.into());
        };

        insert("momentary", mode.contains(crate::Mode::M));
        insert("short_term", mode.contains(crate::Mode::S));
        insert("integrated", mode.contains(crate::Mode::I));
        insert("loudness_range", mode.contains(crate::Mode::LRA));
        insert("sample_peak", mode.contains(crate::Mode::SAMPLE_PEAK));
        insert("true_peak", mode.contains(crate::Mode::TRUE_PEAK));
        insert("histogram", mode.contains(crate::Mode::HISTOGRAM));

        map.into()
    }

    #[wasm_bindgen]
    pub fn channels(&self) -> u32 {
        self.0.channels()
    }

    #[wasm_bindgen]
    pub fn rate(&self) -> u32 {
        self.0.rate()
    }

    // FIXME: That's how it has to be?
    #[wasm_bindgen]
    pub fn channel_map(&self) -> Box<[JsValue]> {
        self.0
            .channel_map()
            .iter()
            .map(|v| JsValue::from(Channel::from(*v) as u32))
            .collect()
    }

    #[wasm_bindgen]
    pub fn max_window(&self) -> usize {
        self.0.max_window()
    }

    #[wasm_bindgen]
    pub fn max_history(&self) -> usize {
        self.0.max_history()
    }

    #[wasm_bindgen]
    pub fn set_channel(&mut self, channel_number: u32, value: Channel) -> Result<(), JsValue> {
        self.0.set_channel(channel_number, value.into())?;

        Ok(())
    }

    #[wasm_bindgen]
    pub fn set_channel_map(&mut self, channel_map: Box<[JsValue]>) -> Result<(), JsValue> {
        let channel_map = channel_map
            .iter()
            .map(|v| {
                if js_sys::Number::is_integer(v) {
                    match v.as_f64() {
                        Some(v) => Channel::try_from(v as u32).map(crate::Channel::from),
                        None => Err(JsValue::from("Channel is no integer")),
                    }
                } else {
                    Err(JsValue::from("Channel is no integer"))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.0.set_channel_map(&channel_map)?;

        Ok(())
    }

    #[wasm_bindgen]
    pub fn change_parameters(&mut self, channels: u32, rate: u32) -> Result<(), JsValue> {
        self.0.change_parameters(channels, rate)?;

        Ok(())
    }

    #[wasm_bindgen]
    pub fn set_max_window(&mut self, window: u32) -> Result<(), JsValue> {
        self.0.set_max_window(window)?;

        Ok(())
    }

    #[wasm_bindgen]
    pub fn set_max_history(&mut self, history: u32) -> Result<(), JsValue> {
        self.0.set_max_history(history)?;

        Ok(())
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.0.reset()
    }

    #[wasm_bindgen]
    pub fn add_frames_i16(&mut self, frames: &[i16]) -> Result<(), JsValue> {
        self.0.add_frames_i16(frames)?;

        Ok(())
    }

    #[wasm_bindgen]
    pub fn add_frames_i32(&mut self, frames: &[i32]) -> Result<(), JsValue> {
        self.0.add_frames_i32(frames)?;

        Ok(())
    }

    #[wasm_bindgen]
    pub fn add_frames_f32(&mut self, frames: &[f32]) -> Result<(), JsValue> {
        self.0.add_frames_f32(frames)?;

        Ok(())
    }

    #[wasm_bindgen]
    pub fn add_frames_f64(&mut self, frames: &[f64]) -> Result<(), JsValue> {
        self.0.add_frames_f64(frames)?;

        Ok(())
    }

    #[wasm_bindgen]
    pub fn loudness_global(&self) -> Result<f64, JsValue> {
        let v = self.0.loudness_global()?;

        Ok(v)
    }

    #[wasm_bindgen]
    pub fn loudness_momentary(&self) -> Result<f64, JsValue> {
        let v = self.0.loudness_momentary()?;

        Ok(v)
    }

    #[wasm_bindgen]
    pub fn loudness_shortterm(&self) -> Result<f64, JsValue> {
        let v = self.0.loudness_shortterm()?;

        Ok(v)
    }

    #[wasm_bindgen]
    pub fn loudness_window(&self, window: u32) -> Result<f64, JsValue> {
        let v = self.0.loudness_window(window)?;

        Ok(v)
    }

    #[wasm_bindgen]
    pub fn loudness_range(&self) -> Result<f64, JsValue> {
        let v = self.0.loudness_range()?;

        Ok(v)
    }

    #[wasm_bindgen]
    pub fn sample_peak(&self, channel_number: u32) -> Result<f64, JsValue> {
        let v = self.0.sample_peak(channel_number)?;

        Ok(v)
    }

    #[wasm_bindgen]
    pub fn prev_sample_peak(&self, channel_number: u32) -> Result<f64, JsValue> {
        let v = self.0.prev_sample_peak(channel_number)?;

        Ok(v)
    }

    #[wasm_bindgen]
    pub fn true_peak(&self, channel_number: u32) -> Result<f64, JsValue> {
        let v = self.0.true_peak(channel_number)?;

        Ok(v)
    }

    #[wasm_bindgen]
    pub fn prev_true_peak(&self, channel_number: u32) -> Result<f64, JsValue> {
        let v = self.0.prev_true_peak(channel_number)?;

        Ok(v)
    }

    #[wasm_bindgen]
    pub fn relative_threshold(&self) -> Result<f64, JsValue> {
        let v = self.0.relative_threshold()?;

        Ok(v)
    }
}
