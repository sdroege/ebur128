use std::f64;

/// Data structure for polyphase FIR interpolator
#[derive(Debug)]
pub struct Interp {
    /// Interpolation factor of the interpolator
    factor: usize,
    /// Taps (prefer odd to increase zero coeffs)
    taps: usize,
    /// Number of channels
    channels: u32,
    /// Size of delay buffer
    delay: usize,
    /// List of subfilters (one for each factor)
    // TODO: Try with SmallVec
    filter: Vec<Filter>,
    /// List of delay buffers (one for each channel)
    z: Vec<f32>,
    /// Current delay buffer index
    zi: usize,
}

#[derive(Debug)]
struct Filter {
    /// List of subfilter coefficients and corresponding delay indices
    coeff: Vec<(f64, usize)>,
}

impl Interp {
    pub fn new(taps: usize, factor: usize, channels: u32) -> Self {
        let delay = (taps + factor - 1) / factor;

        // Initialize the filter memory
        // One subfilter per interpolation factor.
        let mut filter = Vec::with_capacity(factor);

        for _ in 0..factor {
            let f = Filter { coeff: vec![] };
            filter.push(f);
        }

        // One delay buffer per channel.
        let z = vec![0.0; delay * channels as usize];

        // Calculate the filter coefficients.
        for j in 0..taps {
            const ALMOST_ZERO: f64 = 0.000001;

            // Calculate Hanning window
            let w = 0.5 * (1.0 - f64::cos(2.0 * f64::consts::PI * j as f64 / (taps - 1) as f64));

            // Calculate sinc and apply hanning window
            let m = j as f64 - (taps - 1) as f64 / 2.0;
            let c = if m.abs() > ALMOST_ZERO {
                w * f64::sin(m * f64::consts::PI / factor as f64)
                    / (m * f64::consts::PI / factor as f64)
            } else {
                w
            };

            // Ignore any zero coeffs.
            if c.abs() > ALMOST_ZERO {
                // Put the coefficient into the correct subfilter
                let f = j % factor;

                let f = &mut filter[f];
                f.coeff.push((c, j / factor));
            }
        }

        Interp {
            factor,
            taps,
            channels,
            delay,
            filter,
            z,
            zi: 0,
        }
    }

    pub fn process(&mut self, src: &[f32], dst: &mut [f32]) {
        assert!(src.len() * self.factor == dst.len());
        assert!(self.z.len() == self.delay * self.channels as usize);
        assert!(self.filter.len() == self.factor);
        assert!(self.zi < self.delay);

        for (src, dst) in src
            .chunks_exact(self.channels as usize)
            .zip(dst.chunks_exact_mut(self.channels as usize * self.factor))
        {
            for (chan, (src, z)) in src
                .iter()
                .zip(self.z.chunks_exact_mut(self.delay))
                .enumerate()
            {
                // Add sample to delay buffer
                //
                // TODO Ringbuffer without bounds checks for z/zi
                //
                // Safety: zi is checked to be between 0 and self.delay
                *unsafe { z.get_unchecked_mut(self.zi) } = *src;

                // Apply coefficients
                for (filter, dst) in self
                    .filter
                    .iter()
                    .zip(dst.chunks_exact_mut(self.channels as usize))
                {
                    let mut acc = 0.0;
                    for (c, index) in &filter.coeff {
                        let mut i = self.zi as i32 - *index as i32;
                        if i < 0 {
                            i += self.delay as i32;
                        }
                        // Safety: zi is checked to be between 0 and self.delay
                        acc += *unsafe { z.get_unchecked(i as usize) } as f64 * c;
                    }

                    // TODO: Figure out how to get rid of the bounds check here
                    //
                    // Safety: chan is by construction between 0 and self.channels
                    *unsafe { dst.get_unchecked_mut(chan as usize) } = acc as f32;
                }
            }
            self.zi += 1;
            if self.zi == self.delay {
                self.zi = 0;
            }
        }
    }
}

#[cfg(feature = "internal-tests")]
use std::os::raw::c_void;

#[cfg(feature = "internal-tests")]
extern "C" {
    pub fn interp_create_c(taps: u32, factor: u32, channels: u32) -> *mut c_void;
    pub fn interp_process_c(
        interp: *mut c_void,
        frames: usize,
        src: *const f32,
        dst: *mut f32,
    ) -> usize;
    pub fn interp_destroy_c(interp: *mut c_void);
}

#[cfg(all(test, feature = "internal-tests"))]
mod tests {
    use super::*;

    #[test]
    fn compare_c_impl() {
        use float_cmp::approx_eq;

        let mut data = vec![0f32; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator);
            out[0] = val;
            out[1] = val;
            accumulator += step;
        }

        let mut data_out = vec![0.0f32; 48_000 * 5 * 2 * 2];
        let mut data_out_c = vec![0.0f32; 48_000 * 5 * 2 * 2];

        {
            let mut interp = Interp::new(49, 2, 2);
            interp.process(&data, &mut data_out);
        }

        unsafe {
            let interp = interp_create_c(49, 2, 2);
            interp_process_c(interp, 48_000 * 5, data.as_ptr(), data_out_c.as_mut_ptr());
            interp_destroy_c(interp);
        }

        for (i, (r, c)) in data_out.iter().zip(data_out_c.iter()).enumerate() {
            assert!(
                approx_eq!(f32, *r, *c, ulps = 2),
                "Rust and C implementation differ at sample {}: {} != {}",
                i,
                r,
                c
            );
        }
    }
}
