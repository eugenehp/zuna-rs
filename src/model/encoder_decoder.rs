/// Top-level EncoderDecoder + diffusion `sample()` loop (burn 0.20.1)
///
/// Python: `EncoderDecoder.sample()` in transformer.py.
///
/// Rectified-flow sampling (Euler method):
///   z₀ ~ N(0, σ²)
///   dt = 1 / sample_steps
///   for i = sample_steps, …, 1:
///     t  = dt * i
///     vc = decoder(z, enc_out, t, ...)
///     if cfg != 1:  vc = vc_uncond + cfg*(vc - vc_uncond)
///     z  = z - dt * vc
use burn::prelude::*;
use burn::tensor::Distribution;
use crate::model::encoder::EncoderTransformer;
use crate::model::decoder::DecoderTransformer;
use crate::model::rope::RotaryEmbedding;

#[derive(Module, Debug)]
pub struct EncoderDecoder<B: Backend> {
    pub encoder: EncoderTransformer<B>,
    pub decoder: DecoderTransformer<B>,
    pub global_sigma: f32,
}

impl<B: Backend> EncoderDecoder<B> {
    pub fn new(
        input_dim:          usize,  // 32
        encoder_output_dim: usize,  // 32
        dim:                usize,  // 1024
        t_dim:              usize,  // 64
        n_layers:           usize,  // 16
        head_dim:           usize,
        n_heads:            usize,
        n_kv_heads:         usize,
        hidden_dim:         usize,
        norm_eps:           f64,
        downsample_factor:  usize,  // 1
        global_sigma:       f32,    // 0.1
        device:             &B::Device,
    ) -> Self {
        Self {
            encoder: EncoderTransformer::new(
                input_dim, encoder_output_dim, dim, n_layers,
                head_dim, n_heads, n_kv_heads, hidden_dim,
                norm_eps, downsample_factor, device,
            ),
            decoder: DecoderTransformer::new(
                input_dim, encoder_output_dim, dim, t_dim, n_layers,
                head_dim, n_heads, n_kv_heads, hidden_dim, norm_eps, device,
            ),
            global_sigma,
        }
    }

    /// encoder_input: [1, S, input_dim]  (zeroed channels = dropped)
    /// tok_idx:       [S, 4]
    /// Returns:       [1, S, input_dim]  (reconstructed EEG tokens)
    pub fn sample(
        &self,
        encoder_input: Tensor<B, 3>,
        tok_idx:       Tensor<B, 2, Int>,
        rope:          &RotaryEmbedding<B>,
        sample_steps:  usize,
        cfg:           f32,
    ) -> Tensor<B, 3> {
        let device = encoder_input.device();
        let [b, s, d] = encoder_input.dims();
        let dt = 1.0_f32 / sample_steps as f32;

        // Encode once
        let enc_out = self.encoder.forward(encoder_input.clone(), tok_idx.clone(), rope);

        // Initial noise z ~ N(0, σ²) — Python: z = global_sigma * randn_like(...)
        // burn Distribution::Normal(mean, std_dev), same convention as PyTorch.
        let sigma = self.global_sigma as f64;
        let mut z = Tensor::<B, 3>::random(
            [b, s, d],
            Distribution::Normal(0.0, sigma),
            &device,
        );

        // Diffusion loop
        for i in (1..=sample_steps).rev() {
            let t_val = dt * i as f32;
            let time_t = Tensor::<B, 3>::full([b, 1, 1], t_val, &device);

            let vc = self.decoder.forward(
                z.clone(), enc_out.clone(), time_t.clone(), tok_idx.clone(), rope,
            );

            let vc = if (cfg - 1.0).abs() > 1e-4 {
                let enc_zeros = Tensor::zeros([b, s, enc_out.dims()[2]], &device);
                let vc_uncond = self.decoder.forward(
                    z.clone(), enc_zeros, time_t, tok_idx.clone(), rope,
                );
                vc_uncond.clone() + (vc - vc_uncond).mul_scalar(cfg)
            } else {
                vc
            };

            z = z - vc.mul_scalar(dt);
        }

        z
    }
}
