/// Fourier Timestep Conditioner (burn 0.20.1)
///
/// Python (`FourierConditioner` in xattn.py):
///   weight : buffer [output_dim//2, 1]  (frozen random Fourier features)
///   proj   : Linear(output_dim, output_dim, bias=True)
///
///   forward(x):            # x in [0,1] (default min=0, max=1)
///     f = 2π · x @ weight.T
///     return proj(cat([cos(f), sin(f)], dim=-1))
///
/// The weight buffer is stored in safetensors as `decoder.t_embedder.weight`
/// with shape [32, 1] (output_dim//2 = 64//2 = 32).
use burn::prelude::*;
use burn::module::{Param, ParamId};
use burn::nn::{Linear, LinearConfig};

#[derive(Module, Debug)]
pub struct FourierConditioner<B: Backend> {
    /// Frozen random Fourier features.  Shape: [half_dim, 1].
    pub weight: Param<Tensor<B, 2>>,
    /// Linear projection after Fourier features.
    pub proj: Linear<B>,
    pub half_dim: usize,
}

impl<B: Backend> FourierConditioner<B> {
    pub fn new(output_dim: usize, device: &B::Device) -> Self {
        let half_dim = output_dim / 2;
        Self {
            weight: Param::initialized(
                ParamId::new(),
                Tensor::zeros([half_dim, 1], device),
            ),
            proj: LinearConfig::new(output_dim, output_dim)
                .with_bias(true)
                .init(device),
            half_dim,
        }
    }

    /// t: [1, 1, 1]  (scalar timestep, already in [0, 1])
    /// Returns: [1, 1, output_dim]
    pub fn forward(&self, t: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, _] = t.dims();

        // f = 2π · t @ weight.T  →  [b*s, half_dim]
        let t_flat = t.reshape([b * s, 1]);
        let w = self.weight.val();                     // [half_dim, 1]
        let f = t_flat
            .matmul(w.transpose())                     // [b*s, half_dim]
            .mul_scalar(2.0_f32 * std::f32::consts::PI)
            .reshape([b, s, self.half_dim]);

        let features = Tensor::cat(vec![f.clone().cos(), f.sin()], 2); // [b, s, output_dim]
        self.proj.forward(features)
    }
}
