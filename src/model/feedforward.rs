/// SwiGLU Feed-Forward Network (burn 0.20.1)
///
/// Python (`FeedForward` in lingua/transformer.py):
///   w1, w3 : Linear(dim, hidden_dim, bias=False)
///   w2     : Linear(hidden_dim, dim, bias=False)
///   forward(x) = w2(silu(w1(x)) * w3(x))
///
/// hidden_dim = 256 × ⌈int(2×4×dim/3) / 256⌉ = 2816 for dim=1024.
use burn::prelude::*;
use burn::nn::Linear;
use burn::tensor::activation::silu;
use crate::model::linear_zeros;

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub w1: Linear<B>,
    pub w2: Linear<B>,
    pub w3: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(dim: usize, hidden_dim: usize, device: &B::Device) -> Self {
        let z = |i, o| linear_zeros(i, o, false, device);
        Self {
            w1: z(dim, hidden_dim),
            w2: z(hidden_dim, dim),
            w3: z(dim, hidden_dim),
        }
    }

    /// x: [1, S, dim]  →  [1, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x1 = self.w1.forward(x.clone());
        let x3 = self.w3.forward(x);
        self.w2.forward(silu(x1) * x3)
    }
}
