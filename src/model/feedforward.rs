/// SwiGLU Feed-Forward Network (burn 0.20.1)
///
/// Python (`FeedForward` in lingua/transformer.py):
///   w1, w3 : Linear(dim, hidden_dim, bias=False)
///   w2     : Linear(hidden_dim, dim, bias=False)
///   forward(x) = w2(silu(w1(x)) * w3(x))
///
/// hidden_dim = 256 × ⌈int(2×4×dim/3) / 256⌉ = 2816 for dim=1024.
use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::silu;

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub w1: Linear<B>,
    pub w2: Linear<B>,
    pub w3: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(dim: usize, hidden_dim: usize, device: &B::Device) -> Self {
        let nobias = |i, o| LinearConfig::new(i, o).with_bias(false).init(device);
        Self {
            w1: nobias(dim, hidden_dim),
            w2: nobias(hidden_dim, dim),
            w3: nobias(dim, hidden_dim),
        }
    }

    /// x: [1, S, dim]  →  [1, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x1 = self.w1.forward(x.clone());
        let x3 = self.w3.forward(x);
        self.w2.forward(silu(x1) * x3)
    }
}
