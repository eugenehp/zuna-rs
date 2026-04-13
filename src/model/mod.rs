pub mod rope;
pub mod norm;
pub mod feedforward;
pub mod conditioner;
pub mod attention;
pub mod cross_attention;
pub mod encoder_block;
pub mod decoder_block;
pub mod encoder;
pub mod decoder;
pub mod encoder_decoder;

use burn::prelude::*;
use burn::module::{Param, ParamId};
use burn::nn::Linear;

/// Create a [`Linear`] layer with **zero-filled** weights instead of the default
/// random (KaimingUniform) initialization.
///
/// This is used when weights will be immediately overwritten from a safetensors
/// file.  `Tensor::zeros` is essentially free compared to the ChaCha12-based
/// random fill that `LinearConfig::init` performs.
pub fn linear_zeros<B: Backend>(d_input: usize, d_output: usize, bias: bool, device: &B::Device) -> Linear<B> {
    let weight = Param::initialized(
        ParamId::new(),
        Tensor::zeros([d_input, d_output], device),
    );
    let bias = if bias {
        Some(Param::initialized(
            ParamId::new(),
            Tensor::zeros([d_output], device),
        ))
    } else {
        None
    };
    Linear { weight, bias }
}
