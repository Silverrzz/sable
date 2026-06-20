
use crate::Board;

use super::{evaluator::Evaluator, material::is_board_drawn, types::*};

pub(super) const ACTIVATION_Q: i32 = 1024;

#[inline]
pub(super) fn integer_div_round(numerator: i64, denominator: i64) -> i32 {
    if numerator >= 0 {
        ((numerator + denominator / 2) / denominator) as i32
    } else {
        ((numerator - denominator / 2) / denominator) as i32
    }
}

#[inline]
pub(super) fn accumulator_activation(acc: i32, acc_mul: i32, acc_shift: u32, use_screlu: bool) -> i32 {
    let scaled = ((acc as i64 * acc_mul as i64) >> acc_shift) as i32;
    let clamped = scaled.clamp(0, ACTIVATION_Q);
    if use_screlu {
        ((clamped as i64 * clamped as i64) / ACTIVATION_Q as i64) as i32
    } else {
        clamped
    }
}

#[inline]
pub(super) fn integer_forward(
    accumulator: &[i16],
    inference: &NnueInference,
    scratch: &mut NnueEvalScratch,
) -> i32 {
    if inference.hidden.is_none()
        && let IntegerOutputQuantization::BulletScrelu { .. } = inference.output.quantization
    {
        return bullet_screlu_output_forward(accumulator, &inference.output);
    }

    let activations = if let Some(hidden) = &inference.hidden {
        fused_accumulator_hidden_forward(
            accumulator,
            inference.acc_mul,
            inference.acc_shift,
            inference.use_screlu,
            hidden,
            &mut scratch.hidden,
            &mut scratch.activations,
            &mut scratch.sums,
        );
        &scratch.hidden[..hidden.output_size]
    } else {
        for (target, acc) in scratch
            .activations
            .iter_mut()
        .zip(accumulator.iter())
        .take(accumulator.len())
        {
            *target = accumulator_activation(
                i32::from(*acc),
                inference.acc_mul,
                inference.acc_shift,
                inference.use_screlu,
            );
        }
        &scratch.activations[..accumulator.len()]
    };
    integer_output_forward(activations, &inference.output)
}

#[inline]
pub(super) fn integer_forward_dual(
    stm_accumulator: &[i16],
    ntm_accumulator: &[i16],
    inference: &NnueInference,
    scratch: &mut NnueEvalScratch,
) -> i32 {
    debug_assert_eq!(stm_accumulator.len(), ntm_accumulator.len());
    if inference.hidden.is_none()
        && let IntegerOutputQuantization::BulletScrelu { .. } = inference.output.quantization
    {
        return bullet_screlu_output_forward_dual(
            stm_accumulator,
            ntm_accumulator,
            &inference.output,
        );
    }

    let input_size = stm_accumulator.len() + ntm_accumulator.len();
    debug_assert!(scratch.activations.len() >= input_size);
    for (target, acc) in scratch
        .activations
        .iter_mut()
        .zip(stm_accumulator.iter().chain(ntm_accumulator.iter()))
        .take(input_size)
    {
        *target = accumulator_activation(
            i32::from(*acc),
            inference.acc_mul,
            inference.acc_shift,
            inference.use_screlu,
        );
    }
    integer_output_forward(&scratch.activations[..input_size], &inference.output)
}

#[inline]
pub(super) fn fused_accumulator_hidden_forward(
    accumulator: &[i16],
    acc_mul: i32,
    acc_shift: u32,
    use_screlu: bool,
    layer: &IntegerHiddenLayer,
    output: &mut [i32],
    activations: &mut [i32],
    sums: &mut [i64],
) {
    debug_assert_eq!(accumulator.len(), layer.input_size);
    debug_assert!(activations.len() >= layer.input_size);
    for (target, acc) in activations
        .iter_mut()
        .zip(accumulator.iter())
        .take(layer.input_size)
    {
        *target = accumulator_activation(i32::from(*acc), acc_mul, acc_shift, use_screlu);
    }
    crate::simd::matrix_vector_i32(
        &layer.weights,
        &activations[..layer.input_size],
        layer.output_size,
        layer.input_size,
        sums,
    );
    for (row, output_value) in output.iter_mut().enumerate().take(layer.output_size) {
        let weight_sum = sums[row];
        let scaled = layer.bias[row] + integer_div_round(weight_sum, i64::from(ACTIVATION_Q));
        *output_value = scaled.clamp(0, ACTIVATION_Q);
    }
}

#[inline]
pub(super) fn integer_output_forward(input: &[i32], layer: &IntegerOutputLayer) -> i32 {
    debug_assert_eq!(input.len(), layer.weights.len());
    debug_assert!(matches!(
        layer.quantization,
        IntegerOutputQuantization::ActivationQ
    ));
    let weight_sum = crate::simd::dot_product_i32(&layer.weights, input);
    let network_q = layer.bias + integer_div_round(weight_sum, i64::from(ACTIVATION_Q));
    integer_div_round(
        i64::from(network_q) * i64::from(layer.output_scale),
        i64::from(ACTIVATION_Q),
    )
}

#[inline]
pub(super) fn bullet_screlu_output_forward(accumulator: &[i16], layer: &IntegerOutputLayer) -> i32 {
    let IntegerOutputQuantization::BulletScrelu { qa, qb } = layer.quantization else {
        unreachable!("bullet SCReLU output requires bullet quantization metadata");
    };
    debug_assert_eq!(accumulator.len(), layer.weights.len());
    debug_assert!(qa > 0 && qb > 0);

    let qa = i64::from(qa);
    let qb = i64::from(qb);
    let mut output = if let Some(weights) = layer.screlu_weights_i16.as_deref() {
        crate::simd::screlu_dot_i16(accumulator, weights, qa as i16)
    } else {
        let mut output = 0_i64;
        for (&acc, &weight) in accumulator.iter().zip(layer.weights.iter()) {
            let clamped = i64::from(acc).clamp(0, qa);
            output += clamped * clamped * i64::from(weight);
        }
        output
    };
    output /= qa;
    output += i64::from(layer.bias);
    output *= i64::from(layer.output_scale);
    output /= qa * qb;
    output.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

#[inline]
pub(super) fn bullet_screlu_output_forward_dual(
    stm_accumulator: &[i16],
    ntm_accumulator: &[i16],
    layer: &IntegerOutputLayer,
) -> i32 {
    let IntegerOutputQuantization::BulletScrelu { qa, qb } = layer.quantization else {
        unreachable!("bullet SCReLU output requires bullet quantization metadata");
    };
    let hidden = stm_accumulator.len();
    debug_assert_eq!(hidden, ntm_accumulator.len());
    debug_assert_eq!(hidden * 2, layer.weights.len());
    debug_assert!(qa > 0 && qb > 0);

    let qa_i64 = i64::from(qa);
    let qb_i64 = i64::from(qb);
    let mut output = if let Some(weights) = layer.screlu_weights_i16.as_deref() {
        let stm = crate::simd::screlu_dot_i16(stm_accumulator, &weights[..hidden], qa as i16);
        let ntm = crate::simd::screlu_dot_i16(ntm_accumulator, &weights[hidden..], qa as i16);
        stm + ntm
    } else {
        let mut output = 0_i64;
        for (&acc, &weight) in stm_accumulator.iter().zip(layer.weights[..hidden].iter()) {
            let clamped = i64::from(acc).clamp(0, qa_i64);
            output += clamped * clamped * i64::from(weight);
        }
        for (&acc, &weight) in ntm_accumulator.iter().zip(layer.weights[hidden..].iter()) {
            let clamped = i64::from(acc).clamp(0, qa_i64);
            output += clamped * clamped * i64::from(weight);
        }
        output
    };
    output /= qa_i64;
    output += i64::from(layer.bias);
    output *= i64::from(layer.output_scale);
    output /= qa_i64 * qb_i64;
    output.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

pub(super) fn approximate_mul_shift(value: f64) -> (i32, u32) {
    if !value.is_finite() || value <= 0.0 {
        return (0, 0);
    }
    let mut best_mul = 1_i32;
    let mut best_shift = 0_u32;
    let mut best_error = f64::MAX;
    let saturation = (ACTIVATION_Q as f64 / value).ceil().max(1.0) as i64;
    for shift in 0..=24 {
        let mul = (value * f64::from(1_u32 << shift)).round() as i32;
        if mul <= 0 {
            continue;
        }
        let mut samples = Vec::new();
        let mut sample = 0_i64;
        while sample <= 16_384 {
            samples.push(sample);
            sample += 16;
        }
        for offset in [-2, -1, 0, 1, 2] {
            let candidate = saturation.saturating_add(offset).max(0);
            if !samples.contains(&candidate) {
                samples.push(candidate);
            }
        }
        let error = samples
            .into_iter()
            .map(|sample| {
                let exact = (sample as f64 * value).clamp(0.0, ACTIVATION_Q as f64);
                let approx = ((sample * i64::from(mul)) >> shift) as f64;
                let approx = approx.clamp(0.0, ACTIVATION_Q as f64);
                (exact - approx).abs()
            })
            .sum::<f64>();
        if error < best_error {
            best_error = error;
            best_mul = mul;
            best_shift = shift;
        }
    }
    (best_mul, best_shift)
}

pub(super) fn quantize_weight_i32(value: f32) -> i32 {
    value
        .round()
        .clamp(i32::MIN as f32, i32::MAX as f32) as i32
}

pub(super) fn build_integer_hidden_layer(layer: &QuantizedLayer) -> IntegerHiddenLayer {
    let output_size = layer.bias.len();
    let input_size = layer.input_size;
    let activation_scale = ACTIVATION_Q as f32;
    let mut weights = Vec::with_capacity(output_size * input_size);
    for row in 0..output_size {
        let row_start = row * input_size;
        for input_idx in 0..input_size {
            let weight = layer.weights[row_start + input_idx] as f32 * layer.weight_scale;
            weights.push(quantize_weight_i32(weight * activation_scale));
        }
    }
    let bias = layer
        .bias
        .iter()
        .map(|value| {
            (*value as f32 * layer.scale * activation_scale)
                .round()
                .clamp(i32::MIN as f32, i32::MAX as f32) as i32
        })
        .collect();
    IntegerHiddenLayer {
        input_size,
        output_size,
        weights,
        bias,
    }
}

pub(super) fn build_integer_output_layer(layer: &QuantizedLayer, output_scale: i32) -> IntegerOutputLayer {
    let activation_scale = ACTIVATION_Q as f32;
    let weights = layer
        .weights
        .iter()
        .map(|weight| {
            quantize_weight_i32(*weight as f32 * layer.weight_scale * activation_scale)
        })
        .collect();
    let bias = (layer.bias[0] as f32 * layer.scale * activation_scale)
        .round()
        .clamp(i32::MIN as f32, i32::MAX as f32) as i32;
    IntegerOutputLayer {
        weights,
        screlu_weights_i16: None,
        bias,
        output_scale,
        quantization: IntegerOutputQuantization::ActivationQ,
    }
}

pub(super) fn build_bullet_screlu_output_layer(
    layer: &QuantizedLayer,
    output_scale: i32,
    qa: i16,
    qb: i16,
) -> Option<IntegerOutputLayer> {
    if qa <= 0 || qb <= 0 {
        return None;
    }
    let weights: Vec<i32> = layer.weights.iter().map(|weight| i32::from(*weight)).collect();
    let max_weight_abs = layer
        .weights
        .iter()
        .map(|weight| i64::from(*weight).abs())
        .max()
        .unwrap_or(0);
    let screlu_weights_i16 = (i64::from(qa) * i64::from(qa) * max_weight_abs
        <= i64::from(i32::MAX))
    .then(|| layer.weights.clone());
    let bias = i32::try_from(*layer.bias.first()?).ok()?;
    Some(IntegerOutputLayer {
        weights,
        screlu_weights_i16,
        bias,
        output_scale,
        quantization: IntegerOutputQuantization::BulletScrelu {
            qa: i32::from(qa),
            qb: i32::from(qb),
        },
    })
}

pub(super) fn build_inference(
    layers: &[QuantizedLayer],
    output_scale: i32,
    use_screlu: bool,
    qa: i16,
    qb: i16,
) -> Option<NnueInference> {
    let first_layer = layers.first()?;
    if layers.len() < 2 {
        return None;
    }
    let (acc_mul, acc_shift) =
        approximate_mul_shift(f64::from(first_layer.scale * ACTIVATION_Q as f32));
    if layers.len() == 2 {
        return Some(NnueInference {
            acc_mul,
            acc_shift,
            use_screlu,
            hidden: None,
            output: if use_screlu {
                build_bullet_screlu_output_layer(&layers[1], output_scale, qa, qb)?
            } else {
                build_integer_output_layer(&layers[1], output_scale)
            },
        });
    }
    Some(NnueInference {
        acc_mul,
        acc_shift,
        use_screlu,
        hidden: Some(build_integer_hidden_layer(&layers[1])),
        output: build_integer_output_layer(&layers[2], output_scale),
    })
}

pub(super) fn quantize_bias(value: i16, source_scale: f32, target_scale: f32) -> Result<i64, &'static str> {
    if !source_scale.is_finite() || !target_scale.is_finite() || target_scale <= 0.0 {
        return Err("has invalid bias scale metadata");
    }
    let quantized = (f32::from(value) * source_scale / target_scale).round();
    if quantized < i64::MIN as f32 || quantized > i64::MAX as f32 {
        return Err("bias value is outside the quantized runtime range");
    }
    Ok(quantized as i64)
}

pub(crate) fn evaluate_position(board: &Board, evaluator: &Evaluator) -> i32 {
    if is_board_drawn(board) {
        return DRAW_SCORE;
    }

    evaluator.evaluate_for_side_to_move(board)
}
