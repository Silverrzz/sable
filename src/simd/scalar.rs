pub(super) fn apply_feature_delta(accumulator: &mut [i16], weights: &[i16], sign: i32) {
    if sign > 0 {
        for (acc, weight) in accumulator.iter_mut().zip(weights.iter()) {
            *acc += *weight;
        }
    } else if sign < 0 {
        for (acc, weight) in accumulator.iter_mut().zip(weights.iter()) {
            *acc -= *weight;
        }
    }
}

pub(super) fn apply_feature_deltas(
    accumulator: &mut [i16],
    feature_weights: &[i16],
    hidden_size: usize,
    features: &[usize],
    signs: &[i32],
) {
    for (idx, acc) in accumulator.iter_mut().enumerate().take(hidden_size) {
        let mut value = i32::from(*acc);
        for (&feature, &sign) in features.iter().zip(signs.iter()) {
            let weight = i32::from(feature_weights[feature * hidden_size + idx]);
            if sign > 0 {
                value += weight;
            } else if sign < 0 {
                value -= weight;
            }
        }
        *acc = value as i16;
    }
}

pub(super) fn screlu_dot_f32(accumulator: &[i16], weights: &[f32], qa: i16) -> f32 {
    let mut sums = [0.0_f32; 8];
    for (index, (&acc, &weight)) in accumulator.iter().zip(weights.iter()).enumerate() {
        let clamped = f32::from(acc.clamp(0, qa));
        sums[index % 8] += clamped * clamped * weight;
    }
    sums.into_iter().sum::<f32>() / (f32::from(qa) * f32::from(qa))
}
