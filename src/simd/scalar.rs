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

pub(super) fn screlu_dot_i16(accumulator: &[i16], weights: &[i16], qa: i16) -> i64 {
    let qa = i64::from(qa);
    let mut output = 0_i64;
    for (&acc, &weight) in accumulator.iter().zip(weights.iter()) {
        let clamped = i64::from(acc).clamp(0, qa);
        output += clamped * clamped * i64::from(weight);
    }
    output
}

pub(super) fn screlu_dot_i16_dual(
    left_accumulator: &[i16],
    left_weights: &[i16],
    right_accumulator: &[i16],
    right_weights: &[i16],
    qa: i16,
) -> i64 {
    screlu_dot_i16(left_accumulator, left_weights, qa)
        + screlu_dot_i16(right_accumulator, right_weights, qa)
}

