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

pub(super) fn dot_product_i32(left: &[i32], right: &[i32]) -> i64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| i64::from(*a) * i64::from(*b))
        .sum()
}

pub(super) fn matrix_vector_i32(
    weights: &[i32],
    input: &[i32],
    rows: usize,
    cols: usize,
    output: &mut [i64],
) {
    for (row, slot) in output.iter_mut().enumerate().take(rows) {
        let start = row * cols;
        *slot = dot_product_i32(&weights[start..(start + cols)], input);
    }
}

pub(super) fn activate_accumulator_i16(
    input: &[i16],
    acc_mul: i32,
    acc_shift: u32,
    use_screlu: bool,
    output: &mut [i32],
) {
    for (target, acc) in output.iter_mut().zip(input.iter()).take(input.len()) {
        let scaled = ((i64::from(*acc) * i64::from(acc_mul)) >> acc_shift) as i32;
        let clamped = scaled.clamp(0, 1024);
        *target = if use_screlu {
            ((i64::from(clamped) * i64::from(clamped)) / 1024) as i32
        } else {
            clamped
        };
    }
}
