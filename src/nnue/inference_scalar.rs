use super::{L1, L2, NET, Q, Q0, scale_output};

pub(super) const BACKEND_NAME: &str = "scalar";

pub(super) fn feed_forward(us: &[i16; L1], them: &[i16; L1], hmc: &[i16; L1]) -> i32 {
    let mut activations = [0; L1];
    for (values, output) in [us, them].into_iter().zip(activations.chunks_exact_mut(L1 / 2)) {
        for (index, activation) in output.iter_mut().enumerate() {
            let other = index + L1 / 2;
            let left = values[index].wrapping_add(hmc[index]).clamp(0, Q0);
            let right = values[other].wrapping_add(hmc[other]).clamp(0, Q0);
            *activation = (i32::from(left) * i32::from(right)) >> 9;
        }
    }

    let mut l1 = NET.l1b;
    for (inputs, weights) in activations.as_chunks::<4>().0.iter().zip(&NET.l1w) {
        if *inputs == [0; 4] {
            continue;
        }
        for (sum, weights) in l1.iter_mut().zip(weights.as_chunks::<4>().0) {
            let dot: i32 = inputs
                .iter()
                .zip(weights)
                .map(|(&input, &weight)| input * i32::from(weight))
                .sum();
            *sum = sum.wrapping_add(dot);
        }
    }

    let mut l2 = NET.l2b;
    for (index, &raw) in l1[..L2 - 1].iter().enumerate() {
        let shifted = raw >> 2;
        let clipped = shifted.clamp(0, Q.pow(2));
        let squared = shifted.wrapping_mul(shifted).min(Q.pow(4)) >> 12;
        let linear_weights = &NET.l2w[index];
        let squared_weights = &NET.l2w[L2 - 1 + index];
        for ((sum, &linear), &quadratic) in l2.iter_mut().zip(linear_weights).zip(squared_weights) {
            *sum = sum
                .wrapping_add(linear.wrapping_mul(clipped))
                .wrapping_add(quadratic.wrapping_mul(squared));
        }
    }

    let output = l2.into_iter().zip(&NET.l3w).fold(NET.l3b, |sum, (value, &weight)| {
        sum.wrapping_add(value.clamp(0, Q.pow(3)).wrapping_mul(weight))
    });
    scale_output(output, l1[L2 - 1])
}
