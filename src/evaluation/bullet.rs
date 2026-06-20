use std::path::Path;

use crate::EngineError;

use super::{
    features::{build_first_layer_feature_weights, validate_i16_accumulator_range},
    integer::{build_inference, quantize_bias},
    io::{invalid_eval_file, read_i16, read_i32, read_u32},
    types::*,
};

#[derive(Clone, Copy, Debug)]
pub(super) struct BulletQuantHeader {
    input_features: usize,
    hidden_size: usize,
    output_scale: i32,
    flags: u32,
    qa: i16,
    qb: i16,
}

pub(super) fn build_model_from_bullet_quantized(path: &Path, bytes: &[u8]) -> Result<NnueModel, EngineError> {
    let header = parse_bullet_quant_header(path, bytes)?;
    let architecture = validate_input_features(path, header.input_features, header.flags)?;

    build_model_from_bullet_quantized_layout(
        path,
        &bytes[BULLET_QUANT_HEADER_LEN..],
        BulletQuantLayout {
            architecture,
            input_features: header.input_features,
            hidden_size: header.hidden_size,
            output_scale: header.output_scale,
            flags: header.flags,
            qa: header.qa,
            qb: header.qb,
            side_to_move_relative: header.flags & BULLET_FLAG_HAS_SIDE_TO_MOVE == 0,
            use_screlu: true,
        },
    )
}

pub(super) fn build_model_from_native_bullet_quantized(path: &Path, bytes: &[u8]) -> Result<NnueModel, EngineError> {
    let layout = infer_native_bullet_quant_layout(path, bytes)?;
    build_model_from_bullet_quantized_layout(path, bytes, layout)
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BulletQuantLayout {
    pub(super) architecture: NnueArchitecture,
    pub(super) input_features: usize,
    pub(super) hidden_size: usize,
    pub(super) output_scale: i32,
    pub(super) flags: u32,
    pub(super) qa: i16,
    pub(super) qb: i16,
    pub(super) side_to_move_relative: bool,
    pub(super) use_screlu: bool,
}

pub(super) fn build_model_from_bullet_quantized_layout(
    path: &Path,
    payload: &[u8],
    layout: BulletQuantLayout,
) -> Result<NnueModel, EngineError> {
    let architecture = validate_input_features(path, layout.input_features, layout.flags)?;
    if architecture != layout.architecture {
        return Err(invalid_eval_file(
            path,
            "bullet quantized architecture metadata does not match the declared input features",
        ));
    }

    let hidden = layout.hidden_size;
    if hidden == 0 {
        return Err(invalid_eval_file(path, "bullet quantized hidden size must be positive"));
    }

    let input_features = layout.input_features;
    let first_weights_count = hidden
        .checked_mul(input_features)
        .ok_or_else(|| invalid_eval_file(path, "bullet quantized first layer size overflow"))?;
    let first_bias_count = hidden;
    let output_weights_count = layout.architecture.output_input_size(hidden);
    let output_bias_count = 1_usize;
    let total_values = first_weights_count
        .checked_add(first_bias_count)
        .and_then(|count| count.checked_add(output_weights_count))
        .and_then(|count| count.checked_add(output_bias_count))
        .ok_or_else(|| invalid_eval_file(path, "bullet quantized tensor count overflow"))?;
    let required_bytes = total_values
        .checked_mul(2)
        .ok_or_else(|| invalid_eval_file(path, "bullet quantized payload size overflow"))?;
    if payload.len() < required_bytes {
        return Err(invalid_eval_file(
            path,
            "bullet quantized payload is shorter than the declared network layout",
        ));
    }
    let trailing = payload.len() - required_bytes;
    if trailing >= 64 {
        return Err(invalid_eval_file(
            path,
            "bullet quantized payload has too much trailing data",
        ));
    }
    if !has_valid_bullet_padding(&payload[required_bytes..]) {
        return Err(invalid_eval_file(path, "bullet quantized padding is invalid"));
    }

    let values = payload[..required_bytes]
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    let first_weights_end = first_weights_count;
    let first_bias_end = first_weights_end + first_bias_count;
    let output_weights_end = first_bias_end + output_weights_count;

    let first_weights_column_major = &values[..first_weights_end];
    let first_bias_values = &values[first_weights_end..first_bias_end];
    let output_weights = values[first_bias_end..output_weights_end].to_vec();
    let output_bias_value = values[output_weights_end];

    let mut first_weights = vec![0; first_weights_count];
    for feature in 0..input_features {
        let column_start = feature * hidden;
        for neuron in 0..hidden {
            first_weights[(neuron * input_features) + feature] =
                first_weights_column_major[column_start + neuron];
        }
    }

    let qa = f32::from(layout.qa);
    let qb = f32::from(layout.qb);
    if qa <= 0.0 || qb <= 0.0 {
        return Err(invalid_eval_file(
            path,
            "bullet quantized header must declare positive quantization factors",
        ));
    }
    let first_weight_scale = 1.0 / qa;
    let output_weight_scale = 1.0 / qb;
    let first_layer_scale = first_weight_scale;
    let output_layer_scale = first_layer_scale * output_weight_scale;
    let first_bias = first_bias_values
        .iter()
        .copied()
        .map(|value| quantize_bias(value, first_weight_scale, first_layer_scale))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|message| invalid_eval_file(path, message))?;
    let output_bias = vec![
        quantize_bias(
            output_bias_value,
            1.0 / (qa * qb),
            output_layer_scale,
        )
        .map_err(|message| invalid_eval_file(path, message))?,
    ];

    let first_layer = QuantizedLayer {
        input_size: input_features,
        weights: first_weights,
        bias: first_bias,
        scale: first_layer_scale,
        weight_scale: first_weight_scale,
    };
    let output_layer = QuantizedLayer {
        input_size: output_weights_count,
        weights: output_weights,
        bias: output_bias,
        scale: output_layer_scale,
        weight_scale: output_weight_scale,
    };
    let layers = vec![first_layer, output_layer];
    let inference = build_inference(
        &layers,
        layout.output_scale,
        layout.use_screlu,
        layout.qa,
        layout.qb,
    )
    .ok_or_else(|| {
        invalid_eval_file(path, "bullet quantized network could not build inference tables")
    })?;
    let first_layer_feature_weights = build_first_layer_feature_weights(
        path,
        &layers[0].weights,
        hidden,
        input_features,
    )?;
    validate_i16_accumulator_range(
        path,
        layout.architecture,
        &layers[0],
        &first_layer_feature_weights,
        hidden,
        layout.flags & BULLET_FLAG_HAS_SIDE_TO_MOVE != 0,
    )?;

    Ok(NnueModel {
        architecture: layout.architecture,
        layers,
        inference,
        output_scale: layout.output_scale,
        has_side_to_move_feature: layout.flags & BULLET_FLAG_HAS_SIDE_TO_MOVE != 0,
        side_to_move_relative: layout.side_to_move_relative,
        first_layer_feature_weights,
    })
}

pub(super) fn infer_native_bullet_quant_layout(path: &Path, bytes: &[u8]) -> Result<BulletQuantLayout, EngineError> {
    for (architecture, input_features, flags) in [
        (NnueArchitecture::nightweave(), KING_BUCKET_FEATURES, 0),
        (NnueArchitecture::vex(), VEX_INPUT_FEATURES, 0),
        (NnueArchitecture::nightweave(), SIDE_TO_MOVE_FEATURE + 1, BULLET_FLAG_HAS_SIDE_TO_MOVE),
    ] {
        let Some(hidden_size) = infer_native_hidden_size_with_output_factor(
            bytes.len(),
            input_features,
            architecture.output_input_size(1),
        ) else {
            continue;
        };
        let required_values = native_bullet_value_count(
            hidden_size,
            input_features,
            architecture.output_input_size(hidden_size),
        )
            .ok_or_else(|| invalid_eval_file(path, "native bullet quantized size overflow"))?;
        let required_bytes = required_values
            .checked_mul(2)
            .ok_or_else(|| invalid_eval_file(path, "native bullet quantized byte size overflow"))?;
        if has_valid_bullet_padding(&bytes[required_bytes..]) {
            return Ok(BulletQuantLayout {
                architecture,
                input_features,
                hidden_size,
                output_scale: NATIVE_BULLET_OUTPUT_SCALE,
                flags,
                qa: NATIVE_BULLET_QA,
                qb: NATIVE_BULLET_QB,
                side_to_move_relative: true,
                use_screlu: true,
            });
        }
    }

    Err(invalid_eval_file(
        path,
        "unknown native bullet quantized network layout",
    ))
}

pub(super) fn infer_native_hidden_size_with_output_factor(
    file_len: usize,
    input_features: usize,
    output_factor: usize,
) -> Option<usize> {
    let values_with_padding = file_len / 2;
    let values_per_hidden = input_features.checked_add(1)?.checked_add(output_factor)?;
    let min_values = file_len.saturating_sub(63).div_ceil(2);
    for values in min_values..=values_with_padding {
        if values <= 1 {
            continue;
        }
        let without_output_bias = values - 1;
        if without_output_bias % values_per_hidden == 0 {
            let hidden = without_output_bias / values_per_hidden;
            if hidden > 0 {
                return Some(hidden);
            }
        }
    }
    None
}

pub(super) fn native_bullet_value_count(
    hidden_size: usize,
    input_features: usize,
    output_weights: usize,
) -> Option<usize> {
    hidden_size
        .checked_mul(input_features)?
        .checked_add(hidden_size)?
        .checked_add(output_weights)?
        .checked_add(1)
}

pub(super) fn has_valid_bullet_padding(padding: &[u8]) -> bool {
    const PAD: &[u8] = b"bullet";
    padding
        .iter()
        .enumerate()
        .all(|(idx, byte)| *byte == PAD[idx % PAD.len()])
}

pub(super) fn parse_bullet_quant_header(path: &Path, bytes: &[u8]) -> Result<BulletQuantHeader, EngineError> {
    if bytes.len() < BULLET_QUANT_HEADER_LEN {
        return Err(invalid_eval_file(
            path,
            "bullet quantized file is too short for its header",
        ));
    }
    let version = read_u32(bytes, 8)?;
    if version != 1 {
        return Err(invalid_eval_file(
            path,
            "unsupported bullet quantized format version",
        ));
    }
    let input_features = read_u32(bytes, 12)? as usize;
    let hidden_size = read_u32(bytes, 16)? as usize;
    let output_scale = read_i32(bytes, 20)?;
    let flags = read_u32(bytes, 24)?;
    let qa = read_i16(bytes, 28)?;
    let qb = read_i16(bytes, 30)?;
    Ok(BulletQuantHeader {
        input_features,
        hidden_size,
        output_scale,
        flags,
        qa,
        qb,
    })
}

pub(super) fn validate_input_features(
    path: &Path,
    input_features: usize,
    flags: u32,
) -> Result<NnueArchitecture, EngineError> {
    const SUPPORTED_FLAGS: u32 = BULLET_FLAG_HAS_SIDE_TO_MOVE;
    if flags & !SUPPORTED_FLAGS != 0 {
        return Err(invalid_eval_file(
            path,
            "bullet quantized flags include unsupported score-base or layout bits",
        ));
    }
    if flags & BULLET_FLAG_HAS_SIDE_TO_MOVE != 0 {
        if input_features == SIDE_TO_MOVE_FEATURE + 1 {
            return Ok(NnueArchitecture::nightweave());
        }
    } else if input_features == KING_BUCKET_FEATURES {
        return Ok(NnueArchitecture::nightweave());
    } else if input_features == VEX_INPUT_FEATURES {
        return Ok(NnueArchitecture::vex());
    }
    Err(invalid_eval_file(
        path,
        "bullet quantized input feature count does not match a supported architecture",
    ))
}
