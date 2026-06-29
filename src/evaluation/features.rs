use std::path::Path;

use crate::{Board, Color, EngineError, Move, Piece, Square, chess::{Rank}};

use super::{io::invalid_eval_file, types::*};

pub(super) fn build_first_layer_feature_weights(
    path: &Path,
    values: &[i16],
    hidden: usize,
    input_features: usize,
) -> Result<Vec<i16>, EngineError> {
    if hidden == 0 || values.len() != hidden * input_features {
        return Err(invalid_eval_file(
            path,
            "bullet quantized first layer shape does not match the declared network layout",
        ));
    }
    let mut by_feature = vec![0; input_features * hidden];
    for feature in 0..input_features {
        for neuron in 0..hidden {
            by_feature[(feature * hidden) + neuron] =
                values[(neuron * input_features) + feature];
        }
    }
    Ok(by_feature)
}

pub(super) fn validate_i16_accumulator_range(
    path: &Path,
    architecture: NnueArchitecture,
    first_layer: &QuantizedLayer,
    feature_weights: &[i16],
    hidden: usize,
    has_side_to_move_feature: bool,
) -> Result<(), EngineError> {
    let required_features = if has_side_to_move_feature {
        architecture.side_to_move_feature_index() + 1
    } else {
        architecture.input_features
    };
    if first_layer.bias.len() != hidden || feature_weights.len() < required_features * hidden {
        return Err(invalid_eval_file(
            path,
            "bullet quantized first layer shape does not match i16 accumulator validation",
        ));
    }

    for neuron in 0..hidden {
        let Some(bias_abs) = first_layer.bias[neuron].checked_abs() else {
            return Err(invalid_eval_file(
                path,
                "bullet quantized first layer bias is outside the i16 accumulator validation range",
            ));
        };
        let side_to_move_abs = if has_side_to_move_feature {
            i64::from(i32::from(
                feature_weights[architecture.side_to_move_feature_index() * hidden + neuron],
            ).abs())
        } else {
            0
        };

        for king_bucket in 0..architecture.bucket_count() {
            let mut top = [0_i32; 32];
            let bucket_start = king_bucket * PIECE_SQUARE_FEATURES;
            for piece_feature in 0..PIECE_SQUARE_FEATURES {
                let feature = bucket_start + piece_feature;
                let magnitude = i32::from(feature_weights[feature * hidden + neuron]).abs();
                insert_top_magnitude(&mut top, magnitude);
            }

            let piece_sum = top.iter().map(|value| i64::from(*value)).sum::<i64>();
            let bound = bias_abs
                .saturating_add(side_to_move_abs)
                .saturating_add(piece_sum);
            if bound > i64::from(i16::MAX) {
                return Err(invalid_eval_file(
                    path,
                    "bullet quantized first layer can overflow i16 accumulators",
                ));
            }
        }
    }

    Ok(())
}

fn insert_top_magnitude(top: &mut [i32; 32], magnitude: i32) {
    if magnitude <= top[0] {
        return;
    }
    top[0] = magnitude;
    let mut idx = 0;
    while idx + 1 < top.len() && top[idx] > top[idx + 1] {
        top.swap(idx, idx + 1);
        idx += 1;
    }
}

pub(super) fn apply_feature_delta(
    accumulator: &mut [i16],
    hidden_size: usize,
    feature_weights: &[i16],
    feature_index: usize,
    sign: i32,
) {
    let start = feature_index * hidden_size;
    let end = start + hidden_size;
    crate::simd::apply_feature_delta(accumulator, &feature_weights[start..end], sign);
}

pub(super) fn collect_move_feature_updates(
    before: &Board,
    mv: Move,
    architecture: NnueArchitecture,
    include_side_to_move: bool,
    perspective: Color,
) -> Option<FeatureUpdateList> {
    let side = crate::chess::side_to_move(before);
    let moving_piece = crate::chess::piece_on(before, mv.from)?;
    if moving_piece == Piece::King
        && (side == perspective || crate::chess::color_on(before, mv.to) == Some(side))
    {
        return None;
    }
    let king_square = oriented_king_square(before, perspective)?;
    let mut updates = FeatureUpdateList::new();
    updates.push(feature_update(
        king_square,
        architecture,
        perspective,
        side,
        moving_piece,
        mv.from,
        -1,
    ))?;

    if let Some((captured_piece, captured_square)) = captured_piece_for_move(before, mv, moving_piece) {
        updates.push(feature_update(
            king_square,
            architecture,
            perspective,
            !side,
            captured_piece,
            captured_square,
            -1,
        ))?;
    }

    updates.push(feature_update(
        king_square,
        architecture,
        perspective,
        side,
        mv.promotion.unwrap_or(moving_piece),
        mv.to,
        1,
    ))?;
    if include_side_to_move {
        let sign = if side == Color::White { 1 } else { -1 };
        updates.push(FeatureUpdate {
            feature: architecture.side_to_move_feature_index(),
            sign,
        })?;
    }
    Some(updates)
}

#[inline(always)]
pub(super) fn feature_update(
    king_square: usize,
    architecture: NnueArchitecture,
    perspective: Color,
    piece_color: Color,
    piece: Piece,
    square: Square,
    sign: i32,
) -> FeatureUpdate {
    FeatureUpdate {
        feature: feature_index_for_perspective(
            architecture,
            perspective,
            king_square,
            piece_color,
            piece,
            square as usize,
        ),
        sign,
    }
}

pub(super) fn captured_piece_for_move(
    before: &Board,
    mv: Move,
    moving_piece: Piece,
) -> Option<(Piece, Square)> {
    let side = crate::chess::side_to_move(before);
    let is_en_passant = moving_piece == Piece::Pawn
        && mv.from.file() != mv.to.file()
        && crate::chess::en_passant(before) == Some(mv.to.file())
        && crate::chess::piece_on(before, mv.to).is_none();
    if is_en_passant {
        return Some((
            Piece::Pawn,
            Square::new(mv.to.file(), Rank::Fifth.relative_to(side)),
        ));
    }
    let piece = crate::chess::piece_on(before, mv.to)?;
    if crate::chess::color_on(before, mv.to) == Some(!side) {
        Some((piece, mv.to))
    } else {
        None
    }
}

#[inline(always)]
pub(super) fn oriented_king_square(board: &Board, perspective: Color) -> Option<usize> {
    let king_square = (crate::chess::pieces(board, Piece::King) & crate::chess::colors(board, perspective))
        .into_iter()
        .next()? as usize;
    Some(if perspective == Color::White {
        king_square
    } else {
        king_square ^ 56
    })
}

#[inline(always)]
pub(super) fn feature_index_for_perspective(
    architecture: NnueArchitecture,
    perspective: Color,
    king_square: usize,
    piece_color: Color,
    piece: Piece,
    square_index: usize,
) -> usize {
    let oriented_square = if perspective == Color::White {
        square_index
    } else {
        square_index ^ 56
    };
    let mirrored_square = if matches!(
        architecture.feature_layout,
        NnueFeatureLayout::MirroredKingBuckets16
    ) && king_square % 8 > 3
    {
        oriented_square ^ 7
    } else {
        oriented_square
    };
    let color_offset = if piece_color == perspective { 0 } else { 384 };
    let piece_square_feature = color_offset + piece_plane_offset(piece) + mirrored_square;
    king_bucket_index(architecture, king_square) * PIECE_SQUARE_FEATURES + piece_square_feature
}

#[inline(always)]
fn king_bucket_index(architecture: NnueArchitecture, king_square: usize) -> usize {
    match architecture.feature_layout {
        NnueFeatureLayout::KingBuckets64 => king_square,
        NnueFeatureLayout::MirroredKingBuckets16 => {
            let rank = king_square / 8;
            let file = king_square % 8;
            let mirrored_file = if file > 3 { 7 - file } else { file };
            VEX_BUCKET_LAYOUT[rank * 4 + mirrored_file]
        }
    }
}

pub(super) fn piece_plane_offset(piece: Piece) -> usize {
    match piece {
        Piece::Pawn => 0,
        Piece::Knight => 64,
        Piece::Bishop => 128,
        Piece::Rook => 192,
        Piece::Queen => 256,
        Piece::King => 320,
    }
}
