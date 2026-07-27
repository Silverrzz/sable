use std::path::Path;

use crate::{Board, Color, EngineError, Move, Piece, Square};

use super::types::*;

pub(super) fn validate_i16_accumulator_range(
    path: &Path,
    bias: &[i16],
    feature_weights: &[i16],
    hidden_size: usize,
) -> Result<(), EngineError> {
    if bias.len() != hidden_size
        || SHARD_INPUT_FEATURES
            .checked_mul(hidden_size)
            .is_none_or(|expected| feature_weights.len() != expected)
    {
        return Err(invalid_eval_file(
            path,
            "shard first-layer dimensions do not match the declared hidden width",
        ));
    }

    for neuron in 0..hidden_size {
        let bias_abs = i64::from(i32::from(bias[neuron]).abs());
        for king_bucket in 0..SHARD_KING_BUCKETS {
            let mut top = [0_i32; 32];
            let bucket_start = king_bucket * PIECE_SQUARE_FEATURES;
            for piece_feature in 0..PIECE_SQUARE_FEATURES {
                let feature = bucket_start + piece_feature;
                let magnitude =
                    i32::from(feature_weights[feature * hidden_size + neuron]).abs();
                insert_top_magnitude(&mut top, magnitude);
            }

            let piece_sum = top.iter().map(|value| i64::from(*value)).sum::<i64>();
            if bias_abs.saturating_add(piece_sum) > i64::from(i16::MAX) {
                return Err(invalid_eval_file(
                    path,
                    "shard first layer can overflow i16 accumulators",
                ));
            }
        }
    }

    Ok(())
}

fn invalid_eval_file(path: &Path, message: &str) -> EngineError {
    EngineError::InvalidEvalFile {
        path: path.display().to_string(),
        message: message.to_owned(),
    }
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
    feature_weights: &[i16],
    feature_index: usize,
    sign: i32,
) {
    let hidden_size = accumulator.len();
    let start = feature_index * hidden_size;
    let end = start + hidden_size;
    crate::simd::apply_feature_delta(accumulator, &feature_weights[start..end], sign);
}

pub(super) fn apply_feature_deltas(
    source: &[i16],
    target: &mut [i16],
    feature_weights: &[i16],
    updates: &FeatureUpdateList,
) {
    let hidden_size = source.len();
    debug_assert_eq!(target.len(), hidden_size);
    if updates.len == 2
        && updates.updates[0].sign == -1
        && updates.updates[1].sign == 1
    {
        crate::simd::copy_feature_delta_pair(
            source,
            target,
            feature_weights,
            hidden_size,
            updates.updates[0].feature,
            updates.updates[1].feature,
        );
        return;
    }
    if updates.len == 3
        && updates.updates[0].sign == -1
        && updates.updates[1].sign == -1
        && updates.updates[2].sign == 1
    {
        crate::simd::copy_feature_delta_triplet(
            source,
            target,
            feature_weights,
            hidden_size,
            updates.updates[0].feature,
            updates.updates[1].feature,
            updates.updates[2].feature,
        );
        return;
    }

    let mut features = [0_usize; MAX_MOVE_FEATURE_UPDATES];
    let mut signs = [0_i32; MAX_MOVE_FEATURE_UPDATES];
    let mut len = 0_usize;
    for update in updates.iter() {
        features[len] = update.feature;
        signs[len] = update.sign;
        len += 1;
    }
    target.copy_from_slice(source);
    apply_feature_delta_batch(
        target,
        feature_weights,
        &features[..len],
        &signs[..len],
    );
}

pub(super) fn apply_feature_delta_batch(
    accumulator: &mut [i16],
    feature_weights: &[i16],
    features: &[usize],
    signs: &[i32],
) {
    let hidden_size = accumulator.len();
    crate::simd::apply_feature_deltas(
        accumulator,
        feature_weights,
        hidden_size,
        features,
        signs,
    );
}

pub(super) fn collect_move_feature_updates(
    before: &Board,
    mv: Move,
    side: Color,
    moving_piece: Piece,
    captured: Option<(Piece, Square)>,
    is_castle: bool,
    perspective: Color,
) -> Option<FeatureUpdateList> {
    if moving_piece == Piece::King && (side == perspective || is_castle) {
        return None;
    }
    let king_square = oriented_king_square(before, perspective)?;
    let mut updates = FeatureUpdateList::new();
    updates.push(feature_update(
        king_square,
        perspective,
        side,
        moving_piece,
        mv.from,
        -1,
    ))?;

    if let Some((captured_piece, captured_square)) = captured {
        updates.push(feature_update(
            king_square,
            perspective,
            !side,
            captured_piece,
            captured_square,
            -1,
        ))?;
    }

    updates.push(feature_update(
        king_square,
        perspective,
        side,
        mv.promotion.unwrap_or(moving_piece),
        mv.to,
        1,
    ))?;
    Some(updates)
}

#[inline(always)]
pub(super) fn feature_update(
    king_square: usize,
    perspective: Color,
    piece_color: Color,
    piece: Piece,
    square: Square,
    sign: i32,
) -> FeatureUpdate {
    FeatureUpdate {
        feature: feature_index_for_perspective(
            perspective,
            king_square,
            piece_color,
            piece,
            square as usize,
        ),
        sign,
    }
}

#[inline(always)]
pub(super) fn oriented_king_square(board: &Board, perspective: Color) -> Option<usize> {
    let king_square = crate::chess::colored_pieces(board, perspective, Piece::King)
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
    let mirrored_square = if king_square % 8 > 3 {
        oriented_square ^ 7
    } else {
        oriented_square
    };
    let color_offset = if piece_color == perspective { 0 } else { 384 };
    let piece_square_feature = color_offset + piece_plane_offset(piece) + mirrored_square;
    king_bucket_index(king_square) * PIECE_SQUARE_FEATURES + piece_square_feature
}

#[inline(always)]
fn king_bucket_index(king_square: usize) -> usize {
    let rank = king_square / 8;
    let file = king_square % 8;
    let mirrored_file = if file > 3 { 7 - file } else { file };
    SHARD_BUCKET_LAYOUT[rank * 4 + mirrored_file]
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
