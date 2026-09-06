use crate::{Board, Piece, evaluation::LOSS_SCORE};

use super::scoring::piece_value;
use crate::search::constants::*;

#[inline]
pub(in crate::search) fn internal_iterative_reduction(
    depth: u32,
    repetition: bool,
    is_pv_node: bool,
    expected_cut_node: bool,
    in_check: bool,
    needs_full_mate_search: bool,
    has_hash_move: bool,
) -> u32 {
    if repetition
        || in_check
        || needs_full_mate_search
        || has_hash_move
        || depth < INTERNAL_ITERATIVE_REDUCTION_MIN_DEPTH()
    {
        return 0;
    }

    if is_pv_node || expected_cut_node {
        INTERNAL_ITERATIVE_REDUCTION().min(depth.saturating_sub(1))
    } else {
        0
    }
}

#[inline]
pub(in crate::search) fn late_move_reduction(
    depth: u32,
    searched_moves: u32,
    is_pv_node: bool,
    is_quiet: bool,
    in_check: bool,
    gives_check: bool,
    improving: bool,
    history_adjustment: impl FnOnce() -> i32,
) -> u32 {
    let minimum_searched_moves = if is_pv_node { 3 } else { 1 };
    if depth < LMR_MIN_DEPTH()
        || searched_moves < minimum_searched_moves
        || !is_quiet
    {
        return 0;
    }

    let move_number = searched_moves.saturating_add(1);
    let mut reduction = LMR_BASE()
        + depth.ilog2() as i32 * move_number.ilog2() as i32 * LMR_DEPTH_MOVE_WEIGHT();
    if !is_pv_node {
        reduction += LMR_SCALE;
    }
    if !in_check && !improving {
        reduction += LMR_SCALE;
    }
    if in_check {
        reduction -= LMR_SCALE;
    }
    if gives_check {
        reduction -= SPARSE_ENDGAME_QUIET_CHECK_LMR_PROTECTION() as i32 * LMR_SCALE;
    }

    reduction -= history_adjustment();
    let reduction = ((reduction.max(0) + LMR_SCALE / 2) / LMR_SCALE) as u32;
    reduction.min(depth.saturating_sub(1))
}

#[inline]
pub(in crate::search) fn should_try_null_move(
    board: &Board,
    depth: u32,
    is_pv_node: bool,
    in_check: bool,
    allow_null_move: bool,
) -> bool {
    allow_null_move
        && depth >= NULL_MOVE_MIN_DEPTH()
        && !is_pv_node
        && !in_check
        && side_has_non_pawn_material(board)
}

#[inline]
pub(in crate::search) fn null_move_reduction(
    depth: u32,
    static_eval: i32,
    beta: i32,
    sparse_pawnless_endgame: bool,
    zugzwang_prone: bool,
) -> u32 {
    let eval_margin = static_eval.saturating_sub(beta).max(0);
    let eval_reduction = (eval_margin / NULL_MOVE_EVAL_MARGIN_PER_REDUCTION()) as u32;
    let mut reduction = NULL_MOVE_BASE_REDUCTION()
        .saturating_add(depth / NULL_MOVE_DEPTH_REDUCTION_DIVISOR())
        .saturating_add(eval_reduction.min(NULL_MOVE_MAX_EVAL_REDUCTION()));
    if sparse_pawnless_endgame || zugzwang_prone {
        reduction = reduction.saturating_sub(NULL_MOVE_SPARSE_ENDGAME_REDUCTION_PROTECTION());
    }
    reduction.min(depth.saturating_sub(1))
}

#[inline]
pub(in crate::search) fn should_verify_null_move(
    depth: u32,
    sparse_pawnless_endgame: bool,
    zugzwang_prone: bool,
) -> bool {
    depth >= NULL_MOVE_VERIFICATION_MIN_DEPTH() || sparse_pawnless_endgame || zugzwang_prone
}

pub(in crate::search) fn is_zugzwang_prone(board: &Board) -> bool {
    let side = crate::chess::side_to_move(board);
    let side_pieces = crate::chess::colors(board, side);
    let pawns = side_pieces & crate::chess::pieces(board, Piece::Pawn);
    let minors = side_pieces
        & (crate::chess::pieces(board, Piece::Knight)
            | crate::chess::pieces(board, Piece::Bishop));
    let heavy = side_pieces
        & (crate::chess::pieces(board, Piece::Rook) | crate::chess::pieces(board, Piece::Queen));
    !pawns.is_empty() && heavy.is_empty() && minors.len() <= 1
}

pub(in crate::search) fn is_sparse_pawnless_endgame(board: &Board) -> bool {
    crate::chess::pieces(board, Piece::Pawn).is_empty()
        && [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen]
            .into_iter()
            .map(|piece| crate::chess::pieces(board, piece).len())
            .sum::<u32>()
            <= SPARSE_ENDGAME_MAX_NON_KING_PIECES
}

#[inline]
pub(in crate::search) fn side_has_non_pawn_material(board: &Board) -> bool {
    let side = crate::chess::side_to_move(board);
    let non_pawn_material = crate::chess::pieces(board, Piece::Knight)
        | crate::chess::pieces(board, Piece::Bishop)
        | crate::chess::pieces(board, Piece::Rook)
        | crate::chess::pieces(board, Piece::Queen);
    !(crate::chess::colors(board, side) & non_pawn_material).is_empty()
}

#[inline]
pub(in crate::search) fn can_use_static_eval(
    repetition: bool,
    in_check: bool,
    alpha: i32,
    beta: i32,
) -> bool {
    !repetition && !in_check && is_non_mate_search_window(alpha, beta)
}

#[inline]
pub(in crate::search) fn can_use_static_eval_pruning(
    repetition: bool,
    is_pv_node: bool,
    in_check: bool,
    alpha: i32,
    beta: i32,
) -> bool {
    !is_pv_node && can_use_static_eval(repetition, in_check, alpha, beta)
}

#[inline]
pub(in crate::search) fn is_non_mate_search_window(alpha: i32, beta: i32) -> bool {
    alpha > LOSS_SCORE + MATE_PRUNING_GUARD && beta < -LOSS_SCORE - MATE_PRUNING_GUARD
}

#[inline]
pub(in crate::search) fn requires_full_mate_search(alpha: i32, beta: i32) -> bool {
    let bounded_alpha = alpha > i32::MIN / 2;
    let bounded_beta = beta < i32::MAX / 2;
    (bounded_alpha && alpha <= LOSS_SCORE + MATE_PRUNING_GUARD)
        || (bounded_beta && beta >= -LOSS_SCORE - MATE_PRUNING_GUARD)
}

#[inline]
pub(in crate::search) fn reverse_futility_margin(depth: u32, base_margin: i32) -> i32 {
    base_margin
        + REVERSE_FUTILITY_MARGIN_PER_DEPTH().saturating_mul(depth.min(32) as i32)
}

#[inline]
pub(in crate::search) fn razor_margin(depth: u32) -> i32 {
    RAZOR_BASE_MARGIN() + RAZOR_MARGIN_PER_DEPTH().saturating_mul(depth.min(32) as i32)
}

#[inline]
pub(in crate::search) fn futility_margin(depth: u32, improving: bool) -> i32 {
    let base =
        FUTILITY_BASE_MARGIN() + FUTILITY_MARGIN_PER_DEPTH().saturating_mul(depth.min(32) as i32);
    if improving {
        base.saturating_add(FUTILITY_IMPROVING_MARGIN())
    } else {
        base
    }
}

#[inline]
pub(in crate::search) fn see_pruning_margin(depth: u32) -> i32 {
    SEE_PRUNING_BASE_MARGIN() + SEE_PRUNING_MARGIN_PER_DEPTH().saturating_mul(depth.min(32) as i32)
}

#[inline]
pub(in crate::search) fn should_reverse_futility_prune(
    depth: u32,
    static_eval: i32,
    beta: i32,
    base_margin: i32,
) -> Option<i32> {
    if depth > REVERSE_FUTILITY_MAX_DEPTH() {
        return None;
    }
    let score = static_eval.saturating_sub(reverse_futility_margin(depth, base_margin));
    (score >= beta).then_some(score)
}

#[inline]
pub(in crate::search) fn should_try_razoring(depth: u32, static_eval: i32, alpha: i32) -> bool {
    depth <= RAZOR_MAX_DEPTH() && static_eval.saturating_add(razor_margin(depth)) <= alpha
}

#[inline]
pub(in crate::search) fn should_futility_prune_quiet(
    depth: u32,
    static_eval: i32,
    alpha: i32,
    quiet_score: i32,
    improving: bool,
) -> bool {
    depth <= FUTILITY_MAX_DEPTH()
        && quiet_score < COUNTER_MOVE_SCORE
        && static_eval.saturating_add(futility_margin(depth, improving)) <= alpha
}

#[inline]
pub(in crate::search) fn is_see_prune_candidate(
    depth: u32,
    is_pv_node: bool,
    searched_moves: u32,
    see: i32,
) -> bool {
    can_try_see_pruning(depth, is_pv_node, searched_moves) && see < -see_pruning_margin(depth)
}

#[inline]
pub(in crate::search) fn can_try_see_pruning(
    depth: u32,
    is_pv_node: bool,
    searched_moves: u32,
) -> bool {
    depth <= SEE_PRUNING_MAX_DEPTH() && !is_pv_node && searched_moves > 0
}

#[inline]
pub(in crate::search) fn should_q_delta_prune_capture(
    stand_pat: i32,
    alpha: i32,
    captured_piece: Piece,
    promotion: Option<Piece>,
    moving_piece: Piece,
) -> bool {
    let promotion_gain = promotion
        .map(|piece| piece_value(piece).saturating_sub(piece_value(moving_piece)))
        .unwrap_or(0)
        .max(0);
    stand_pat
        .saturating_add(piece_value(captured_piece))
        .saturating_add(promotion_gain)
        .saturating_add(Q_DELTA_PRUNING_MARGIN())
        <= alpha
}

#[inline]
pub(in crate::search) fn late_quiet_pruning_threshold(depth: u32, improving: bool) -> u32 {
    let depth = depth.max(1).min(LATE_QUIET_PRUNING_MAX_DEPTH());
    let threshold = LATE_QUIET_PRUNING_BASE_THRESHOLD().saturating_add(depth.saturating_mul(depth));
    let threshold = if improving { threshold } else { threshold / 2 };
    let shallow_floor = match depth {
        1 => 5,
        2 => 8,
        3 => 10,
        _ => 0,
    };
    threshold.max(shallow_floor)
}

#[inline]
pub(in crate::search) fn should_prune_late_quiet(
    depth: u32,
    searched_moves: u32,
    quiet_score: i32,
    improving: bool,
) -> bool {
    depth <= LATE_QUIET_PRUNING_MAX_DEPTH()
        && quiet_score < COUNTER_MOVE_SCORE
        && searched_moves >= late_quiet_pruning_threshold(depth, improving)
}

#[inline]
pub(in crate::search) fn apply_mate_distance_pruning(
    alpha: &mut i32,
    beta: &mut i32,
    ply: u16,
) -> Option<i32> {
    let ply = ply as i32;
    let worst_score = LOSS_SCORE.saturating_add(ply);
    let best_score = (-LOSS_SCORE).saturating_sub(ply).saturating_sub(1);
    *alpha = (*alpha).max(worst_score);
    *beta = (*beta).min(best_score);
    (*alpha >= *beta).then_some(*alpha)
}
