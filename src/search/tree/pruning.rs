
use crate::{
    Board, Move, Piece,
    evaluation::LOSS_SCORE,
    pieces::NON_PAWN_MATERIAL,
};

use super::{
    constants::*,
    context::SearchContext,
    negamax::negamax,
    root::{PvMove, SearchOutcome, search_child_with_pvs},
    scoring::piece_value,
    search_profile::SearchProfile,
};

pub(in crate::search) fn should_use_pvs(is_pv_node: bool, searched_moves: u32, alpha: i32, beta: i32) -> bool {
    is_pv_node && searched_moves > 0 && beta > alpha.saturating_add(1)
}

pub(in crate::search) struct ChildSearchParams<'a> {
    pub(in crate::search) board: &'a Board,
    pub(in crate::search) repetition: bool,
    pub(in crate::search) depth: u32,
    pub(in crate::search) parent_depth: u32,
    pub(in crate::search) root_depth: u32,
    pub(in crate::search) alpha: i32,
    pub(in crate::search) beta: i32,
    pub(in crate::search) child_pv: &'a [PvMove],
    pub(in crate::search) previous_move: Move,
    pub(in crate::search) searched_moves: u32,
    pub(in crate::search) is_pv_node: bool,
    pub(in crate::search) is_quiet: bool,
    pub(in crate::search) in_check: bool,
    pub(in crate::search) gives_check: bool,
    pub(in crate::search) move_score: i32,
    pub(in crate::search) allow_reduction: bool,
    pub(in crate::search) search_profile: SearchProfile,
    pub(in crate::search) ply: u16,
}

#[inline]
pub(in crate::search) fn search_child_with_lmr(
    params: ChildSearchParams<'_>,
    context: &mut SearchContext<'_>,
) -> Option<SearchOutcome> {
    let ChildSearchParams {
        board,
        repetition,
        depth,
        parent_depth,
        root_depth,
        alpha,
        beta,
        child_pv,
        previous_move,
        searched_moves,
        is_pv_node,
        is_quiet,
        in_check,
        gives_check,
        move_score,
        allow_reduction,
        search_profile,
        ply,
    } = params;

    let reduction = if allow_reduction {
        late_move_reduction(
            parent_depth,
            searched_moves,
            is_pv_node,
            is_quiet,
            in_check,
            gives_check,
            move_score,
            search_profile,
        )
    } else {
        0
    };
    if reduction == 0 {
        return search_child_with_pvs(
            board,
            repetition,
            depth,
            root_depth,
            alpha,
            beta,
            child_pv,
            previous_move,
            context,
            ply,
            should_use_pvs(is_pv_node, searched_moves, alpha, beta),
        );
    }

    let reduced_depth = depth.saturating_sub(reduction);
    let scout_beta = alpha.saturating_neg();
    let scout_alpha = scout_beta.saturating_sub(1);
    let reduced = negamax(
        board,
        repetition,
        reduced_depth,
        root_depth,
        scout_alpha,
        scout_beta,
        &[],
        Some(previous_move),
        context,
        ply,
        true,
    )?;
    let reduced_score = -reduced.score;
    if reduced_score <= alpha {
        return Some(reduced);
    }

    search_child_with_pvs(
        board,
        repetition,
        depth,
        root_depth,
        alpha,
        beta,
        child_pv,
        previous_move,
        context,
        ply,
        false,
    )
}

#[inline]
pub(in crate::search) fn late_move_reduction(
    depth: u32,
    searched_moves: u32,
    is_pv_node: bool,
    is_quiet: bool,
    in_check: bool,
    gives_check: bool,
    move_score: i32,
    search_profile: SearchProfile,
) -> u32 {
    if depth < LMR_MIN_DEPTH
        || searched_moves < LMR_MIN_MOVE_INDEX
        || !is_quiet
        || in_check
        || (gives_check && !search_profile.reduce_late_quiet_checks())
        || move_score >= COUNTER_MOVE_SCORE
    {
        return 0;
    }

    let move_number = searched_moves.saturating_add(1);
    let mut reduction: u32 = 1;
    if depth >= 5 {
        reduction += 1;
    }
    if depth >= 8 {
        reduction += 1;
    }
    if move_number >= 6 {
        reduction += 1;
    }
    if move_number >= 12 {
        reduction += 1;
    }
    if move_number >= 24 {
        reduction += 1;
    }
    if !is_pv_node && depth >= 4 && move_number >= 4 {
        reduction += 1;
    }
    if !is_pv_node && depth >= 10 && move_number >= 12 {
        reduction += 1;
    }

    if is_pv_node {
        reduction = reduction.saturating_sub(1);
    }
    if move_score > 0 {
        reduction = reduction.saturating_sub(1);
    }
    if gives_check {
        reduction = reduction.saturating_sub(SPARSE_ENDGAME_QUIET_CHECK_LMR_PROTECTION);
    }

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
        && depth >= NULL_MOVE_MIN_DEPTH
        && !is_pv_node
        && !in_check
        && side_has_non_pawn_material(board)
}

#[inline]
pub(in crate::search) fn null_move_reduction(
    depth: u32,
    static_eval: i32,
    beta: i32,
    search_profile: SearchProfile,
) -> u32 {
    let eval_margin = static_eval.saturating_sub(beta).max(0);
    let eval_reduction = (eval_margin / NULL_MOVE_EVAL_MARGIN_PER_REDUCTION) as u32;
    let mut reduction = NULL_MOVE_BASE_REDUCTION
        .saturating_add(depth / NULL_MOVE_DEPTH_REDUCTION_DIVISOR)
        .saturating_add(eval_reduction.min(NULL_MOVE_MAX_EVAL_REDUCTION));
    if search_profile.sparse_pawnless_endgame() {
        reduction = reduction.saturating_sub(NULL_MOVE_SPARSE_ENDGAME_REDUCTION_PROTECTION);
    }
    reduction.min(depth.saturating_sub(1))
}

#[inline]
pub(in crate::search) fn should_verify_null_move(
    depth: u32,
    search_profile: SearchProfile,
) -> bool {
    depth >= NULL_MOVE_VERIFICATION_MIN_DEPTH || search_profile.sparse_pawnless_endgame()
}

#[inline]
pub(in crate::search) fn side_has_non_pawn_material(board: &Board) -> bool {
    let side = board.side_to_move();
    let non_pawn_material = NON_PAWN_MATERIAL
        .into_iter()
        .fold(cozy_chess::BitBoard::EMPTY, |pieces, piece| pieces | board.pieces(piece));
    !(board.colors(side) & non_pawn_material).is_empty()
}

#[inline]
pub(in crate::search) fn can_use_static_eval(
    repetition: bool,
    in_check: bool,
    alpha: i32,
    beta: i32,
) -> bool {
    !repetition
        && !in_check
        && is_non_mate_search_window(alpha, beta)
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
pub(in crate::search) fn reverse_futility_margin(depth: u32) -> i32 {
    REVERSE_FUTILITY_BASE_MARGIN
        + REVERSE_FUTILITY_MARGIN_PER_DEPTH.saturating_mul(depth.min(32) as i32)
}

#[inline]
pub(in crate::search) fn razor_margin(depth: u32) -> i32 {
    RAZOR_BASE_MARGIN + RAZOR_MARGIN_PER_DEPTH.saturating_mul(depth.min(32) as i32)
}

#[inline]
pub(in crate::search) fn futility_margin(depth: u32, improving: bool) -> i32 {
    let base = FUTILITY_BASE_MARGIN
        + FUTILITY_MARGIN_PER_DEPTH.saturating_mul(depth.min(32) as i32);
    if improving {
        base.saturating_add(FUTILITY_IMPROVING_MARGIN)
    } else {
        base
    }
}

#[inline]
pub(in crate::search) fn see_pruning_margin(depth: u32) -> i32 {
    SEE_PRUNING_BASE_MARGIN + SEE_PRUNING_MARGIN_PER_DEPTH.saturating_mul(depth.min(32) as i32)
}

#[inline]
pub(in crate::search) fn should_reverse_futility_prune(
    depth: u32,
    static_eval: i32,
    beta: i32,
) -> Option<i32> {
    if depth > REVERSE_FUTILITY_MAX_DEPTH {
        return None;
    }
    let score = static_eval.saturating_sub(reverse_futility_margin(depth));
    (score >= beta).then_some(score)
}

#[inline]
pub(in crate::search) fn should_try_razoring(depth: u32, static_eval: i32, alpha: i32) -> bool {
    depth <= RAZOR_MAX_DEPTH && static_eval.saturating_add(razor_margin(depth)) <= alpha
}

#[inline]
pub(in crate::search) fn should_futility_prune_quiet(
    depth: u32,
    static_eval: i32,
    alpha: i32,
    quiet_score: i32,
    improving: bool,
) -> bool {
    depth <= FUTILITY_MAX_DEPTH
        && quiet_score < COUNTER_MOVE_SCORE
        && static_eval
            .saturating_add(futility_margin(depth, improving))
            <= alpha
}

#[inline]
pub(in crate::search) fn should_see_prune_capture(
    depth: u32,
    is_pv_node: bool,
    gives_check: bool,
    searched_moves: u32,
    see: i32,
) -> bool {
    depth <= SEE_PRUNING_MAX_DEPTH
        && !is_pv_node
        && !gives_check
        && searched_moves > 0
        && see < -see_pruning_margin(depth)
}

#[inline]
pub(in crate::search) fn is_see_prune_candidate(depth: u32, is_pv_node: bool, searched_moves: u32, see: i32) -> bool {
    depth <= SEE_PRUNING_MAX_DEPTH
        && !is_pv_node
        && searched_moves > 0
        && see < -see_pruning_margin(depth)
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
        .saturating_add(Q_DELTA_PRUNING_MARGIN)
        <= alpha
}

#[inline]
pub(in crate::search) fn late_quiet_pruning_threshold(depth: u32, improving: bool) -> u32 {
    let depth = depth.max(1).min(LATE_QUIET_PRUNING_MAX_DEPTH);
    let threshold = LATE_QUIET_PRUNING_BASE_THRESHOLD
        .saturating_add(depth.saturating_mul(depth));
    let threshold = if improving {
        threshold
    } else {
        threshold / 2
    };
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
    depth <= LATE_QUIET_PRUNING_MAX_DEPTH
        && quiet_score < COUNTER_MOVE_SCORE
        && searched_moves >= late_quiet_pruning_threshold(depth, improving)
}

#[inline]
pub(in crate::search) fn apply_mate_distance_pruning(alpha: &mut i32, beta: &mut i32, ply: u16) -> Option<i32> {
    let ply = ply as i32;
    let worst_score = LOSS_SCORE.saturating_add(ply);
    let best_score = (-LOSS_SCORE).saturating_sub(ply).saturating_sub(1);
    *alpha = (*alpha).max(worst_score);
    *beta = (*beta).min(best_score);
    (*alpha >= *beta).then_some(*alpha)
}
