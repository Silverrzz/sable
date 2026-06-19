use crate::{Board, Color, Move};

use super::{
    negamax,
    static_eval::StaticEvalState,
    super::{
        constants::*,
        context::SearchContext,
        correction_history::should_update_correction_history,
        move_generation::{MoveFilter, collect_moves, priority_move_for_node},
        move_ordering::{MovePicker, ScoredMove},
        position_key::position_key,
        pruning::{
            ChildSearchParams, is_see_prune_candidate, search_child_with_lmr,
            should_futility_prune_quiet, should_prune_late_quiet, should_see_prune_capture,
        },
        root::{PvMove, SearchOutcome, is_better_score, parent_outcome, terminal_outcome},
        search_profile::SearchProfile,
        see::{move_gives_check, static_exchange_eval},
        transposition::{Bound, TranspositionEntry, is_mate_score, score_from_tt},
    },
};

pub(super) struct MoveLoopParams<'a> {
    pub(super) board: &'a Board,
    pub(super) previous_pv: &'a [PvMove],
    pub(super) previous_move: Option<Move>,
    pub(super) repetition: bool,
    pub(super) depth: u32,
    pub(super) root_depth: u32,
    pub(super) alpha: i32,
    pub(super) beta: i32,
    pub(super) is_pv_node: bool,
    pub(super) in_check: bool,
    pub(super) needs_full_mate_search: bool,
    pub(super) static_eval: StaticEvalState,
    pub(super) ply: u16,
    pub(super) tt_entry: Option<TranspositionEntry>,
    pub(super) excluded_move: Option<Move>,
}

pub(super) struct MoveLoopResult {
    best: SearchOutcome,
}

pub(super) fn search_move_loop(
    params: MoveLoopParams<'_>,
    context: &mut SearchContext<'_>,
) -> Option<MoveLoopResult> {
    let MoveLoopParams {
        board,
        previous_pv,
        previous_move,
        repetition,
        depth,
        root_depth,
        mut alpha,
        beta,
        is_pv_node,
        in_check,
        needs_full_mate_search,
        static_eval,
        ply,
        tt_entry,
        excluded_move,
    } = params;

    let side = board.side_to_move();
    let search_profile = SearchProfile::for_board(board);
    let pv_move = previous_pv.first().map(|pv| pv.mv);
    let tt_move = tt_entry.and_then(|entry| entry.best_move);
    let priority_move = priority_move_for_node(board, pv_move, tt_move, in_check);
    let mut moves = collect_moves(
        board,
        MoveFilter::All,
        priority_move,
        previous_move,
        ply,
    );
    let mut best = SearchOutcome {
        score: i32::MIN,
        repetition_draw: false,
        pv: Vec::new(),
    };
    let mut searched_moves = 0_u32;
    let mut captures_tried = 0_u32;
    let child_depth = if in_check && u32::from(ply) < root_depth.saturating_mul(2) {
        depth
    } else {
        depth - 1
    };
    while let Some(ordered) = moves.next(board, context.ordering()) {
        if context.should_stop().is_some() {
            return None;
        }
        if Some(ordered.mv) == excluded_move {
            continue;
        }
        let capture_prune = see_capture_prune(
            SeeCapturePruneParams {
                board,
                ordered,
                depth,
                is_pv_node,
                in_check,
                needs_full_mate_search,
                pv_move,
                captures_tried,
            },
        );
        if capture_prune.pruned {
            captures_tried += 1;
            continue;
        }
        let mut next = board.clone();
        next.play_unchecked(ordered.mv);
        let gives_check = capture_prune
            .gives_check
            .unwrap_or_else(|| !next.checkers().is_empty());
        if should_static_prune_quiet(
            static_eval,
            depth,
            alpha,
            ordered,
            searched_moves,
            gives_check,
        ) {
            continue;
        }
        let extension = singular_extension(
            SingularExtensionParams {
                board,
                repetition,
                previous_move,
                ordered,
                depth,
                root_depth,
                in_check,
                needs_full_mate_search,
                ply,
                tt_entry,
                tt_move,
                excluded_move,
            },
            context,
        )?;
        let next_key = position_key(&next);
        let next_repetition = context.push_position(&next, next_key);
        context.push_eval_state(board, &next, ordered.mv);
        let child_pv = if Some(ordered.mv) == pv_move {
            &previous_pv[1..]
        } else {
            &[]
        };
        let Some(child) = search_child_with_lmr(
            ChildSearchParams {
                board: &next,
                repetition: next_repetition,
                depth: child_depth.saturating_add(extension),
                parent_depth: depth,
                root_depth,
                alpha,
                beta,
                child_pv,
                previous_move: ordered.mv,
                searched_moves,
                is_pv_node,
                is_quiet: ordered.is_quiet,
                in_check,
                gives_check,
                move_score: ordered.score,
                allow_reduction: !needs_full_mate_search,
                search_profile,
                ply: ply + 1,
            },
            context,
        ) else {
            context.pop_eval_state(board, ordered.mv);
            context.pop_position(next_key);
            return None;
        };
        context.note_searched_move(ply + 1);
        context.pop_eval_state(board, ordered.mv);
        context.pop_position(next_key);
        if ordered.captured_piece.is_some() {
            captures_tried += 1;
        }
        searched_moves += 1;
        let child_score = -child.score;
        let raised_alpha = is_better_score(child_score, child.repetition_draw, &best);
        if raised_alpha {
            best = parent_outcome(board, ordered.mv, child, context.chess960());
        }
        alpha = alpha.max(best.score);
        let caused_cutoff = alpha >= beta;
        if caused_cutoff {
            record_cutoff_and_failures(&moves, ordered, side, previous_move, depth, ply, context);
            break;
        }
    }

    if excluded_move.is_some() && best.score == i32::MIN {
        best = terminal_outcome(alpha, false);
    }

    Some(MoveLoopResult { best })
}

struct SingularExtensionParams<'a> {
    board: &'a Board,
    repetition: bool,
    previous_move: Option<Move>,
    ordered: ScoredMove,
    depth: u32,
    root_depth: u32,
    in_check: bool,
    needs_full_mate_search: bool,
    ply: u16,
    tt_entry: Option<TranspositionEntry>,
    tt_move: Option<Move>,
    excluded_move: Option<Move>,
}

fn singular_extension(
    params: SingularExtensionParams<'_>,
    context: &mut SearchContext<'_>,
) -> Option<u32> {
    if params.excluded_move.is_some()
        || params.in_check
        || params.needs_full_mate_search
        || params.depth < SINGULAR_EXTENSION_MIN_DEPTH
        || Some(params.ordered.mv) != params.tt_move
    {
        return Some(0);
    }
    let Some(entry) = params.tt_entry else {
        return Some(0);
    };
    if !matches!(entry.bound, Bound::Lower | Bound::Exact)
        || u32::from(entry.depth).saturating_add(SINGULAR_EXTENSION_TT_DEPTH_MARGIN) < params.depth
    {
        return Some(0);
    }
    let tt_score = score_from_tt(entry.score, params.ply);
    if is_mate_score(tt_score) {
        return Some(0);
    }

    let singular_beta = tt_score.saturating_sub(singular_extension_margin(params.depth));
    let excluded = negamax(
        params.board,
        params.repetition,
        singular_extension_search_depth(params.depth),
        params.root_depth,
        singular_beta.saturating_sub(1),
        singular_beta,
        &[],
        params.previous_move,
        context,
        params.ply,
        false,
        Some(params.ordered.mv),
    )?;

    if excluded.score < singular_beta {
        Some(1)
    } else {
        Some(0)
    }
}

#[inline]
fn singular_extension_margin(depth: u32) -> i32 {
    SINGULAR_EXTENSION_BASE_MARGIN
        + SINGULAR_EXTENSION_MARGIN_PER_DEPTH.saturating_mul(depth.min(32) as i32)
}

#[inline]
fn singular_extension_search_depth(depth: u32) -> u32 {
    depth.saturating_sub(1) / 2
}

struct SeeCapturePruneParams<'a> {
    board: &'a Board,
    ordered: ScoredMove,
    depth: u32,
    is_pv_node: bool,
    in_check: bool,
    needs_full_mate_search: bool,
    pv_move: Option<Move>,
    captures_tried: u32,
}

struct SeeCapturePruneResult {
    gives_check: Option<bool>,
    pruned: bool,
}

fn see_capture_prune(params: SeeCapturePruneParams<'_>) -> SeeCapturePruneResult {
    if params.in_check || params.needs_full_mate_search || Some(params.ordered.mv) == params.pv_move
    {
        return SeeCapturePruneResult {
            gives_check: None,
            pruned: false,
        };
    }
    let Some(captured_piece) = params.ordered.captured_piece else {
        return SeeCapturePruneResult {
            gives_check: None,
            pruned: false,
        };
    };
    let see = params
        .ordered
        .see
        .unwrap_or_else(|| static_exchange_eval(params.board, params.ordered.mv));
    if !is_see_prune_candidate(params.depth, params.is_pv_node, params.captures_tried, see) {
        return SeeCapturePruneResult {
            gives_check: None,
            pruned: false,
        };
    }
    let gives_check = move_gives_check(
        params.board,
        params.ordered.mv,
        params.ordered.moving_piece,
        Some(captured_piece),
    );
    SeeCapturePruneResult {
        gives_check: Some(gives_check),
        pruned: should_see_prune_capture(
            params.depth,
            params.is_pv_node,
            gives_check,
            params.captures_tried,
            see,
        ),
    }
}

fn should_static_prune_quiet(
    static_eval: StaticEvalState,
    depth: u32,
    alpha: i32,
    ordered: ScoredMove,
    searched_moves: u32,
    gives_check: bool,
) -> bool {
    if !static_eval.can_prune || !ordered.is_quiet || searched_moves == 0 || gives_check {
        return false;
    }
    let Some(eval) = static_eval.corrected else {
        return false;
    };
    should_futility_prune_quiet(depth, eval, alpha, ordered.score, static_eval.improving)
        || should_prune_late_quiet(depth, searched_moves, ordered.score, static_eval.improving)
}

fn record_cutoff_and_failures(
    moves: &MovePicker,
    ordered: ScoredMove,
    side: Color,
    previous_move: Option<Move>,
    depth: u32,
    ply: u16,
    context: &mut SearchContext<'_>,
) {
    if ordered.is_quiet {
        context
            .ordering_mut()
            .record_quiet_cutoff(side, ordered.mv, previous_move, depth, ply);
    } else if let Some(captured_piece) = ordered.captured_piece {
        context.ordering_mut().record_capture_cutoff(
            side,
            ordered.moving_piece,
            ordered.mv,
            Some(captured_piece),
            depth,
        );
    }

    for index in 0..moves.len {
        let candidate = moves.get(index);
        if !candidate.tried || candidate.mv == ordered.mv {
            continue;
        }
        if candidate.is_quiet() {
            context.ordering_mut().record_quiet_failure(
                side,
                previous_move,
                candidate.mv,
                depth,
            );
        } else if let Some(captured_piece) = candidate.captured_piece {
            context.ordering_mut().record_capture_failure(
                side,
                candidate.mv,
                candidate.moving_piece,
                captured_piece,
                depth,
            );
        }
    }
}

pub(super) struct FinishNodeParams<'a> {
    pub(super) board: &'a Board,
    pub(super) previous_move: Option<Move>,
    pub(super) depth: u32,
    pub(super) alpha_start: i32,
    pub(super) beta: i32,
    pub(super) key: u64,
    pub(super) use_tt: bool,
    pub(super) raw_static_eval: Option<i32>,
    pub(super) ply: u16,
}

pub(super) fn finish_node(
    params: FinishNodeParams<'_>,
    result: MoveLoopResult,
    context: &mut SearchContext<'_>,
) -> Option<SearchOutcome> {
    let bound = if result.best.score <= params.alpha_start {
        Bound::Upper
    } else if result.best.score >= params.beta {
        Bound::Lower
    } else {
        Bound::Exact
    };
    let best_move = result.best.pv.first().map(|pv| pv.mv);
    if params.use_tt
        && !result.best.repetition_draw
        && let Some(raw_eval) = params.raw_static_eval
        && should_update_correction_history(
            params.board,
            best_move,
            bound,
            raw_eval,
            result.best.score,
        )
    {
        context.update_correction_history(
            params.board,
            params.previous_move,
            raw_eval,
            result.best.score,
            params.depth,
        );
    }
    if params.use_tt && !result.best.repetition_draw {
        context.transposition_table().store(
            params.key,
            params.depth,
            result.best.score,
            bound,
            best_move,
            params.raw_static_eval,
            params.ply,
        );
    }
    Some(result.best)
}
