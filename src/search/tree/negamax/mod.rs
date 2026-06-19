mod move_loop;
mod null_move;
mod static_eval;
mod tt;

use crate::{Board, Move};

use super::{
    context::SearchContext,
    position_key::position_key,
    pruning::{
        apply_mate_distance_pruning, internal_iterative_reduction, requires_full_mate_search,
    },
    quiescence::quiescence,
    root::{SearchOutcome, terminal_outcome},
    scoring::terminal_score,
};

use move_loop::{FinishNodeParams, MoveLoopParams, finish_node, search_move_loop};
use null_move::{NullMoveParams, try_null_move_pruning};
use static_eval::{
    StaticEvalParams, StaticPruningParams, prepare_static_eval, try_static_eval_pruning,
};
use tt::tt_cutoff;

enum PruneResult {
    Continue,
    Done(SearchOutcome),
    Interrupted,
}

pub(in crate::search) fn negamax(
    board: &Board,
    repetition: bool,
    depth: u32,
    root_depth: u32,
    mut alpha: i32,
    mut beta: i32,
    previous_pv: &[super::root::PvMove],
    previous_move: Option<Move>,
    context: &mut SearchContext<'_>,
    ply: u16,
    allow_null_move: bool,
    excluded_move: Option<Move>,
) -> Option<SearchOutcome> {
    context.clear_static_eval_at_ply(ply);
    if context.should_stop().is_some() {
        return None;
    }
    if let Some(score) = terminal_score(board, repetition, ply) {
        return Some(terminal_outcome(score, repetition));
    }
    if depth == 0 {
        let result = quiescence(
            board,
            repetition,
            alpha,
            beta,
            previous_move,
            previous_pv,
            context,
            ply,
        );
        return result;
    }

    let alpha_start = alpha;
    let is_pv_node = beta > alpha.saturating_add(1);
    let key = position_key(board);
    let use_tt = !repetition && excluded_move.is_none();
    let tt_entry = if use_tt {
        context.transposition_table().probe(key)
    } else {
        None
    };
    if let Some(outcome) = tt_cutoff(board, depth, alpha, beta, is_pv_node, tt_entry, context, ply) {
        return Some(outcome);
    }

    let in_check = !board.checkers().is_empty();
    let needs_full_mate_search = requires_full_mate_search(alpha, beta);
    let expected_cut_node = !is_pv_node && beta == alpha.saturating_add(1);
    let hash_move = tt_entry.and_then(|entry| entry.best_move);
    let iir = if excluded_move.is_none() {
        internal_iterative_reduction(
            depth,
            repetition,
            is_pv_node,
            expected_cut_node,
            in_check,
            needs_full_mate_search,
            hash_move.is_some(),
        )
    } else {
        0
    };
    let depth = depth.saturating_sub(iir);
    if let Some(score) = apply_mate_distance_pruning(&mut alpha, &mut beta, ply) {
        return Some(terminal_outcome(score, false));
    }
    let static_eval = prepare_static_eval(
        StaticEvalParams {
            board,
            repetition,
            in_check,
            is_pv_node,
            alpha,
            beta,
            previous_move,
            tt_entry,
            ply,
        },
        context,
    );
    match try_static_eval_pruning(
        StaticPruningParams {
            board,
            repetition,
            depth,
            alpha,
            beta,
            previous_move,
            ply,
        },
        static_eval,
        context,
    ) {
        PruneResult::Continue => {}
        PruneResult::Done(outcome) => return Some(outcome),
        PruneResult::Interrupted => return None,
    }

    match try_null_move_pruning(
        NullMoveParams {
            board,
            repetition,
            depth,
            root_depth,
            beta,
            is_pv_node,
            in_check,
            needs_full_mate_search,
            static_eval: static_eval.corrected,
            allow_null_move,
            ply,
        },
        context,
    ) {
        PruneResult::Continue => {}
        PruneResult::Done(outcome) => return Some(outcome),
        PruneResult::Interrupted => return None,
    }

    let loop_result = search_move_loop(
        MoveLoopParams {
            board,
            previous_pv,
            previous_move,
            repetition,
            depth,
            root_depth,
            alpha,
            beta,
            is_pv_node,
            in_check,
            needs_full_mate_search,
            static_eval,
            ply,
            tt_entry,
            excluded_move,
        },
        context,
    )?;
    finish_node(
        FinishNodeParams {
            board,
            previous_move,
            depth,
            alpha_start,
            beta,
            key,
            use_tt,
            raw_static_eval: static_eval.raw,
            ply,
        },
        loop_result,
        context,
    )
}
