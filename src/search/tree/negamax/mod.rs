mod move_loop;
mod null_move;
mod static_eval;
mod tt;

use crate::{Board, Move};

use super::{
    constants::*,
    context::SearchContext,
    correction_history::CorrectionContext,
    move_generation::{MoveFilter, collect_moves_into},
    move_ordering::MovePicker,
    position_key::position_key,
    pruning::{
        apply_mate_distance_pruning, internal_iterative_reduction, requires_full_mate_search,
    },
    quiescence::quiescence,
    root::{SearchOutcome, terminal_outcome},
    scoring::terminal_score,
    see::static_exchange_eval_for_move,
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
    correction_context: CorrectionContext,
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
            correction_context,
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

    let in_check = !crate::chess::checkers(board).is_empty();
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
            correction_context,
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
            correction_context,
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
            correction_context,
            ply,
        },
        context,
    ) {
        PruneResult::Continue => {}
        PruneResult::Done(outcome) => return Some(outcome),
        PruneResult::Interrupted => return None,
    }

    match try_probcut(
        ProbCutParams {
            board,
            repetition,
            depth,
            root_depth,
            beta,
            is_pv_node,
            in_check,
            needs_full_mate_search,
            previous_move,
            correction_context,
            ply,
            tt_entry,
            excluded_move,
        },
        static_eval,
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
            correction_context,
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
            depth,
            alpha_start,
            beta,
            key,
            use_tt,
            raw_static_eval: static_eval.raw,
            corrected_static_eval: static_eval.corrected,
            correction_context,
            ply,
        },
        loop_result,
        context,
    )
}

struct ProbCutParams<'a> {
    board: &'a Board,
    repetition: bool,
    depth: u32,
    root_depth: u32,
    beta: i32,
    is_pv_node: bool,
    in_check: bool,
    needs_full_mate_search: bool,
    previous_move: Option<Move>,
    correction_context: CorrectionContext,
    ply: u16,
    tt_entry: Option<super::transposition::TranspositionEntry>,
    excluded_move: Option<Move>,
}

fn try_probcut(
    params: ProbCutParams<'_>,
    static_eval: static_eval::StaticEvalState,
    context: &mut SearchContext<'_>,
) -> PruneResult {
    if params.depth < PROBCUT_MIN_DEPTH
        || params.is_pv_node
        || params.in_check
        || params.needs_full_mate_search
        || params.repetition
        || params.excluded_move.is_some()
        || static_eval.corrected.is_none()
    {
        return PruneResult::Continue;
    }

    let probcut_beta = params.beta.saturating_add(PROBCUT_MARGIN);
    let child_alpha = probcut_beta.saturating_neg();
    let child_beta = child_alpha.saturating_add(1);
    let probcut_depth = params.depth.saturating_sub(PROBCUT_DEPTH_REDUCTION).max(1);
    let tt_move = params.tt_entry.and_then(|entry| entry.best_move);
    let mut moves = MovePicker::new();
    collect_moves_into(
        params.board,
        MoveFilter::Tactical,
        tt_move,
        params.previous_move,
        params.ply,
        &mut moves,
    );

    while let Some(ordered) = moves.next(params.board, context.ordering()) {
        if context.should_stop().is_some() {
            return PruneResult::Interrupted;
        }
        let see = ordered
            .see
            .unwrap_or_else(|| {
                static_exchange_eval_for_move(
                    params.board,
                    ordered.mv,
                    ordered.moving_piece,
                    ordered.captured_piece,
                )
            });
        if see < PROBCUT_SEE_THRESHOLD {
            continue;
        }

        let mut next = params.board.clone();
        crate::chess::play_unchecked(&mut next, ordered.mv);
        let next_key = position_key(&next);
        let next_repetition = context.push_position(&next, next_key);
        context.push_eval_state(params.board, &next, ordered.mv);
        let child_correction_context =
            params.correction_context.after_move(ordered.mv, ordered.moving_piece);

        let qsearch = quiescence(
            &next,
            next_repetition,
            child_alpha,
            child_beta,
            Some(ordered.mv),
            child_correction_context,
            &[],
            context,
            params.ply + 1,
        );
        context.note_searched_move(params.ply + 1);
        let Some(qsearch) = qsearch else {
            context.pop_eval_state(params.board, ordered.mv);
            context.pop_position(next_key);
            return PruneResult::Interrupted;
        };
        if -qsearch.score < probcut_beta {
            context.pop_eval_state(params.board, ordered.mv);
            context.pop_position(next_key);
            continue;
        }

        let reduced = negamax(
            &next,
            next_repetition,
            probcut_depth,
            params.root_depth,
            child_alpha,
            child_beta,
            &[],
            Some(ordered.mv),
            child_correction_context,
            context,
            params.ply + 1,
            true,
            None,
        );
        let Some(reduced) = reduced else {
            context.pop_eval_state(params.board, ordered.mv);
            context.pop_position(next_key);
            return PruneResult::Interrupted;
        };
        context.pop_eval_state(params.board, ordered.mv);
        context.pop_position(next_key);

        let score = -reduced.score;
        if score >= probcut_beta {
            return PruneResult::Done(terminal_outcome(score, false));
        }
    }

    PruneResult::Continue
}
