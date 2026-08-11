use crate::{Board, Move};

use super::super::{
    constants::*,
    moves::{move_generation::ordered_root_moves, move_ordering::ScoredMove},
    state::{
        context::SearchContext, correction_history::CorrectionContext, position_key::position_key,
    },
    tree::{negamax::negamax, pruning::late_move_reduction},
};
use super::outcome::{SearchOutcome, debug_validate_pv, is_better_root_score, parent_outcome};

pub(in crate::search) fn search_root_iteration(
    board: &Board,
    candidate_moves: &[Move],
    depth: u32,
    previous_score: i32,
    previous_pv: &[Move],
    completed_depth: u32,
    context: &mut SearchContext<'_>,
) -> Option<(Move, SearchOutcome)> {
    let use_aspiration =
        completed_depth > 0 && depth >= ASPIRATION_MIN_DEPTH() && !previous_pv.is_empty();
    let mut window = ASPIRATION_INITIAL_WINDOW();
    let mut alpha = if use_aspiration {
        previous_score.saturating_sub(window).max(i32::MIN + 1)
    } else {
        i32::MIN + 1
    };
    let mut beta = if use_aspiration {
        previous_score.saturating_add(window)
    } else {
        i32::MAX
    };

    loop {
        context.refresh_static_eval_at_ply(board, CorrectionContext::default(), 0);
        let root_repetitions = context.actual_game_repetition_count(board);
        let mut best_move = None;
        let mut best_outcome = SearchOutcome {
            score: i32::MIN,
            repetition_draw: false,
            pv: Vec::new(),
        };
        let mut search_alpha = alpha;
        let pv_move = previous_pv.last().copied();
        let moves = ordered_root_moves(board, candidate_moves, pv_move, context.ordering());
        let is_pv_node = beta > search_alpha.saturating_add(1);

        for (searched_moves, ordered) in moves.into_iter().enumerate() {
            let child = search_root_child(
                board,
                ordered,
                depth,
                previous_pv,
                pv_move,
                search_alpha,
                beta,
                searched_moves as u32,
                is_pv_node,
                context,
            )?;
            let score = -child.score;
            if is_better_root_score(
                score,
                child.repetition_draw,
                &best_outcome,
                root_repetitions,
            ) {
                best_outcome = parent_outcome(ordered.mv, child);
                best_move = Some(ordered.mv);
            }
            search_alpha = search_alpha.max(best_outcome.score);
            if search_alpha >= beta {
                break;
            }
        }
        debug_validate_pv(board, &best_outcome.pv, "ROOTDEPTH");
        let result = (best_move?, best_outcome);
        if !use_aspiration {
            return Some(result);
        }
        let score = result.1.score;
        if score <= alpha {
            if alpha == i32::MIN + 1 {
                return Some(result);
            }
            window = (window.saturating_mul(2)).min(ASPIRATION_MAX_WINDOW);
            alpha = score.saturating_sub(window).max(i32::MIN + 1);
            continue;
        }
        if score >= beta {
            if beta == i32::MAX {
                return Some(result);
            }
            window = (window.saturating_mul(2)).min(ASPIRATION_MAX_WINDOW);
            beta = score.saturating_add(window);
            continue;
        }
        return Some(result);
    }
}

pub(in crate::search) fn search_root_child(
    board: &Board,
    ordered: ScoredMove,
    depth: u32,
    previous_pv: &[Move],
    pv_move: Option<Move>,
    alpha: i32,
    beta: i32,
    searched_moves: u32,
    is_pv_node: bool,
    context: &mut SearchContext<'_>,
) -> Option<SearchOutcome> {
    if context.should_stop() {
        return None;
    }
    let root_key = position_key(board);
    context.push_repetition_key(root_key);
    let mut next = board.clone();
    crate::chess::play_generated_move_unchecked(
        &mut next,
        ordered.mv,
        ordered.moving_piece,
        ordered.captured_piece,
    );
    let next_key = position_key(&next);
    context.transposition_table().prefetch(next_key);
    let next_repetition = context.push_position(&next, next_key);
    context.push_eval_state(
        &next,
        ordered.mv,
        ordered.moving_piece,
        ordered.captured_piece,
    );
    let child_pv = if Some(ordered.mv) == pv_move && !previous_pv.is_empty() {
        &previous_pv[..previous_pv.len() - 1]
    } else {
        &[]
    };
    let child_depth = depth - 1;
    let correction_context =
        CorrectionContext::default().after_move(ordered.mv, ordered.moving_piece);
    let use_pvs = is_pv_node && searched_moves > 0 && beta > alpha.saturating_add(1);
    let child = if use_pvs {
        let scout_beta = alpha.saturating_neg();
        let scout_alpha = scout_beta.saturating_sub(1);
        let reduction = late_move_reduction(
            depth,
            searched_moves,
            true,
            ordered.is_quiet,
            !crate::chess::checkers(board).is_empty(),
            !crate::chess::checkers(&next).is_empty(),
            true,
            ordered.score,
        );
        let mut scout = negamax(
            &next,
            next_repetition,
            child_depth.saturating_sub(reduction),
            scout_alpha,
            scout_beta,
            &[],
            Some(ordered.mv),
            correction_context,
            context,
            1,
            true,
            None,
        );
        if reduction > 0 && scout.as_ref().is_some_and(|outcome| -outcome.score > alpha) {
            scout = negamax(
                &next,
                next_repetition,
                child_depth,
                scout_alpha,
                scout_beta,
                &[],
                Some(ordered.mv),
                correction_context,
                context,
                1,
                true,
                None,
            );
        }
        match scout {
            Some(outcome) if -outcome.score > alpha && -outcome.score < beta => negamax(
                &next,
                next_repetition,
                child_depth,
                beta.saturating_neg(),
                alpha.saturating_neg(),
                child_pv,
                Some(ordered.mv),
                correction_context,
                context,
                1,
                true,
                None,
            ),
            outcome => outcome,
        }
    } else {
        negamax(
            &next,
            next_repetition,
            child_depth,
            beta.saturating_neg(),
            alpha.saturating_neg(),
            child_pv,
            Some(ordered.mv),
            correction_context,
            context,
            1,
            true,
            None,
        )
    };
    context.pop_eval_state();
    context.pop_position(next_key);
    context.pop_position(root_key);
    child
}
