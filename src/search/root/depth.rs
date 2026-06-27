use crate::{Board, Move};

use super::super::{
    constants::*,
    moves::{
        move_generation::ordered_root_moves,
        move_ordering::ScoredMove,
    },
    state::{
        correction_history::CorrectionContext,
        context::SearchContext,
        position_key::position_key,
    },
    tree::{negamax::negamax, pruning::should_use_pvs},
};
use super::{
    outcome::{
        SearchOutcome, debug_validate_pv, is_better_root_score, parent_outcome,
    },
    PvMove,
};

pub(in crate::search) fn search_root_depth(
    board: &Board,
    candidate_moves: &[Move],
    depth: u32,
    previous_pv: &[PvMove],
    alpha: i32,
    beta: i32,
    root_repetitions: u8,
    context: &mut SearchContext<'_>,
    chess960: bool,
) -> Option<(Move, SearchOutcome)> {
    let mut best_move = None;
    let mut best_outcome = SearchOutcome {
        score: i32::MIN,
        repetition_draw: false,
        pv: Vec::new(),
    };
    let mut alpha = alpha;
    let pv_move = previous_pv.last().map(|pv| pv.mv);
    let moves = ordered_root_moves(board, candidate_moves, pv_move, context.ordering());
    let is_pv_node = beta > alpha.saturating_add(1);
    let mut searched_moves = 0_u32;
    for ordered in moves {
        let Some(child) = search_root_child(
            board,
            ordered,
            depth,
            previous_pv,
            pv_move,
            alpha,
            beta,
            searched_moves,
            is_pv_node,
            context,
        ) else {
            return None;
        };
        searched_moves += 1;
        let score = -child.score;
        let raised_alpha =
            is_better_root_score(score, child.repetition_draw, &best_outcome, root_repetitions);
        if raised_alpha {
            best_outcome = parent_outcome(board, ordered.mv, child, chess960);
            best_move = Some(ordered.mv);
        }
        alpha = alpha.max(best_outcome.score);
        if alpha >= beta {
            break;
        }
    }
    debug_validate_pv(board, &best_outcome.pv, "ROOTDEPTH");
    best_move.map(|mv| (mv, best_outcome))
}

pub(in crate::search) fn search_root_depth_dispatch(
    board: &Board,
    candidate_moves: &[Move],
    depth: u32,
    previous_pv: &[PvMove],
    alpha: i32,
    beta: i32,
    context: &mut SearchContext<'_>,
    chess960: bool,
) -> Option<(Move, SearchOutcome)> {
    context.refresh_static_eval_at_ply(board, CorrectionContext::default(), 0);
    let root_repetitions = context.actual_game_repetition_count(board);
    search_root_depth(
        board,
        candidate_moves,
        depth,
        previous_pv,
        alpha,
        beta,
        root_repetitions,
        context,
        chess960,
    )
}

pub(in crate::search) fn search_root_ordered_move(
    board: &Board,
    ordered: ScoredMove,
    depth: u32,
    previous_pv: &[PvMove],
    pv_move: Option<Move>,
    alpha: i32,
    beta: i32,
    searched_moves: u32,
    is_pv_node: bool,
    context: &mut SearchContext<'_>,
    chess960: bool,
) -> Option<(Move, SearchOutcome)> {
    let child = search_root_child(
        board,
        ordered,
        depth,
        previous_pv,
        pv_move,
        alpha,
        beta,
        searched_moves,
        is_pv_node,
        context,
    )?;
    let outcome = parent_outcome(board, ordered.mv, child, chess960);
    debug_validate_pv(board, &outcome.pv, "ORDERED");
    Some((ordered.mv, outcome))
}

pub(in crate::search) fn search_root_iteration(
    board: &Board,
    candidate_moves: &[Move],
    depth: u32,
    previous_score: i32,
    previous_pv: &[PvMove],
    completed_depth: u32,
    context: &mut SearchContext<'_>,
    chess960: bool,
) -> Option<(Move, SearchOutcome)> {
    if completed_depth == 0 || depth < ASPIRATION_MIN_DEPTH || previous_pv.is_empty() {
        return search_root_depth_dispatch(
            board,
            candidate_moves,
            depth,
            previous_pv,
            i32::MIN + 1,
            i32::MAX,
            context,
            chess960,
        );
    }

    let mut window = ASPIRATION_INITIAL_WINDOW;
    let mut alpha = previous_score.saturating_sub(window).max(i32::MIN + 1);
    let mut beta = previous_score.saturating_add(window);

    loop {
        let result = search_root_depth_dispatch(
            board,
            candidate_moves,
            depth,
            previous_pv,
            alpha,
            beta,
            context,
            chess960,
        )?;
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

fn search_root_child(
    board: &Board,
    ordered: ScoredMove,
    depth: u32,
    previous_pv: &[PvMove],
    pv_move: Option<Move>,
    alpha: i32,
    beta: i32,
    searched_moves: u32,
    is_pv_node: bool,
    context: &mut SearchContext<'_>,
) -> Option<SearchOutcome> {
    if context.should_stop().is_some() {
        return None;
    }
    let root_key = position_key(board);
    context.push_repetition_key(root_key);
    let mut next = board.clone();
    next.play_unchecked(ordered.mv);
    let next_key = position_key(&next);
    let next_repetition = context.push_position(&next, next_key);
    context.push_eval_state(board, &next, ordered.mv);
    let child_pv = if Some(ordered.mv) == pv_move && !previous_pv.is_empty() {
        &previous_pv[..previous_pv.len() - 1]
    } else {
        &[]
    };
    let use_pvs = should_use_pvs(is_pv_node, searched_moves, alpha, beta);
    let child = search_child_with_pvs(
        &next,
        next_repetition,
        depth - 1,
        depth,
        alpha,
        beta,
        child_pv,
        ordered.mv,
        CorrectionContext::default().after_move(ordered.mv, ordered.moving_piece),
        context,
        1,
        use_pvs,
    );
    context.pop_eval_state(board, ordered.mv);
    context.pop_position(next_key);
    context.pop_position(root_key);
    let child = child?;
    context.note_searched_move(1);
    Some(child)
}

#[inline]
pub(in crate::search) fn search_child_with_pvs(
    board: &Board,
    repetition: bool,
    depth: u32,
    root_depth: u32,
    alpha: i32,
    beta: i32,
    child_pv: &[PvMove],
    previous_move: Move,
    correction_context: CorrectionContext,
    context: &mut SearchContext<'_>,
    ply: u16,
    use_pvs: bool,
) -> Option<SearchOutcome> {
    debug_assert!(alpha < beta);
    debug_assert!(!use_pvs || beta > alpha.saturating_add(1));

    if use_pvs {
        let scout_beta = alpha.saturating_neg();
        let scout_alpha = scout_beta.saturating_sub(1);
        let scout_child = negamax(
            board,
            repetition,
            depth,
            root_depth,
            scout_alpha,
            scout_beta,
            &[],
            Some(previous_move),
            correction_context,
            context,
            ply,
            true,
            None,
        )?;
        let scout_score = -scout_child.score;
        if scout_score <= alpha || scout_score >= beta {
            return Some(scout_child);
        }
    }

    negamax(
        board,
        repetition,
        depth,
        root_depth,
        beta.saturating_neg(),
        alpha.saturating_neg(),
        child_pv,
        Some(previous_move),
        correction_context,
        context,
        ply,
        true,
        None,
    )
}
