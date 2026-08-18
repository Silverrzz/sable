use std::{
    sync::atomic::{AtomicBool, AtomicU64},
    time::Instant,
};

use crate::{Board, Move, evaluation::Evaluator};

use super::super::{
    state::{
        context::{PersistentSearchState, SearchContext, SearchContextConfig},
        position_key::PositionKey,
        transposition::TranspositionTable,
    },
    tree::scoring::terminal_score,
    types::*,
    uci_info::build_search_info,
};
use super::{
    depth::search_root_iteration,
    lazy_smp::{lazy_smp_worker_depth, run_lazy_smp_search},
    multi_pv::{RootMoveResult, search_root_multi_pv_iteration},
    outcome::should_defer_repetition_root_switch,
    time_manager::IterativeTimeManager,
};

pub(crate) fn dispatch_search<F>(
    board: &Board,
    game_history: &[PositionKey],
    request: &SearchRequest,
    candidate_moves: &[Move],
    budget: SearchBudget,
    max_depth: u32,
    transposition_table: TranspositionTable,
    search_state: PersistentSearchState,
    threads: u32,
    multi_pv: u32,
    evaluator: Evaluator,
    stop_flag: Option<&AtomicBool>,
    ponder_flag: Option<&AtomicBool>,
    observer: F,
) -> (SearchResult, PersistentSearchState)
where
    F: FnMut(&SearchInfo),
{
    let started = Instant::now();
    let search_threads = threads.max(1);
    let multi_pv = multi_pv.max(1);
    transposition_table.next_age();
    if multi_pv == 1 && search_threads > 1 && candidate_moves.len() > 1 && can_use_lazy_smp(request)
    {
        return run_lazy_smp_search(
            board,
            game_history,
            request,
            candidate_moves,
            budget,
            max_depth,
            transposition_table,
            search_state,
            search_threads,
            evaluator,
            stop_flag,
            ponder_flag,
            started,
            observer,
        );
    }

    iterative_deepening(
        RootSearchJob {
            board,
            game_history,
            request,
            candidate_moves,
            max_depth,
            budget,
            transposition_table,
            search_state,
            evaluator,
            started,
            worker_id: 0,
            multi_pv,
            stop_flag,
            ponder_flag,
            lazy_stop_flag: None,
            shared_nodes: None,
        },
        observer,
    )
}

fn can_use_lazy_smp(request: &SearchRequest) -> bool {
    request.limits.nodes.is_none()
        && request.limits.soft_nodes.is_none()
        && request.limits.hard_nodes.is_none()
}

pub(in crate::search) struct RootSearchJob<'a> {
    pub(in crate::search) board: &'a Board,
    pub(in crate::search) game_history: &'a [PositionKey],
    pub(in crate::search) request: &'a SearchRequest,
    pub(in crate::search) candidate_moves: &'a [Move],
    pub(in crate::search) max_depth: u32,
    pub(in crate::search) budget: SearchBudget,
    pub(in crate::search) transposition_table: TranspositionTable,
    pub(in crate::search) search_state: PersistentSearchState,
    pub(in crate::search) evaluator: Evaluator,
    pub(in crate::search) started: Instant,
    pub(in crate::search) worker_id: usize,
    pub(in crate::search) multi_pv: u32,
    pub(in crate::search) stop_flag: Option<&'a AtomicBool>,
    pub(in crate::search) ponder_flag: Option<&'a AtomicBool>,
    pub(in crate::search) lazy_stop_flag: Option<&'a AtomicBool>,
    pub(in crate::search) shared_nodes: Option<&'a AtomicU64>,
}

pub(in crate::search) fn iterative_deepening<F>(
    job: RootSearchJob<'_>,
    mut observer: F,
) -> (SearchResult, PersistentSearchState)
where
    F: FnMut(&SearchInfo),
{
    let RootSearchJob {
        board,
        game_history,
        request,
        candidate_moves,
        max_depth,
        budget,
        transposition_table,
        search_state,
        evaluator,
        started,
        worker_id,
        multi_pv,
        stop_flag,
        ponder_flag,
        lazy_stop_flag,
        shared_nodes,
    } = job;

    let mut context = SearchContext::new(SearchContextConfig {
        root_board: board,
        started,
        hard_time_ms: budget.hard_time_ms,
        node_limit: request.limits.hard_nodes.or(request.limits.nodes),
        soft_node_limit: request.limits.soft_nodes,
        evaluator,
        stop_flag,
        ponder_flag,
        game_history,
        transposition_table,
        search_state,
    });
    context.set_lazy_smp_state(lazy_stop_flag, shared_nodes);

    let movegen = crate::chess::MoveGenState::new(board);
    let mut best_move = candidate_moves.first().copied();
    let mut best_score =
        terminal_score(board, &movegen, false, 0).unwrap_or_else(|| context.evaluate(board));
    let mut best_pv = Vec::new();
    let mut completed_depth = 0;
    let mut time_manager = IterativeTimeManager::new(&budget);

    if !candidate_moves.is_empty() {
        let requested_multi_pv = (multi_pv as usize).min(candidate_moves.len());
        let mut previous_multi_pv = Vec::<RootMoveResult>::new();
        for nominal_depth in 1..=max_depth {
            let depth = lazy_smp_worker_depth(nominal_depth, worker_id, max_depth);
            if depth <= completed_depth {
                continue;
            }
            if context.should_stop()
                || context.should_stop_before_iteration_for_nodes(completed_depth)
                || time_manager.should_stop_before_iteration(completed_depth, &mut context)
            {
                break;
            }

            if requested_multi_pv > 1 {
                let Some(iteration_results) = search_root_multi_pv_iteration(
                    board,
                    candidate_moves,
                    depth,
                    &previous_multi_pv,
                    requested_multi_pv,
                    &mut context,
                ) else {
                    break;
                };
                let Some(best_result) = iteration_results.first() else {
                    break;
                };
                time_manager.record_completed_iteration(
                    context.clock_elapsed_ms(),
                    context.local_nodes(),
                    best_result.mv,
                    best_result.score,
                );
                best_move = Some(best_result.mv);
                best_score = best_result.score;
                best_pv.clone_from(&best_result.pv);
                completed_depth = depth;
                for (idx, result) in iteration_results
                    .iter()
                    .take(requested_multi_pv)
                    .enumerate()
                {
                    let mut info = build_search_info(
                        board,
                        &budget,
                        depth,
                        &mut context,
                        result.score,
                        &result.pv,
                    );
                    info.multi_pv = Some(idx as u32 + 1);
                    observer(&info);
                }
                previous_multi_pv = iteration_results;
                continue;
            }

            let Some((iteration_move, iteration_outcome)) = search_root_iteration(
                board,
                candidate_moves,
                depth,
                best_score,
                &best_pv,
                completed_depth,
                &mut context,
            ) else {
                break;
            };
            time_manager.record_completed_iteration(
                context.clock_elapsed_ms(),
                context.local_nodes(),
                iteration_move,
                iteration_outcome.score,
            );
            if should_defer_repetition_root_switch(
                completed_depth,
                best_move,
                best_score,
                iteration_move,
                &iteration_outcome,
            ) {
                continue;
            }
            best_move = Some(iteration_move);
            best_score = iteration_outcome.score;
            best_pv = iteration_outcome.pv;
            completed_depth = depth;
            observer(&build_search_info(
                board,
                &budget,
                depth,
                &mut context,
                best_score,
                &best_pv,
            ));
        }
    }

    context.flush_shared_node_counts();
    let ponder_move = best_pv
        .len()
        .checked_sub(2)
        .and_then(|index| best_pv.get(index))
        .copied();
    let info = build_search_info(
        board,
        &budget,
        completed_depth,
        &mut context,
        best_score,
        &best_pv,
    );
    let search_state = context.take_persistent_state();
    (
        SearchResult {
            best_move,
            ponder_move,
            info,
        },
        search_state,
    )
}
