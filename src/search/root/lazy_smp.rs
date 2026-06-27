use std::{
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};

use crate::{Board, Move, evaluation::Evaluator};

use super::super::{
    constants::ASPIRATION_MIN_DEPTH,
    state::{
        context::PersistentSearchState,
        position_key::PositionKey,
        transposition::TranspositionTable,
    },
    types::*,
    uci_info::nodes_per_second,
};
use super::outcome::{PvMove, debug_validate_pv};
use super::{
    RootSearchControls, RootSearchInput, RootSearchJob, RootSearchRuntime, run_search_single,
};

#[derive(Debug)]
struct LazySmpWorkerResult {
    worker_id: usize,
    result: SearchResult,
}

pub(in crate::search) fn run_lazy_smp_search<F>(
    board: &Board,
    game_history: &[PositionKey],
    request: &SearchRequest,
    candidate_moves: &[Move],
    budget: SearchBudget,
    max_depth: u32,
    transposition_table: TranspositionTable,
    search_state: PersistentSearchState,
    threads: u32,
    evaluator: Evaluator,
    stop_flag: Option<&AtomicBool>,
    ponder_flag: Option<&AtomicBool>,
    chess960: bool,
    started: Instant,
    mut observer: F,
) -> (SearchResult, PersistentSearchState)
where
    F: FnMut(&SearchInfo),
{
    let worker_count = (threads as usize).max(1);
    let lazy_stop = AtomicBool::new(false);
    let helper_start = AtomicBool::new(false);
    let shared_nodes = AtomicU64::new(0);
    let report_budget = budget.clone();
    let mut main_result = None;
    let main_search_state = search_state.clone();
    let mut worker_results = Vec::with_capacity(worker_count.saturating_sub(1));

    thread::scope(|scope| {
        let mut worker_handles = Vec::with_capacity(worker_count.saturating_sub(1));
        for worker_id in 1..worker_count {
            let transposition_table = transposition_table.clone();
            let evaluator = evaluator.clone();
            let search_state = search_state.clone();
            let mut budget = budget.clone();
            budget.soft_time_ms = None;
            let helper_start = &helper_start;
            let lazy_stop = &lazy_stop;
            let shared_nodes = &shared_nodes;
            worker_handles.push(scope.spawn(move || {
                if !wait_for_lazy_smp_start(helper_start, lazy_stop, stop_flag) {
                    return LazySmpWorkerResult {
                        worker_id,
                        result: SearchResult::default(),
                    };
                }
                let (result, _search_state) = run_search_single(
                    RootSearchJob {
                        input: RootSearchInput {
                            board,
                            game_history,
                            request,
                            candidate_moves,
                            max_depth,
                            chess960,
                        },
                        runtime: RootSearchRuntime {
                            budget,
                            transposition_table,
                            search_state,
                            evaluator,
                            started,
                            worker_id,
                            multi_pv: 1,
                        },
                        controls: RootSearchControls {
                            stop_flag,
                            ponder_flag,
                            lazy_stop_flag: Some(lazy_stop),
                            shared_nodes: Some(shared_nodes),
                        },
                    },
                    |_| {},
                );
                LazySmpWorkerResult { worker_id, result }
            }));
        }

        let (result, search_state) = run_search_single(
            RootSearchJob {
                input: RootSearchInput {
                    board,
                    game_history,
                    request,
                    candidate_moves,
                    max_depth,
                    chess960,
                },
                runtime: RootSearchRuntime {
                    budget,
                    transposition_table: transposition_table.clone(),
                    search_state: main_search_state,
                    evaluator,
                    started,
                    worker_id: 0,
                    multi_pv: 1,
                },
                controls: RootSearchControls {
                    stop_flag,
                    ponder_flag,
                    lazy_stop_flag: Some(&lazy_stop),
                    shared_nodes: Some(&shared_nodes),
                },
            },
            |info| {
                if info.depth.unwrap_or(0) >= ASPIRATION_MIN_DEPTH {
                    helper_start.store(true, Ordering::Relaxed);
                }
                observer(info);
            },
        );
        helper_start.store(true, Ordering::Relaxed);
        lazy_stop.store(true, Ordering::Relaxed);
        for handle in worker_handles {
            if let Ok(result) = handle.join() {
                worker_results.push(result);
            }
        }
        let result = select_lazy_smp_result(result, &worker_results);
        main_result = Some((result, search_state));
    });

    let (best, search_state) = main_result.unwrap_or_default();

    let pv: Vec<PvMove> = best
        .info
        .pv
        .iter()
        .rev()
        .map(|&mv| PvMove::new(board, mv, chess960))
        .collect();
    debug_validate_pv(board, &pv, "SMPFINAL");

    let best = refresh_lazy_smp_info(
        best,
        started,
        &report_budget,
        &transposition_table,
        shared_nodes.load(Ordering::Relaxed),
    );

    (best, search_state)
}

pub(in crate::search) fn lazy_smp_worker_depth(nominal_depth: u32, worker_id: usize, max_depth: u32) -> u32 {
    let offset = lazy_smp_worker_depth_offset(nominal_depth, worker_id);
    nominal_depth.saturating_add(offset).min(max_depth)
}

fn lazy_smp_worker_depth_offset(nominal_depth: u32, worker_id: usize) -> u32 {
    if worker_id == 0 || nominal_depth <= ASPIRATION_MIN_DEPTH {
        return 0;
    }
    if worker_id % 2 == 0 { 2 } else { 1 }
}

fn wait_for_lazy_smp_start(
    helper_start: &AtomicBool,
    lazy_stop: &AtomicBool,
    stop_flag: Option<&AtomicBool>,
) -> bool {
    while !helper_start.load(Ordering::Relaxed) {
        if lazy_stop.load(Ordering::Relaxed)
            || stop_flag
                .map(|flag| flag.load(Ordering::Relaxed))
                .unwrap_or(false)
        {
            return false;
        }
        thread::sleep(Duration::from_millis(1));
    }
    true
}

fn select_lazy_smp_result(
    main_result: SearchResult,
    worker_results: &[LazySmpWorkerResult],
) -> SearchResult {
    let mut best = main_result;
    let mut best_worker_id = 0;
    for worker in worker_results {
        if prefer_lazy_smp_result(&worker.result, worker.worker_id, &best, best_worker_id) {
            best = worker.result.clone();
            best_worker_id = worker.worker_id;
        }
    }
    best
}

fn prefer_lazy_smp_result(
    candidate: &SearchResult,
    candidate_worker_id: usize,
    current: &SearchResult,
    current_worker_id: usize,
) -> bool {
    let candidate_has_move = candidate.best_move.is_some();
    let current_has_move = current.best_move.is_some();
    if candidate_has_move != current_has_move {
        return candidate_has_move;
    }

    let candidate_depth = completed_depth(candidate);
    let current_depth = completed_depth(current);
    if candidate_depth != current_depth {
        return candidate_depth > current_depth;
    }

    current_worker_id != 0 && candidate_worker_id == 0
}

fn completed_depth(result: &SearchResult) -> u32 {
    result.info.depth.unwrap_or(0)
}

fn refresh_lazy_smp_info(
    mut result: SearchResult,
    started: Instant,
    budget: &SearchBudget,
    transposition_table: &TranspositionTable,
    nodes: u64,
) -> SearchResult {
    let elapsed = started.elapsed();
    let elapsed_ns = elapsed.as_nanos();
    result.info.budget = budget.clone();
    result.info.nodes = Some(nodes);
    result.info.time_ms = Some(elapsed.as_millis().min(u128::from(u64::MAX)) as u64);
    result.info.nps = nodes_per_second(nodes, elapsed_ns);
    result.info.hashfull = Some(transposition_table.hashfull());
    result
}
