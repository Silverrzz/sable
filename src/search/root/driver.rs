use std::{
    sync::atomic::{AtomicBool, AtomicU64},
    time::Instant,
};

use crate::{
    Board, Move,
    evaluation::Evaluator,
};

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
    lazy_smp::{lazy_smp_worker_depth, run_lazy_smp_search},
    multi_pv::{RootMoveResult, search_root_multi_pv_iteration},
    outcome::{PvMove, should_defer_repetition_root_switch},
    search_root_iteration,
    time_manager::IterativeTimeManager,
};

pub(crate) fn run_search<F>(
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
    chess960: bool,
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
    if multi_pv == 1
        && search_threads > 1
        && candidate_moves.len() > 1
        && can_use_lazy_smp(request)
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
            chess960,
            started,
            observer,
        );
    }

    run_search_single(
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
                worker_id: 0,
                multi_pv,
            },
            controls: RootSearchControls {
                stop_flag,
                ponder_flag,
                lazy_stop_flag: None,
                shared_nodes: None,
            },
        },
        observer,
    )
}

fn can_use_lazy_smp(request: &SearchRequest) -> bool {
    request.limits.nodes.is_none()
        && request.limits.soft_nodes.is_none()
        && request.limits.hard_nodes.is_none()
}

pub(in crate::search) struct RootSearchInput<'a> {
    pub(in crate::search) board: &'a Board,
    pub(in crate::search) game_history: &'a [PositionKey],
    pub(in crate::search) request: &'a SearchRequest,
    pub(in crate::search) candidate_moves: &'a [Move],
    pub(in crate::search) max_depth: u32,
    pub(in crate::search) chess960: bool,
}

pub(in crate::search) struct RootSearchRuntime {
    pub(in crate::search) budget: SearchBudget,
    pub(in crate::search) transposition_table: TranspositionTable,
    pub(in crate::search) search_state: PersistentSearchState,
    pub(in crate::search) evaluator: Evaluator,
    pub(in crate::search) started: Instant,
    pub(in crate::search) worker_id: usize,
    pub(in crate::search) multi_pv: u32,
}

pub(in crate::search) struct RootSearchControls<'a> {
    pub(in crate::search) stop_flag: Option<&'a AtomicBool>,
    pub(in crate::search) ponder_flag: Option<&'a AtomicBool>,
    pub(in crate::search) lazy_stop_flag: Option<&'a AtomicBool>,
    pub(in crate::search) shared_nodes: Option<&'a AtomicU64>,
}

pub(in crate::search) struct RootSearchJob<'a> {
    pub(in crate::search) input: RootSearchInput<'a>,
    pub(in crate::search) runtime: RootSearchRuntime,
    pub(in crate::search) controls: RootSearchControls<'a>,
}

pub(in crate::search) fn run_search_single<F>(
    job: RootSearchJob<'_>,
    mut observer: F,
) -> (SearchResult, PersistentSearchState)
where
    F: FnMut(&SearchInfo),
{
    let RootSearchJob {
        input:
            RootSearchInput {
                board,
                game_history,
                request,
                candidate_moves,
                max_depth,
                chess960,
            },
        runtime:
            RootSearchRuntime {
                budget,
                transposition_table,
                search_state,
                evaluator,
                started,
                worker_id,
                multi_pv,
            },
        controls:
            RootSearchControls {
                stop_flag,
                ponder_flag,
                lazy_stop_flag,
                shared_nodes,
            },
    } = job;

    let mut context = SearchContext::new(SearchContextConfig {
        root_board: board,
        started,
        hard_time_ms: budget.hard_time_ms,
        node_limit: request.limits.hard_nodes.or(request.limits.nodes),
        soft_node_limit: request.limits.soft_nodes,
        evaluator: evaluator.clone(),
        stop_flag,
        ponder_flag,
        game_history,
        transposition_table,
        chess960,
        search_state,
    });
    context.set_lazy_smp_state(lazy_stop_flag, shared_nodes);

    let mut root = RootSearchState::new(board, candidate_moves, &budget, &mut context);
    if !candidate_moves.is_empty() && multi_pv > 1 {
        run_multi_pv_iterations(
            board,
            candidate_moves,
            &budget,
            max_depth,
            multi_pv,
            chess960,
            worker_id,
            &mut context,
            &mut root,
            &mut observer,
        );
    } else if !candidate_moves.is_empty() {
        run_single_pv_iterations(
            board,
            candidate_moves,
            &budget,
            max_depth,
            chess960,
            worker_id,
            &mut context,
            &mut root,
            &mut observer,
        );
    }

    finish_search(&budget, root, context)
}

struct RootSearchState {
    best_move: Option<Move>,
    best_score: i32,
    best_pv: Vec<PvMove>,
    completed_depth: u32,
    time_manager: IterativeTimeManager,
}

impl RootSearchState {
    fn new(
        board: &Board,
        candidate_moves: &[Move],
        budget: &SearchBudget,
        context: &mut SearchContext<'_>,
    ) -> Self {
        let terminal = terminal_score(board, false, 0);
        let best_score = terminal.unwrap_or_else(|| context.evaluate(board));
        Self {
            best_move: candidate_moves.first().copied(),
            best_score,
            best_pv: Vec::new(),
            completed_depth: 0,
            time_manager: IterativeTimeManager::new(budget),
        }
    }
}

fn run_multi_pv_iterations<F>(
    board: &Board,
    candidate_moves: &[Move],
    budget: &SearchBudget,
    max_depth: u32,
    multi_pv: u32,
    chess960: bool,
    worker_id: usize,
    context: &mut SearchContext<'_>,
    root: &mut RootSearchState,
    observer: &mut F,
)
where
    F: FnMut(&SearchInfo),
{
    let requested_multi_pv = (multi_pv as usize).min(candidate_moves.len());
    let mut previous_multi_pv = Vec::<RootMoveResult>::new();
    for nominal_depth in 1..=max_depth {
        let depth = lazy_smp_worker_depth(nominal_depth, worker_id, max_depth);
        match iteration_gate(depth, context, root) {
            IterationGate::Search => {}
            IterationGate::Skip => continue,
            IterationGate::Stop => break,
        }
        let Some(iteration_results) = search_root_multi_pv_iteration(
            board,
            candidate_moves,
            depth,
            &previous_multi_pv,
            requested_multi_pv,
            chess960,
            context,
        ) else {
            break;
        };
        let Some(best_result) = iteration_results.first() else {
            break;
        };
        record_completed_iteration(context, root, best_result.mv, best_result.score);
        root.best_move = Some(best_result.mv);
        root.best_score = best_result.score;
        root.best_pv = best_result.pv.clone();
        root.completed_depth = depth;
        for (idx, result) in iteration_results.iter().take(requested_multi_pv).enumerate() {
            let mut info = build_search_info(
                budget,
                root.completed_depth,
                context,
                result.score,
                &result.pv,
            );
            info.multi_pv = Some(idx as u32 + 1);
            observer(&info);
        }
        previous_multi_pv = iteration_results;
    }
}

fn run_single_pv_iterations<F>(
    board: &Board,
    candidate_moves: &[Move],
    budget: &SearchBudget,
    max_depth: u32,
    chess960: bool,
    worker_id: usize,
    context: &mut SearchContext<'_>,
    root: &mut RootSearchState,
    observer: &mut F,
)
where
    F: FnMut(&SearchInfo),
{
    for nominal_depth in 1..=max_depth {
        let depth = lazy_smp_worker_depth(nominal_depth, worker_id, max_depth);
        match iteration_gate(depth, context, root) {
            IterationGate::Search => {}
            IterationGate::Skip => continue,
            IterationGate::Stop => break,
        }
        let Some((iteration_move, iteration_outcome)) = search_root_iteration(
            board,
            candidate_moves,
            depth,
            root.best_score,
            &root.best_pv,
            root.completed_depth,
            context,
            chess960,
        ) else {
            break;
        };
        record_completed_iteration(context, root, iteration_move, iteration_outcome.score);
        if should_defer_repetition_root_switch(
            root.completed_depth,
            root.best_move,
            root.best_score,
            iteration_move,
            &iteration_outcome,
        ) {
            continue;
        }
        root.best_move = Some(iteration_move);
        root.best_score = iteration_outcome.score;
        root.best_pv = iteration_outcome.pv.clone();
        root.completed_depth = depth;
        let info = build_search_info(
            budget,
            root.completed_depth,
            context,
            root.best_score,
            &root.best_pv,
        );
        observer(&info);
    }
}

enum IterationGate {
    Search,
    Skip,
    Stop,
}

fn iteration_gate(
    depth: u32,
    context: &mut SearchContext<'_>,
    root: &mut RootSearchState,
) -> IterationGate {
    if depth <= root.completed_depth {
        return IterationGate::Skip;
    }
    if context.should_stop().is_some() {
        return IterationGate::Stop;
    }
    if context.should_stop_before_iteration_for_nodes(root.completed_depth) {
        return IterationGate::Stop;
    }
    if root
        .time_manager
        .should_stop_before_iteration(root.completed_depth, context)
    {
        return IterationGate::Stop;
    }
    IterationGate::Search
}

fn record_completed_iteration(
    context: &mut SearchContext<'_>,
    root: &mut RootSearchState,
    best_move: Move,
    score: i32,
) {
    let elapsed_ms = context.clock_elapsed_ms();
    root.time_manager
        .record_completed_iteration(elapsed_ms, context.local_nodes(), best_move, score);
}

fn finish_search(
    budget: &SearchBudget,
    root: RootSearchState,
    mut context: SearchContext<'_>,
) -> (SearchResult, PersistentSearchState) {
    context.flush_shared_node_counts();
    let ponder_move = root.best_pv.get(1).map(|pv| pv.mv);
    let info = build_search_info(
        budget,
        root.completed_depth,
        &mut context,
        root.best_score,
        &root.best_pv,
    );
    let search_state = context.take_persistent_state();
    (
        SearchResult {
            best_move: root.best_move,
            ponder_move,
            info,
        },
        search_state,
    )
}
