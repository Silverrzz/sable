mod shared_state;
mod time_budget;
mod verbose_eval;

use std::sync::{Arc, atomic::AtomicBool};

pub use verbose_eval::{VerboseEval, VerboseEvalSquare};

use crate::{
    Board, Color, EngineError, EngineOptions, GameStatus, Move,
    chess::{board_from_fen, generate_moves},
    evaluation::{DRAW_SCORE, LOSS_SCORE, NnueArchitectureId, is_board_drawn},
    nnue::Accumulator,
    options::apply_engine_option,
    perft::perft,
    protocol::uci::{format_uci_move_for_board, mate_score_to_uci, parse_legal_move_for_board},
    search::{
        PersistentSearchState, PositionKey, SearchBudget, SearchInfo, SearchRequest, SearchResult,
        StaticEval, StaticEvalSource, TranspositionTable, is_claimable_repetition_draw,
        dispatch_search, max_depth_from_limits, position_key, select_candidate_moves,
    },
};

use shared_state::SharedSearchState;
use time_budget::compute_search_budget;
use verbose_eval::build_verbose_eval;

#[derive(Clone, Debug)]
pub struct Engine {
    board: Board,
    game_history: Vec<PositionKey>,
    options: EngineOptions,
    transposition_table: TranspositionTable,
    search_state: Arc<SharedSearchState>,
}

fn terminal_static_eval(score_cp: i32) -> StaticEval {
    StaticEval {
        score_cp,
        score_mate: mate_score_to_uci(score_cp),
        source: StaticEvalSource::Terminal,
    }
}

impl Default for Engine {
    fn default() -> Self {
        let board = Board::default();
        let key = position_key(&board);
        let game_history = vec![key];
        let options = EngineOptions::default();
        let transposition_table = TranspositionTable::new(options.hash_mb);
        let search_state = Arc::new(SharedSearchState::default());
        Self {
            board,
            game_history,
            options,
            transposition_table,
            search_state,
        }
    }
}

impl Engine {
    fn reset_game_history(&mut self) {
        self.game_history.clear();
        self.game_history.push(position_key(&self.board));
    }

    pub fn reset(&mut self) {
        self.board = Board::default();
        self.reset_game_history();
        self.transposition_table = TranspositionTable::new(self.options.hash_mb);
        self.search_state.reset();
    }

    pub fn clear_hash(&mut self) {
        self.transposition_table = TranspositionTable::new(self.options.hash_mb);
    }

    pub fn active_nnue_architecture_id(&self) -> Option<NnueArchitectureId> {
        Some(NnueArchitectureId::Goliath)
    }

    pub fn loaded_nnue_architecture_id(&self) -> Option<NnueArchitectureId> {
        self.active_nnue_architecture_id()
    }

    pub fn set_option(&mut self, name: &str, value: Option<&str>) -> Result<(), EngineError> {
        let normalized = name.to_ascii_lowercase().replace(' ', "");
        let previous_hash_mb = self.options.hash_mb;
        if normalized == "clearhash" {
            self.clear_hash();
            return Ok(());
        }
        apply_engine_option(&mut self.options, name, value)?;
        if normalized == "hash" && self.options.hash_mb != previous_hash_mb {
            self.transposition_table = TranspositionTable::new(self.options.hash_mb);
        }
        Ok(())
    }

    pub fn set_startpos_with_moves(&mut self, moves: &[String]) -> Result<(), EngineError> {
        self.board = Board::default();
        self.reset_game_history();
        self.apply_moves(moves)
    }

    pub fn set_fen_with_moves(&mut self, fen: &str, moves: &[String]) -> Result<(), EngineError> {
        self.board = parse_fen(fen, self.options.uci_chess960)?;
        self.reset_game_history();
        self.apply_moves(moves)
    }

    pub fn set_board(&mut self, board: Board) {
        self.board = board;
        self.reset_game_history();
        self.search_state.reset();
    }

    pub fn apply_moves(&mut self, moves: &[String]) -> Result<(), EngineError> {
        for mv in moves {
            let parsed = parse_legal_move_for_board(&self.board, mv, self.options.uci_chess960)?;
            crate::chess::play(&mut self.board, parsed);
            self.game_history.push(position_key(&self.board));
        }
        Ok(())
    }

    pub fn full_fen(&self) -> String {
        self.board.to_string()
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let mut legal_moves = Vec::new();
        generate_moves(&self.board, |piece_moves| {
            legal_moves.extend(piece_moves);
            false
        });
        legal_moves
    }

    pub fn play_move(&mut self, mv: Move) {
        crate::chess::play(&mut self.board, mv);
        self.game_history.push(position_key(&self.board));
    }

    pub fn search(&self, request: &SearchRequest) -> Result<SearchResult, EngineError> {
        self.search_with_observer(request, None, |_| {})
    }

    pub fn search_with_observer<F>(
        &self,
        request: &SearchRequest,
        stop_flag: Option<&AtomicBool>,
        observer: F,
    ) -> Result<SearchResult, EngineError>
    where
        F: FnMut(&SearchInfo),
    {
        self.search_with_controls(request, stop_flag, None, observer)
    }

    pub fn search_with_controls<F>(
        &self,
        request: &SearchRequest,
        stop_flag: Option<&AtomicBool>,
        ponder_flag: Option<&AtomicBool>,
        observer: F,
    ) -> Result<SearchResult, EngineError>
    where
        F: FnMut(&SearchInfo),
    {
        let request = self.search_request_with_option_defaults(request);
        let candidate_moves = select_candidate_moves(
            &self.board,
            &request.search_moves,
            self.options.uci_chess960,
        )?;
        let max_depth = max_depth_from_limits(&request);
        let budget = self.compute_search_budget(&request);
        let persistent = self.options.multi_pv <= 1;
        let (search_state_generation, search_state) = self.search_state_for_request(persistent);
        let transposition_table = self.transposition_table_for_request(persistent);
        let (result, search_state) = dispatch_search(
            &self.board,
            &self.game_history,
            &request,
            &candidate_moves,
            budget,
            max_depth,
            transposition_table,
            search_state,
            self.options.threads,
            self.options.multi_pv,
            stop_flag,
            ponder_flag,
            observer,
        );
        if persistent {
            self.search_state
                .store_if_current(search_state_generation, search_state);
        }
        Ok(result)
    }

    fn search_request_with_option_defaults(&self, request: &SearchRequest) -> SearchRequest {
        let mut request = request.clone();
        if request.limits.nodes == Some(1) {
            request.limits.nodes = None;
            request.limits.depth = Some(1);
        }
        if self.options.use_soft_nodes
            && request.limits.soft_nodes.is_none()
            && let Some(nodes) = request.limits.nodes.take()
        {
            request.limits.soft_nodes = Some(nodes);
        }
        if self.options.use_soft_nodes
            && request.limits.hard_nodes.is_none()
            && let Some(soft_nodes) = request.limits.soft_nodes
        {
            request.limits.hard_nodes = Some(soft_nodes.saturating_mul(10));
        }
        request
    }

    fn search_state_for_request(&self, persistent: bool) -> (u64, PersistentSearchState) {
        if persistent {
            self.search_state.snapshot()
        } else {
            (0, PersistentSearchState::default())
        }
    }

    fn transposition_table_for_request(&self, persistent: bool) -> TranspositionTable {
        if persistent {
            self.transposition_table.clone()
        } else {
            TranspositionTable::new(self.options.hash_mb)
        }
    }

    pub fn compute_search_budget(&self, request: &SearchRequest) -> SearchBudget {
        compute_search_budget(&self.board, &self.options, request)
    }

    pub fn side_to_move(&self) -> Color {
        crate::chess::side_to_move(&self.board)
    }

    pub fn status(&self) -> GameStatus {
        crate::chess::status(&self.board)
    }

    pub fn perft(&self, depth: u32) -> u64 {
        perft(&self.board, depth)
    }

    pub fn format_uci_move(&self, mv: Move) -> String {
        format_uci_move_for_board(&self.board, mv, self.options.uci_chess960)
    }

    pub fn format_uci_pv(&self, moves: &[Move]) -> Vec<String> {
        let mut board = self.board.clone();
        let mut formatted = Vec::with_capacity(moves.len());
        for &mv in moves {
            if crate::chess::status(&board) != GameStatus::Ongoing
                || !crate::chess::is_legal(&board, mv)
            {
                break;
            }
            formatted.push(format_uci_move_for_board(
                &board,
                mv,
                self.options.uci_chess960,
            ));
            crate::chess::play_unchecked(&mut board, mv);
        }
        formatted
    }

    pub fn verbose_eval(&self) -> Result<VerboseEval, EngineError> {
        Ok(build_verbose_eval(&self.board, self.static_eval()?))
    }

    pub fn static_eval(&self) -> Result<StaticEval, EngineError> {
        if is_claimable_repetition_draw(&self.board, &self.game_history)
            || is_board_drawn(&self.board)
        {
            return Ok(terminal_static_eval(DRAW_SCORE));
        }
        match crate::chess::status(&self.board) {
            GameStatus::Drawn => return Ok(terminal_static_eval(DRAW_SCORE)),
            GameStatus::Won => return Ok(terminal_static_eval(LOSS_SCORE)),
            GameStatus::Ongoing => {}
        }

        let score_cp = Accumulator::from_board(&self.board).eval(&self.board);
        Ok(StaticEval {
            score_cp,
            score_mate: mate_score_to_uci(score_cp),
            source: StaticEvalSource::Nnue,
        })
    }
}

fn parse_fen(fen: &str, chess960: bool) -> Result<Board, EngineError> {
    match board_from_fen(fen, chess960) {
        Ok(board) => Ok(board),
        Err(_) => {
            let normalized = normalize_fen(fen);
            if normalized == fen {
                return Err(EngineError::InvalidFen(fen.to_owned()));
            }
            board_from_fen(&normalized, chess960)
                .map_err(|_| EngineError::InvalidFen(fen.to_owned()))
        }
    }
}

fn normalize_fen(fen: &str) -> String {
    let Some((placement, rest)) = fen.split_once(' ') else {
        return normalize_fen_placement(fen);
    };
    format!("{} {}", normalize_fen_placement(placement), rest)
}

fn normalize_fen_placement(placement: &str) -> String {
    let mut normalized = String::with_capacity(placement.len());
    let mut empty_squares = 0u32;

    for char in placement.chars() {
        match char {
            '1'..='8' => empty_squares += char.to_digit(10).unwrap_or(0),
            _ => {
                flush_empty_squares(&mut normalized, &mut empty_squares);
                normalized.push(char);
            }
        }
    }

    flush_empty_squares(&mut normalized, &mut empty_squares);
    normalized
}

fn flush_empty_squares(normalized: &mut String, empty_squares: &mut u32) {
    if *empty_squares > 0 {
        normalized.push_str(&empty_squares.to_string());
        *empty_squares = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_node_request_becomes_depth_one_search() {
        let mut engine = Engine::default();
        engine.options.use_soft_nodes = true;
        let mut request = SearchRequest::default();
        request.limits.nodes = Some(1);

        let normalized = engine.search_request_with_option_defaults(&request);

        assert_eq!(normalized.limits.depth, Some(1));
        assert_eq!(normalized.limits.nodes, None);
        assert_eq!(normalized.limits.soft_nodes, None);
        assert_eq!(normalized.limits.hard_nodes, None);
    }

    #[test]
    fn search_result_contains_a_forward_playable_pv() {
        let engine = Engine::default();
        let mut request = SearchRequest::default();
        request.limits.depth = Some(3);

        let result = engine.search(&request).expect("depth search succeeds");
        assert_eq!(result.best_move, result.info.pv.first().copied());

        let mut board = engine.board.clone();
        for mv in result.info.pv {
            assert!(crate::chess::is_legal(&board, mv));
            crate::chess::play_unchecked(&mut board, mv);
        }
    }

    #[test]
    fn multi_pv_reports_each_requested_line() {
        let mut engine = Engine::default();
        engine.options.multi_pv = 3;
        let mut request = SearchRequest::default();
        request.limits.depth = Some(2);
        let mut final_depth_lines = Vec::new();

        engine
            .search_with_observer(&request, None, |info| {
                if info.depth == 2 {
                    final_depth_lines.push(info.multi_pv);
                }
            })
            .expect("multi-PV search succeeds");

        assert_eq!(final_depth_lines, [Some(1), Some(2), Some(3)]);
    }
}
