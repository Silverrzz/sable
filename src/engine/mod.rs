mod shared_state;
mod static_eval;
mod time_budget;
mod verbose_eval;

use std::sync::{
    Arc,
    atomic::AtomicBool,
};

pub use verbose_eval::{VerboseEval, VerboseEvalSquare};

use crate::{
    Board, Color, EngineError, EngineOptions, GameStatus, Move,
    evaluation::{
        DRAW_SCORE, EvalMode, Evaluator, LOSS_SCORE, NnueArchitectureId, NnueModel,
        is_board_drawn,
    },
    options::apply_engine_option,
    perft::perft,
    search::{
        PersistentSearchState, PositionKey, SearchBudget, SearchInfo, SearchRequest, SearchResult,
        StaticEval, StaticEvalSource, TranspositionTable, is_claimable_repetition_draw,
        max_depth_from_limits, position_key, run_search, select_candidate_moves,
    },
    protocol::uci::{format_uci_move_for_board, mate_score_to_uci, parse_legal_move_for_board},
};

use shared_state::SharedSearchState;
use static_eval::terminal_static_eval;
use time_budget::compute_search_budget;
use verbose_eval::build_verbose_eval;

#[derive(Clone, Debug)]
pub struct Engine {
    board: Board,
    game_history: Vec<PositionKey>,
    options: EngineOptions,
    evaluator: Evaluator,
    transposition_table: TranspositionTable,
    search_state: Arc<SharedSearchState>,
    startup_warnings: Vec<String>,
}

impl Default for Engine {
    fn default() -> Self {
        let board = Board::default();
        let key = position_key(&board);
        let game_history = vec![key];
        let mut options = EngineOptions::default();
        let mut startup_warnings = Vec::new();
        let nnue = match NnueModel::shared_embedded_default() {
            Some(Ok(model)) => {
                options.eval_file = Some("embedded".to_owned());
                Some(model)
            }
            Some(Err(error)) => {
                options.eval_mode = EvalMode::Hce;
                startup_warnings.push(format!(
                    "embedded eval model failed to load, falling back to hce: {error}"
                ));
                None
            }
            None => None,
        };
        if options.eval_mode == EvalMode::Nnue && nnue.is_none() {
            options.eval_mode = EvalMode::Hce;
        }
        let evaluator = Evaluator::new(options.eval_mode, nnue);
        let transposition_table = TranspositionTable::new(options.hash_mb);
        let search_state = Arc::new(SharedSearchState::default());
        Self {
            board,
            game_history,
            options,
            evaluator,
            transposition_table,
            search_state,
            startup_warnings,
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
        self.evaluator
            .active_nnue_model()
            .map(|model| model.architecture_id())
    }

    pub fn set_option(&mut self, name: &str, value: Option<&str>) -> Result<(), EngineError> {
        let normalized = name.to_ascii_lowercase().replace(' ', "");
        let previous_hash_mb = self.options.hash_mb;
        if normalized == "clearhash" {
            self.clear_hash();
            return Ok(());
        } else if normalized == "evalfile" {
            self.set_eval_file_option(name, value)?;
        } else if normalized == "eval" || normalized == "evaluation" {
            self.set_eval_mode_option(name, value)?;
        }
        apply_engine_option(&mut self.options, name, value)?;
        if should_reset_transposition_table(&normalized, self.options.hash_mb, previous_hash_mb) {
            self.transposition_table = TranspositionTable::new(self.options.hash_mb);
        }
        Ok(())
    }

    fn set_eval_file_option(
        &mut self,
        name: &str,
        value: Option<&str>,
    ) -> Result<(), EngineError> {
        let Some(path) = value else {
            return Err(EngineError::InvalidOptionValue {
                option: name.to_owned(),
                value: "<missing>".to_owned(),
            });
        };
        if path.eq_ignore_ascii_case("embedded") || path.starts_with("embedded:") {
            let model = match NnueModel::shared_embedded_default() {
                Some(Ok(model)) => model,
                Some(Err(error)) => return Err(error),
                None => {
                    return Err(EngineError::InvalidEvalFile {
                        path: path.to_owned(),
                        message: "no embedded eval model was compiled in".to_owned(),
                    });
                }
            };
            self.evaluator.set_nnue_model(model);
        } else {
            let model = NnueModel::load_from_file(path)?;
            self.evaluator.set_nnue_model(Arc::new(model));
        }
        Ok(())
    }

    fn set_eval_mode_option(
        &mut self,
        name: &str,
        value: Option<&str>,
    ) -> Result<(), EngineError> {
        let raw = value.ok_or_else(|| EngineError::InvalidOptionValue {
            option: name.to_owned(),
            value: "<missing>".to_owned(),
        })?;
        let mode = EvalMode::from_uci(raw).ok_or_else(|| EngineError::InvalidOptionValue {
            option: name.to_owned(),
            value: raw.to_owned(),
        })?;
        if mode == EvalMode::Nnue && !self.evaluator.has_nnue_model() {
            return Err(EngineError::InvalidOptionValue {
                option: name.to_owned(),
                value: raw.to_owned(),
            });
        }
        self.evaluator.set_mode(mode);
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
            self.board.play(parsed);
            self.game_history.push(position_key(&self.board));
        }
        Ok(())
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
        let candidate_moves = select_candidate_moves(
            &self.board,
            &request.search_moves,
            self.options.uci_chess960,
        )?;
        let max_depth = max_depth_from_limits(request);
        let budget = self.compute_search_budget(request);
        let persistent = self.options.multi_pv <= 1;
        let (search_state_generation, search_state) = self.search_state_for_request(persistent);
        let transposition_table = self.transposition_table_for_request(persistent);
        let (result, search_state) = run_search(
            &self.board,
            &self.game_history,
            request,
            &candidate_moves,
            budget,
            max_depth,
            transposition_table,
            search_state,
            self.options.threads,
            self.options.multi_pv,
            self.options.uci_chess960,
            self.evaluator.clone(),
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
        self.board.side_to_move()
    }

    pub fn status(&self) -> GameStatus {
        self.board.status()
    }

    pub fn perft(&self, depth: u32) -> u64 {
        perft(&self.board, depth)
    }

    pub fn format_uci_move(&self, mv: Move) -> String {
        format_uci_move_for_board(&self.board, mv, self.options.uci_chess960)
    }

    pub fn eval_file_option_value(&self) -> Option<&str> {
        self.options.eval_file.as_deref()
    }

    pub fn eval_mode_option_value(&self) -> EvalMode {
        self.options.eval_mode
    }

    pub fn show_wdl_option_value(&self) -> bool {
        self.options.uci_show_wdl
    }

    pub fn startup_warnings(&self) -> &[String] {
        &self.startup_warnings
    }

    pub fn verbose_eval(&self) -> VerboseEval {
        build_verbose_eval(&self.board, &self.evaluator, self.static_eval())
    }

    pub fn static_eval(&self) -> StaticEval {
        if is_claimable_repetition_draw(&self.board, &self.game_history)
            || is_board_drawn(&self.board)
        {
            return terminal_static_eval(DRAW_SCORE);
        }
        match self.board.status() {
            GameStatus::Drawn => return terminal_static_eval(DRAW_SCORE),
            GameStatus::Won => return terminal_static_eval(LOSS_SCORE),
            GameStatus::Ongoing => {}
        }

        let score_cp = self.evaluator.evaluate_for_side_to_move(&self.board);
        let source = if self.evaluator.active_nnue_model().is_some() {
            StaticEvalSource::Nnue
        } else {
            StaticEvalSource::Hce
        };
        StaticEval {
            score_cp,
            score_mate: mate_score_to_uci(score_cp),
            source,
        }
    }
}

fn should_reset_transposition_table(
    normalized_option: &str,
    hash_mb: u32,
    previous_hash_mb: u32,
) -> bool {
    match normalized_option {
        "hash" => hash_mb != previous_hash_mb,
        "eval" | "evaluation" | "evalfile" => true,
        _ => false,
    }
}

fn parse_fen(fen: &str, chess960: bool) -> Result<Board, EngineError> {
    match Board::from_fen(fen, chess960) {
        Ok(board) => Ok(board),
        Err(_) => {
            let normalized = normalize_fen(fen);
            if normalized == fen {
                return Err(EngineError::InvalidFen(fen.to_owned()));
            }
            Board::from_fen(&normalized, chess960)
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
