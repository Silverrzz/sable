use std::str::FromStr;

use crate::{Board, EngineError, Move};

use super::castling::{
    castling_target_square, map_castling_target_notation, map_castling_uci_notation,
};

pub(crate) fn parse_legal_move_for_board(
    board: &Board,
    mv: &str,
    chess960: bool,
) -> Result<Move, EngineError> {
    if let Ok(parsed) = Move::from_str(mv)
        && board.is_legal(parsed)
    {
        return Ok(parsed);
    }

    if !chess960
        && let Some(mapped) = map_castling_uci_notation(mv)
        && let Ok(parsed) = Move::from_str(mapped)
        && board.is_legal(parsed)
    {
        return Ok(parsed);
    }

    if let Some(mapped) = map_castling_target_notation(board, mv)
        && board.is_legal(mapped)
    {
        return Ok(mapped);
    }

    Err(EngineError::InvalidMove(mv.to_owned()))
}

pub(crate) fn format_uci_move_for_board(board: &Board, mv: Move, chess960: bool) -> String {
    if chess960 {
        return mv.to_string();
    }
    let Some(to) = castling_target_square(board, mv) else {
        return mv.to_string();
    };
    format!("{}{}", mv.from, to)
}
