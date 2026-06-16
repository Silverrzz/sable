use std::str::FromStr;

use crate::{
    Board, EngineError, Move,
    protocol::uci::{map_castling_target_notation, map_castling_uci_notation},
};

pub(crate) fn select_candidate_moves(
    board: &Board,
    search_moves: &[String],
    chess960: bool,
) -> Result<Vec<Move>, EngineError> {
    if search_moves.is_empty() {
        return Ok(legal_moves(board));
    }

    let mut filtered = Vec::new();
    for mv in search_moves {
        let parsed = Move::from_str(mv).map_err(|_| EngineError::InvalidSearchMove(mv.clone()))?;
        if board.is_legal(parsed) {
            filtered.push(parsed);
        } else if !chess960
            && let Some(mapped) = map_castling_uci_notation(mv)
            && let Ok(mapped) = Move::from_str(mapped)
            && board.is_legal(mapped)
        {
            filtered.push(mapped);
        } else if let Some(mapped) = map_castling_target_notation(board, mv)
            && board.is_legal(mapped)
        {
            filtered.push(mapped);
        } else {
            return Err(EngineError::InvalidSearchMove(mv.clone()));
        }
    }
    Ok(filtered)
}

fn legal_moves(board: &Board) -> Vec<Move> {
    let mut legal_moves = Vec::new();
    board.generate_moves(|piece_moves| {
        legal_moves.extend(piece_moves);
        false
    });
    legal_moves
}
