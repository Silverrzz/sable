use cozy_chess::{File, Rank};

use crate::{Board, Move, Piece, Square};

pub(crate) fn map_castling_uci_notation(mv: &str) -> Option<&'static str> {
    match mv {
        "e1g1" => Some("e1h1"),
        "e1c1" => Some("e1a1"),
        "e8g8" => Some("e8h8"),
        "e8c8" => Some("e8a8"),
        "e1h1" => Some("e1g1"),
        "e1a1" => Some("e1c1"),
        "e8h8" => Some("e8g8"),
        "e8a8" => Some("e8c8"),
        _ => None,
    }
}

pub(crate) fn map_castling_target_notation(board: &Board, mv: &str) -> Option<Move> {
    if mv.len() != 4 {
        return None;
    }
    let from = mv.get(0..2)?.parse::<Square>().ok()?;
    let to = mv.get(2..4)?.parse::<Square>().ok()?;
    let color = board.side_to_move();
    let rank = Rank::First.relative_to(color);
    if from != board.king(color) || from.rank() != rank || to.rank() != rank {
        return None;
    }
    let rook_file = match to.file() {
        File::G => board.castle_rights(color).short,
        File::C => board.castle_rights(color).long,
        _ => None,
    }?;
    Some(Move {
        from,
        to: Square::new(rook_file, rank),
        promotion: None,
    })
}

pub(super) fn castling_target_square(board: &Board, mv: Move) -> Option<Square> {
    let (castle_side, rank) = castling_side(board, mv)?;
    Some(match castle_side {
        CastleSide::Short => Square::new(File::G, rank),
        CastleSide::Long => Square::new(File::C, rank),
    })
}

#[derive(Clone, Copy)]
enum CastleSide {
    Short,
    Long,
}

fn castling_side(board: &Board, mv: Move) -> Option<(CastleSide, Rank)> {
    if board.piece_on(mv.from) != Some(Piece::King) {
        return None;
    }
    if board.color_on(mv.from) != Some(board.side_to_move()) || !board.is_legal(mv) {
        return None;
    }

    let rank = Rank::First.relative_to(board.side_to_move());
    let rights = board.castle_rights(board.side_to_move());
    if rights.short.map(|file| Square::new(file, rank)) == Some(mv.to) {
        return Some((CastleSide::Short, rank));
    }
    if rights.long.map(|file| Square::new(file, rank)) == Some(mv.to) {
        return Some((CastleSide::Long, rank));
    }
    None
}
