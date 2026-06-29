use crate::{
    Board, Move, Piece, Square,
    chess::{File, Rank},
};

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
    let color = crate::chess::side_to_move(board);
    let rank = Rank::First.relative_to(color);
    if from != crate::chess::king(board, color) || from.rank() != rank || to.rank() != rank {
        return None;
    }
    let rook_file = match to.file() {
        File::G => crate::chess::castle_rights(board, color).short,
        File::C => crate::chess::castle_rights(board, color).long,
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
    if crate::chess::piece_on(board, mv.from) != Some(Piece::King) {
        return None;
    }
    if crate::chess::color_on(board, mv.from) != Some(crate::chess::side_to_move(board)) || !crate::chess::is_legal(board, mv) {
        return None;
    }

    let rank = Rank::First.relative_to(crate::chess::side_to_move(board));
    let rights = crate::chess::castle_rights(board, crate::chess::side_to_move(board));
    if rights.short.map(|file| Square::new(file, rank)) == Some(mv.to) {
        return Some((CastleSide::Short, rank));
    }
    if rights.long.map(|file| Square::new(file, rank)) == Some(mv.to) {
        return Some((CastleSide::Long, rank));
    }
    None
}
