
use crate::{
    Board, Color, Move, Piece, Square,
    chess::{BoardParts, Rank},
    pieces::ALL_PIECES,
};

use super::{
    board_moves::{en_passant_target, is_en_passant},
    scoring::piece_value,
};

pub(in crate::search) fn static_exchange_eval_for_move(
    board: &Board,
    mv: Move,
    moving_piece: Piece,
    captured_piece: Option<Piece>,
) -> i32 {
    let side = crate::chess::side_to_move(board);
    let ep_target = en_passant_target(board, side);
    static_exchange_eval_with_target(board, mv, moving_piece, captured_piece, side, ep_target)
}

fn static_exchange_eval_with_target(
    board: &Board,
    mv: Move,
    moving_piece: Piece,
    captured_piece: Option<Piece>,
    side: Color,
    ep_target: Option<Square>,
) -> i32 {
    let Some(captured_piece) = captured_piece else {
        return mv.promotion.map(piece_value).unwrap_or(0);
    };
    let captured_square = if is_en_passant(moving_piece, mv, ep_target) {
        Square::new(mv.to.file(), Rank::Fifth.relative_to(side))
    } else {
        mv.to
    };
    static_exchange_eval_capture(board, mv, moving_piece, captured_piece, captured_square)
}

pub(in crate::search) fn static_exchange_eval_for_quiet_move(
    board: &Board,
    mv: Move,
    moving_piece: Piece,
) -> i32 {
    let side = crate::chess::side_to_move(board);
    let mut parts = parts_after_move(board, side, moving_piece, mv, moving_piece, None);

    static_exchange_eval_on_target(
        mv.to,
        moving_piece,
        0,
        !side,
        &mut parts,
    )
}

pub(in crate::search) fn move_gives_check(
    board: &Board,
    mv: Move,
    moving_piece: Piece,
    captured_piece: Option<Piece>,
) -> bool {
    let side = crate::chess::side_to_move(board);
    let enemy = !side;
    let enemy_king = crate::chess::king(board, enemy);
    let ep_target = en_passant_target(board, side);
    let captured_square = if is_en_passant(moving_piece, mv, ep_target) {
        Square::new(mv.to.file(), Rank::Fifth.relative_to(side))
    } else {
        mv.to
    };
    let placed_piece = mv.promotion.unwrap_or(moving_piece);
    let captured = captured_piece.map(|piece| (enemy, piece, captured_square));
    let parts = parts_after_move(board, side, moving_piece, mv, placed_piece, captured);
    !parts.attackers_to(enemy_king, side).is_empty()
}

pub(in crate::search) fn static_exchange_eval_capture(
    board: &Board,
    mv: Move,
    moving_piece: Piece,
    captured_piece: Piece,
    captured_square: Square,
) -> i32 {
    let side = crate::chess::side_to_move(board);
    let placed_piece = mv.promotion.unwrap_or(moving_piece);
    let promotion_gain = piece_value(placed_piece) - piece_value(moving_piece);
    let mut parts = parts_after_move(
        board,
        side,
        moving_piece,
        mv,
        placed_piece,
        Some((!side, captured_piece, captured_square)),
    );

    static_exchange_eval_on_target(
        mv.to,
        placed_piece,
        piece_value(captured_piece) + promotion_gain,
        !side,
        &mut parts,
    )
}

fn parts_after_move(
    board: &Board,
    side: Color,
    moving_piece: Piece,
    mv: Move,
    placed_piece: Piece,
    captured: Option<(Color, Piece, Square)>,
) -> BoardParts {
    let mut parts = BoardParts::from_board(board);
    parts.remove_piece(side, moving_piece, mv.from);
    if let Some((captured_side, captured_piece, captured_square)) = captured {
        parts.remove_piece(captured_side, captured_piece, captured_square);
    }
    parts.add_piece(side, placed_piece, mv.to);
    parts
}

fn static_exchange_eval_on_target(
    target: Square,
    mut target_piece: Piece,
    initial_gain: i32,
    mut attacker_side: Color,
    parts: &mut BoardParts,
) -> i32 {
    let mut gains = [0_i32; 32];
    let mut depth = 0_usize;
    gains[0] = initial_gain;
    while depth + 1 < gains.len() {
        let Some((attacker_piece, attacker_square)) =
            least_valuable_attacker(target, attacker_side, parts)
        else {
            break;
        };
        depth += 1;
        gains[depth] = piece_value(target_piece) - gains[depth - 1];
        parts.remove_piece(attacker_side, attacker_piece, attacker_square);
        target_piece = attacker_piece;
        attacker_side = !attacker_side;
    }

    while depth > 0 {
        depth -= 1;
        gains[depth] = -(-gains[depth]).max(gains[depth + 1]);
    }
    gains[0]
}

pub(in crate::search) fn least_valuable_attacker(
    target: Square,
    side: Color,
    parts: &BoardParts,
) -> Option<(Piece, Square)> {
    for piece in ALL_PIECES {
        let attackers = parts.attackers_for_piece(target, side, piece) & parts.color(side);
        if let Some(square) = attackers.next_square() {
            return Some((piece, square));
        }
    }
    None
}
