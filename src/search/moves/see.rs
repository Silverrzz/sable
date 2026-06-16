
use crate::{
    Board, Color, Move, Piece, Square,
};
use cozy_chess::{
    BitBoard, Rank, get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks,
    get_rook_moves,
};

use super::{
    board_moves::{captured_piece, en_passant_target, is_en_passant},
    scoring::piece_value,
};

pub(in crate::search) fn static_exchange_eval(board: &Board, mv: Move) -> i32 {
    let side = board.side_to_move();
    let moving_piece = board.piece_on(mv.from).unwrap_or(Piece::Pawn);
    let ep_target = en_passant_target(board, side);
    let Some(captured_piece) = captured_piece(board, moving_piece, mv, ep_target) else {
        return mv.promotion.map(piece_value).unwrap_or(0);
    };
    let captured_square = if is_en_passant(moving_piece, mv, ep_target) {
        Square::new(mv.to.file(), Rank::Fifth.relative_to(side))
    } else {
        mv.to
    };
    static_exchange_eval_capture(board, mv, moving_piece, captured_piece, captured_square)
}

pub(in crate::search) fn move_gives_check(
    board: &Board,
    mv: Move,
    moving_piece: Piece,
    captured_piece: Option<Piece>,
) -> bool {
    let side = board.side_to_move();
    let enemy = !side;
    let enemy_king = board.king(enemy);
    let ep_target = en_passant_target(board, side);
    let captured_square = if is_en_passant(moving_piece, mv, ep_target) {
        Square::new(mv.to.file(), Rank::Fifth.relative_to(side))
    } else {
        mv.to
    };
    let placed_piece = mv.promotion.unwrap_or(moving_piece);
    let mut colors = [board.colors(Color::White), board.colors(Color::Black)];
    let mut pieces = [
        board.pieces(Piece::Pawn),
        board.pieces(Piece::Knight),
        board.pieces(Piece::Bishop),
        board.pieces(Piece::Rook),
        board.pieces(Piece::Queen),
        board.pieces(Piece::King),
    ];
    let mut occupied = board.occupied();

    remove_piece(
        &mut colors,
        &mut pieces,
        &mut occupied,
        side,
        moving_piece,
        mv.from,
    );
    if let Some(captured_piece) = captured_piece {
        remove_piece(
            &mut colors,
            &mut pieces,
            &mut occupied,
            enemy,
            captured_piece,
            captured_square,
        );
    }
    add_piece(
        &mut colors,
        &mut pieces,
        &mut occupied,
        side,
        placed_piece,
        mv.to,
    );

    [
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
        Piece::King,
    ]
    .into_iter()
    .any(|piece| {
        !(attackers_for_piece(enemy_king, side, piece, occupied, &pieces)
            & colors[side as usize])
            .is_empty()
    })
}

pub(in crate::search) fn static_exchange_eval_capture(
    board: &Board,
    mv: Move,
    moving_piece: Piece,
    captured_piece: Piece,
    captured_square: Square,
) -> i32 {
    let mut colors = [board.colors(Color::White), board.colors(Color::Black)];
    let mut pieces = [
        board.pieces(Piece::Pawn),
        board.pieces(Piece::Knight),
        board.pieces(Piece::Bishop),
        board.pieces(Piece::Rook),
        board.pieces(Piece::Queen),
        board.pieces(Piece::King),
    ];
    let mut occupied = board.occupied();
    let side = board.side_to_move();
    let placed_piece = mv.promotion.unwrap_or(moving_piece);
    let promotion_gain = piece_value(placed_piece) - piece_value(moving_piece);
    let mut gains = [0_i32; 32];
    let mut depth = 0_usize;

    remove_piece(
        &mut colors,
        &mut pieces,
        &mut occupied,
        side,
        moving_piece,
        mv.from,
    );
    remove_piece(
        &mut colors,
        &mut pieces,
        &mut occupied,
        !side,
        captured_piece,
        captured_square,
    );
    add_piece(
        &mut colors,
        &mut pieces,
        &mut occupied,
        side,
        placed_piece,
        mv.to,
    );

    gains[0] = piece_value(captured_piece) + promotion_gain;
    let mut target_piece = placed_piece;
    let mut attacker_side = !side;

    while depth + 1 < gains.len() {
        let Some((attacker_piece, attacker_square)) =
            least_valuable_attacker(mv.to, attacker_side, occupied, &colors, &pieces)
        else {
            break;
        };
        depth += 1;
        gains[depth] = piece_value(target_piece) - gains[depth - 1];
        remove_piece(
            &mut colors,
            &mut pieces,
            &mut occupied,
            attacker_side,
            attacker_piece,
            attacker_square,
        );
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
    occupied: BitBoard,
    colors: &[BitBoard; 2],
    pieces: &[BitBoard; 6],
) -> Option<(Piece, Square)> {
    for piece in [
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
        Piece::King,
    ] {
        let attackers =
            attackers_for_piece(target, side, piece, occupied, pieces) & colors[side as usize];
        if let Some(square) = attackers.next_square() {
            return Some((piece, square));
        }
    }
    None
}

pub(in crate::search) fn attackers_for_piece(
    target: Square,
    side: Color,
    piece: Piece,
    occupied: BitBoard,
    pieces: &[BitBoard; 6],
) -> BitBoard {
    match piece {
        Piece::Pawn => get_pawn_attacks(target, !side) & pieces[Piece::Pawn as usize],
        Piece::Knight => get_knight_moves(target) & pieces[Piece::Knight as usize],
        Piece::Bishop => {
            get_bishop_moves(target, occupied) & pieces[Piece::Bishop as usize]
        }
        Piece::Rook => get_rook_moves(target, occupied) & pieces[Piece::Rook as usize],
        Piece::Queen => {
            (get_bishop_moves(target, occupied) | get_rook_moves(target, occupied))
                & pieces[Piece::Queen as usize]
        }
        Piece::King => get_king_moves(target) & pieces[Piece::King as usize],
    }
}

pub(in crate::search) fn remove_piece(
    colors: &mut [BitBoard; 2],
    pieces: &mut [BitBoard; 6],
    occupied: &mut BitBoard,
    color: Color,
    piece: Piece,
    square: Square,
) {
    let square = square.bitboard();
    colors[color as usize] -= square;
    pieces[piece as usize] -= square;
    *occupied -= square;
}

pub(in crate::search) fn add_piece(
    colors: &mut [BitBoard; 2],
    pieces: &mut [BitBoard; 6],
    occupied: &mut BitBoard,
    color: Color,
    piece: Piece,
    square: Square,
) {
    let square = square.bitboard();
    colors[color as usize] |= square;
    pieces[piece as usize] |= square;
    *occupied |= square;
}
