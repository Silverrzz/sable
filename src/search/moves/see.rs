
use crate::{
    Board, Color, Move, Piece, Square,
    chess::{
        BitBoard, Rank, get_bishop_moves, get_king_moves, get_knight_moves,
        get_pawn_attacks, get_rook_moves,
    },
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
    let mut colors = [crate::chess::colors(board, Color::White), crate::chess::colors(board, Color::Black)];
    let mut pieces = [
        crate::chess::pieces(board, Piece::Pawn),
        crate::chess::pieces(board, Piece::Knight),
        crate::chess::pieces(board, Piece::Bishop),
        crate::chess::pieces(board, Piece::Rook),
        crate::chess::pieces(board, Piece::Queen),
        crate::chess::pieces(board, Piece::King),
    ];
    let mut occupied = crate::chess::occupied(board);
    let side = crate::chess::side_to_move(board);

    remove_piece(
        &mut colors,
        &mut pieces,
        &mut occupied,
        side,
        moving_piece,
        mv.from,
    );
    add_piece(
        &mut colors,
        &mut pieces,
        &mut occupied,
        side,
        moving_piece,
        mv.to,
    );

    static_exchange_eval_on_target(
        mv.to,
        moving_piece,
        0,
        !side,
        &mut colors,
        &mut pieces,
        &mut occupied,
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
    let mut colors = [crate::chess::colors(board, Color::White), crate::chess::colors(board, Color::Black)];
    let mut pieces = [
        crate::chess::pieces(board, Piece::Pawn),
        crate::chess::pieces(board, Piece::Knight),
        crate::chess::pieces(board, Piece::Bishop),
        crate::chess::pieces(board, Piece::Rook),
        crate::chess::pieces(board, Piece::Queen),
        crate::chess::pieces(board, Piece::King),
    ];
    let mut occupied = crate::chess::occupied(board);

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
    let mut colors = [crate::chess::colors(board, Color::White), crate::chess::colors(board, Color::Black)];
    let mut pieces = [
        crate::chess::pieces(board, Piece::Pawn),
        crate::chess::pieces(board, Piece::Knight),
        crate::chess::pieces(board, Piece::Bishop),
        crate::chess::pieces(board, Piece::Rook),
        crate::chess::pieces(board, Piece::Queen),
        crate::chess::pieces(board, Piece::King),
    ];
    let mut occupied = crate::chess::occupied(board);
    let side = crate::chess::side_to_move(board);
    let placed_piece = mv.promotion.unwrap_or(moving_piece);
    let promotion_gain = piece_value(placed_piece) - piece_value(moving_piece);

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

    static_exchange_eval_on_target(
        mv.to,
        placed_piece,
        piece_value(captured_piece) + promotion_gain,
        !side,
        &mut colors,
        &mut pieces,
        &mut occupied,
    )
}

fn static_exchange_eval_on_target(
    target: Square,
    mut target_piece: Piece,
    initial_gain: i32,
    mut attacker_side: Color,
    colors: &mut [BitBoard; 2],
    pieces: &mut [BitBoard; 6],
    occupied: &mut BitBoard,
) -> i32 {
    let mut gains = [0_i32; 32];
    let mut depth = 0_usize;
    gains[0] = initial_gain;
    while depth + 1 < gains.len() {
        let Some((attacker_piece, attacker_square)) =
            least_valuable_attacker(target, attacker_side, *occupied, colors, pieces)
        else {
            break;
        };
        depth += 1;
        gains[depth] = piece_value(target_piece) - gains[depth - 1];
        remove_piece(
            colors,
            pieces,
            occupied,
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
