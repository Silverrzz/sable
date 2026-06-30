use std::fmt;

use crate::pieces::ALL_PIECES;

use super::{
    BitBoard, Color, File, GameStatus, Move, Piece, Rank, Square, between_squares,
    get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks, get_queen_moves, get_rook_moves,
    line_squares,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FenParseError;

#[derive(Clone, Debug)]
pub struct Board {
    pieces: [BitBoard; 6],
    colors: [BitBoard; 2],
    side_to_move: Color,
    castle_rights: [CastleRights; 2],
    en_passant: Option<File>,
    halfmove_clock: u8,
    fullmove_number: u16,
    king_squares: [Square; 2],
    hash: u64,
}

impl Default for Board {
    fn default() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false)
            .expect("built-in start position FEN is valid")
    }
}

impl Board {
    fn empty() -> Self {
        Self {
            pieces: [BitBoard::EMPTY; 6],
            colors: [BitBoard::EMPTY; 2],
            side_to_move: Color::White,
            castle_rights: [CastleRights::default(); 2],
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            king_squares: [Square::E1, Square::E8],
            hash: 0,
        }
    }

    fn from_fen(fen: &str, chess960: bool) -> Result<Self, FenParseError> {
        let mut fields = fen.split_whitespace();
        let placement = fields.next().ok_or(FenParseError)?;
        let side_to_move = fields.next().ok_or(FenParseError)?;
        let castling = fields.next().ok_or(FenParseError)?;
        let en_passant = fields.next().ok_or(FenParseError)?;
        let halfmove_clock = fields.next().ok_or(FenParseError)?;
        let fullmove_number = fields.next().ok_or(FenParseError)?;
        if fields.next().is_some() {
            return Err(FenParseError);
        }

        let mut board = Self::empty();
        parse_piece_placement(&mut board, placement)?;
        board.side_to_move = parse_side_to_move(side_to_move)?;
        board.castle_rights = parse_castling_rights(&board, castling, chess960)?;
        board.en_passant = parse_en_passant(en_passant, board.side_to_move)?;
        board.halfmove_clock = parse_halfmove_clock(halfmove_clock)?;
        board.fullmove_number = parse_fullmove_number(fullmove_number)?;
        board.hash = compute_hash(&board);
        Ok(board)
    }

    fn apply_move_unchecked(&mut self, mv: Move) {
        let moving_piece = piece_on(self, mv.from).expect("missing piece on move source");
        self.apply_move_unchecked_with_piece(mv, moving_piece);
    }

    fn apply_move_unchecked_with_piece(&mut self, mv: Move, moving_piece: Piece) {
        let side = self.side_to_move;
        let enemy = !side;
        let ep_target = self
            .en_passant
            .map(|file| Square::new(file, Rank::Sixth.relative_to(side)));
        let is_castle = moving_piece == Piece::King && color_on(self, mv.to) == Some(side);
        let captured = if is_castle {
            None
        } else if moving_piece == Piece::Pawn
            && Some(mv.to) == ep_target
            && piece_on(self, mv.to).is_none()
        {
            Some((Piece::Pawn, Square::new(mv.to.file(), Rank::Fifth.relative_to(side))))
        } else {
            piece_on(self, mv.to).map(|piece| (piece, mv.to))
        };

        self.advance_clocks(side, moving_piece, captured.is_some());
        self.set_en_passant(None);

        if is_castle {
            self.apply_castle(side, mv);
        } else {
            self.remove_piece(side, moving_piece, mv.from);
            if let Some((captured_piece, captured_square)) = captured {
                self.remove_piece(enemy, captured_piece, captured_square);
                self.clear_castle_right_for_rook_square(enemy, captured_square);
            }
            self.add_piece(side, mv.promotion.unwrap_or(moving_piece), mv.to);
            self.update_castle_rights_after_piece_move(side, moving_piece, mv.from);
            self.update_en_passant_after_move(side, moving_piece, mv);
        }

        self.toggle_side_to_move();
    }

    fn advance_clocks(&mut self, side: Color, moving_piece: Piece, capture: bool) {
        self.halfmove_clock = if moving_piece == Piece::Pawn || capture {
            0
        } else {
            self.halfmove_clock.saturating_add(1).min(100)
        };
        if side == Color::Black {
            self.fullmove_number = self.fullmove_number.saturating_add(1);
        }
    }

    fn apply_castle(&mut self, side: Color, mv: Move) {
        let rank = Rank::First.relative_to(side);
        let short = mv.from.file() < mv.to.file();
        let king_to = Square::new(if short { File::G } else { File::C }, rank);
        let rook_to = Square::new(if short { File::F } else { File::D }, rank);

        self.remove_piece(side, Piece::King, mv.from);
        self.remove_piece(side, Piece::Rook, mv.to);
        self.add_piece(side, Piece::King, king_to);
        self.add_piece(side, Piece::Rook, rook_to);
        self.set_castle_right(side, true, None);
        self.set_castle_right(side, false, None);
    }

    fn update_castle_rights_after_piece_move(&mut self, side: Color, piece: Piece, from: Square) {
        if piece == Piece::King {
            self.set_castle_right(side, true, None);
            self.set_castle_right(side, false, None);
        } else if piece == Piece::Rook {
            self.clear_castle_right_for_rook_square(side, from);
        }
    }

    fn clear_castle_right_for_rook_square(&mut self, side: Color, square: Square) {
        let rank = Rank::First.relative_to(side);
        if square.rank() != rank {
            return;
        }
        let rights = self.castle_rights[side as usize];
        if rights.short == Some(square.file()) {
            self.set_castle_right(side, true, None);
        }
        if rights.long == Some(square.file()) {
            self.set_castle_right(side, false, None);
        }
    }

    fn update_en_passant_after_move(&mut self, side: Color, piece: Piece, mv: Move) {
        if piece != Piece::Pawn {
            return;
        }
        let from_rank = mv.from.rank() as i8;
        let to_rank = mv.to.rank() as i8;
        if (from_rank - to_rank).abs() == 2 {
            let start_rank = Rank::Second.relative_to(side);
            if mv.from.rank() == start_rank {
                self.set_en_passant(Some(mv.to.file()));
            }
        }
    }

    fn add_piece(&mut self, color: Color, piece: Piece, square: Square) {
        let bit = square.bitboard();
        self.pieces[piece as usize] |= bit;
        self.colors[color as usize] |= bit;
        if piece == Piece::King {
            self.king_squares[color as usize] = square;
        }
        self.hash ^= piece_key(color, piece, square);
    }

    fn remove_piece(&mut self, color: Color, piece: Piece, square: Square) {
        let bit = square.bitboard();
        self.pieces[piece as usize] -= bit;
        self.colors[color as usize] -= bit;
        self.hash ^= piece_key(color, piece, square);
    }

    fn set_en_passant(&mut self, file: Option<File>) {
        if let Some(old) = self.en_passant {
            self.hash ^= en_passant_key(old);
        }
        self.en_passant = file;
        if let Some(new) = self.en_passant {
            self.hash ^= en_passant_key(new);
        }
    }

    fn set_castle_right(&mut self, color: Color, short: bool, file: Option<File>) {
        let side = if short { 0 } else { 1 };
        let color_idx = color as usize;
        let current = if short {
            &mut self.castle_rights[color_idx].short
        } else {
            &mut self.castle_rights[color_idx].long
        };
        if *current == file {
            return;
        }
        if let Some(old) = *current {
            self.hash ^= castle_key(color, side, old);
        }
        *current = file;
        if let Some(new) = *current {
            self.hash ^= castle_key(color, side, new);
        }
    }

    fn toggle_side_to_move(&mut self) {
        self.side_to_move = !self.side_to_move;
        self.hash ^= side_key();
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in (0..8).rev() {
            let mut empty = 0u8;
            for file in 0..8 {
                let square = Square::new(
                    File::try_index(file).unwrap(),
                    Rank::try_index(rank).unwrap(),
                );
                if let Some(piece) = piece_on(self, square) {
                    if empty > 0 {
                        write!(f, "{empty}")?;
                        empty = 0;
                    }
                    let mut letter = piece_fen_letter(piece);
                    if color_on(self, square) == Some(Color::White) {
                        letter = letter.to_ascii_uppercase();
                    }
                    write!(f, "{letter}")?;
                } else {
                    empty += 1;
                }
            }
            if empty > 0 {
                write!(f, "{empty}")?;
            }
            if rank > 0 {
                write!(f, "/")?;
            }
        }

        write!(
            f,
            " {} {} {} {} {}",
            color_fen_letter(self.side_to_move),
            castling_fen(self),
            en_passant_fen(self),
            self.halfmove_clock,
            self.fullmove_number
        )
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct CastleRights {
    pub short: Option<File>,
    pub long: Option<File>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BoardParts {
    pieces: [BitBoard; 6],
    colors: [BitBoard; 2],
    occupied: BitBoard,
}

impl BoardParts {
    #[inline]
    pub(crate) fn from_board(board: &Board) -> Self {
        let white = board.colors[Color::White as usize];
        let black = board.colors[Color::Black as usize];
        Self {
            pieces: board.pieces,
            colors: board.colors,
            occupied: white | black,
        }
    }

    #[inline]
    pub(crate) fn color(&self, color: Color) -> BitBoard {
        self.colors[color as usize]
    }

    #[inline]
    pub(crate) fn piece(&self, piece: Piece) -> BitBoard {
        self.pieces[piece as usize]
    }

    #[inline]
    pub(crate) fn remove_piece(&mut self, color: Color, piece: Piece, square: Square) {
        let bit = square.bitboard();
        self.pieces[piece as usize] -= bit;
        self.colors[color as usize] -= bit;
        self.occupied -= bit;
    }

    #[inline]
    pub(crate) fn add_piece(&mut self, color: Color, piece: Piece, square: Square) {
        let bit = square.bitboard();
        self.pieces[piece as usize] |= bit;
        self.colors[color as usize] |= bit;
        self.occupied |= bit;
    }

    #[inline]
    pub(crate) fn attackers_for_piece(
        &self,
        target: Square,
        side: Color,
        piece: Piece,
    ) -> BitBoard {
        match piece {
            Piece::Pawn => get_pawn_attacks(target, !side) & self.piece(Piece::Pawn),
            Piece::Knight => get_knight_moves(target) & self.piece(Piece::Knight),
            Piece::Bishop => get_bishop_moves(target, self.occupied) & self.piece(Piece::Bishop),
            Piece::Rook => get_rook_moves(target, self.occupied) & self.piece(Piece::Rook),
            Piece::Queen => get_queen_moves(target, self.occupied) & self.piece(Piece::Queen),
            Piece::King => get_king_moves(target) & self.piece(Piece::King),
        }
    }

    #[inline]
    pub(crate) fn attackers_to(&self, square: Square, attacker: Color) -> BitBoard {
        let attacker_pieces = self.color(attacker);
        let pawns = get_pawn_attacks(square, !attacker) & self.piece(Piece::Pawn) & attacker_pieces;
        let knights = get_knight_moves(square) & self.piece(Piece::Knight) & attacker_pieces;
        let kings = get_king_moves(square) & self.piece(Piece::King) & attacker_pieces;
        let bishops = get_bishop_moves(square, self.occupied)
            & (self.piece(Piece::Bishop) | self.piece(Piece::Queen))
            & attacker_pieces;
        let rooks = get_rook_moves(square, self.occupied)
            & (self.piece(Piece::Rook) | self.piece(Piece::Queen))
            & attacker_pieces;
        pawns | knights | kings | bishops | rooks
    }

    #[inline]
    pub(crate) fn is_square_attacked(&self, square: Square, attacker: Color) -> bool {
        let attacker_pieces = self.color(attacker);
        if !(get_pawn_attacks(square, !attacker) & self.piece(Piece::Pawn) & attacker_pieces).is_empty() {
            return true;
        }
        if !(get_knight_moves(square) & self.piece(Piece::Knight) & attacker_pieces).is_empty() {
            return true;
        }
        if !(get_king_moves(square) & self.piece(Piece::King) & attacker_pieces).is_empty() {
            return true;
        }
        if !(get_bishop_moves(square, self.occupied)
            & (self.piece(Piece::Bishop) | self.piece(Piece::Queen))
            & attacker_pieces)
            .is_empty()
        {
            return true;
        }
        !(get_rook_moves(square, self.occupied)
            & (self.piece(Piece::Rook) | self.piece(Piece::Queen))
            & attacker_pieces)
            .is_empty()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PieceMoves {
    pub piece: Piece,
    pub from: Square,
    pub to: BitBoard,
}

impl PieceMoves {
    pub fn len(&self) -> usize {
        const PROMOTION_MASK: BitBoard = BitBoard(
            Rank::First.bitboard().0 | Rank::Eighth.bitboard().0,
        );
        if self.piece == Piece::Pawn {
            ((self.to & !PROMOTION_MASK).len() + (self.to & PROMOTION_MASK).len() * 4) as usize
        } else {
            self.to.len() as usize
        }
    }

}

impl IntoIterator for PieceMoves {
    type Item = Move;
    type IntoIter = PieceMovesIter;

    fn into_iter(self) -> Self::IntoIter {
        PieceMovesIter {
            moves: self,
            promotion: 0,
        }
    }
}

pub struct PieceMovesIter {
    moves: PieceMoves,
    promotion: u8,
}

impl Iterator for PieceMovesIter {
    type Item = Move;

    fn next(&mut self) -> Option<Self::Item> {
        let from = self.moves.from;
        let to = self.moves.to.next_square()?;
        let is_promotion = self.moves.piece == Piece::Pawn
            && matches!(to.rank(), Rank::First | Rank::Eighth);
        let promotion = if is_promotion {
            let promotion = match self.promotion {
                0 => Piece::Knight,
                1 => Piece::Bishop,
                2 => Piece::Rook,
                3 => Piece::Queen,
                _ => unreachable!(),
            };
            if self.promotion < 3 {
                self.promotion += 1;
            } else {
                self.promotion = 0;
                self.moves.to ^= to.bitboard();
            }
            Some(promotion)
        } else {
            self.moves.to ^= to.bitboard();
            None
        };
        Some(Move {
            from,
            to,
            promotion,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for PieceMovesIter {
    fn len(&self) -> usize {
        self.moves.len() - self.promotion as usize
    }
}

#[inline]
pub(crate) fn side_to_move(board: &Board) -> Color {
    board.side_to_move
}

#[inline]
pub(crate) fn status(board: &Board) -> GameStatus {
    if !has_legal_move(board) {
        return if checkers(board).is_empty() {
            GameStatus::Drawn
        } else {
            GameStatus::Won
        };
    }
    if board.halfmove_clock >= 100 {
        GameStatus::Drawn
    } else {
        GameStatus::Ongoing
    }
}

#[inline]
pub(crate) fn piece_on(board: &Board, square: Square) -> Option<Piece> {
    let square = square.bitboard();
    for piece in ALL_PIECES {
        if !(board.pieces[piece as usize] & square).is_empty() {
            return Some(piece);
        }
    }
    None
}

#[inline]
pub(crate) fn color_on(board: &Board, square: Square) -> Option<Color> {
    let square = square.bitboard();
    if !(board.colors[Color::White as usize] & square).is_empty() {
        Some(Color::White)
    } else if !(board.colors[Color::Black as usize] & square).is_empty() {
        Some(Color::Black)
    } else {
        None
    }
}

#[inline]
pub(crate) fn pieces(board: &Board, piece: Piece) -> BitBoard {
    board.pieces[piece as usize]
}

#[inline]
pub(crate) fn colors(board: &Board, color: Color) -> BitBoard {
    board.colors[color as usize]
}

#[inline]
pub(crate) fn colored_pieces(board: &Board, color: Color, piece: Piece) -> BitBoard {
    colors(board, color) & pieces(board, piece)
}

#[inline]
pub(crate) fn occupied(board: &Board) -> BitBoard {
    board.colors[Color::White as usize] | board.colors[Color::Black as usize]
}

#[inline]
pub(crate) fn en_passant(board: &Board) -> Option<File> {
    board.en_passant
}

#[inline]
pub(crate) fn castle_rights(board: &Board, color: Color) -> CastleRights {
    board.castle_rights[color as usize]
}

#[inline]
pub(crate) fn king(board: &Board, color: Color) -> Square {
    board.king_squares[color as usize]
}

#[inline]
pub(crate) fn checkers(board: &Board) -> BitBoard {
    attackers_to(board, king(board, board.side_to_move), !board.side_to_move)
}

#[inline]
pub(crate) fn halfmove_clock(board: &Board) -> u8 {
    board.halfmove_clock
}

#[inline]
pub(crate) fn hash(board: &Board) -> u64 {
    board.hash
}

#[inline]
pub(crate) fn hash_without_ep(board: &Board) -> u64 {
    match board.en_passant {
        Some(file) => board.hash ^ en_passant_key(file),
        None => board.hash,
    }
}

#[inline]
pub(crate) fn is_legal(board: &Board, mv: Move) -> bool {
    let side = board.side_to_move;
    let Some(moving_piece) = piece_on(board, mv.from) else {
        return false;
    };
    if color_on(board, mv.from) != Some(side) {
        return false;
    }

    let castle = castling_move(board, mv, side, moving_piece);
    if castle.is_none() && color_on(board, mv.to) == Some(side) {
        return false;
    }
    if piece_on(board, mv.to) == Some(Piece::King) {
        return false;
    }
    if !valid_promotion(moving_piece, mv) {
        return false;
    }
    if !is_pseudo_legal(board, mv, side, moving_piece, castle) {
        return false;
    }
    if let Some(castle) = castle
        && !is_castle_path_safe(board, side, castle)
    {
        return false;
    }

    !leaves_king_attacked(board, mv, side, moving_piece, castle)
}

#[inline]
pub(crate) fn play(board: &mut Board, mv: Move) {
    assert!(is_legal(board, mv), "illegal move: {mv}");
    play_unchecked(board, mv);
}

#[inline]
pub(crate) fn play_unchecked(board: &mut Board, mv: Move) {
    board.apply_move_unchecked(mv);
}

#[inline]
pub(crate) fn play_unchecked_with_piece(board: &mut Board, mv: Move, piece: Piece) {
    board.apply_move_unchecked_with_piece(mv, piece);
}

#[inline]
pub(crate) fn null_move(board: &Board) -> Option<Board> {
    if !checkers(board).is_empty() {
        return None;
    }
    let mut next = board.clone();
    next.halfmove_clock = next.halfmove_clock.saturating_add(1).min(100);
    if next.side_to_move == Color::Black {
        next.fullmove_number = next.fullmove_number.saturating_add(1);
    }
    next.set_en_passant(None);
    next.toggle_side_to_move();
    Some(next)
}

#[inline]
pub(crate) fn board_from_fen(fen: &str, chess960: bool) -> Result<Board, FenParseError> {
    Board::from_fen(fen, chess960)
}

#[inline]
pub(crate) fn generate_moves<F>(board: &Board, mut listener: F)
where
    F: FnMut(PieceMoves) -> bool,
{
    generate_legal_moves_filtered(board, |_, _| BitBoard::FULL, |moves| listener(moves))
}

#[inline]
pub(crate) fn generate_tactical_moves<F>(board: &Board, mut listener: F)
where
    F: FnMut(PieceMoves) -> bool,
{
    let side = board.side_to_move;
    let enemy = colors(board, !side);
    let promotion_rank = Rank::Eighth.relative_to(side).bitboard();
    let ep = board
        .en_passant
        .map(|file| Square::new(file, Rank::Sixth.relative_to(side)).bitboard())
        .unwrap_or(BitBoard::EMPTY);
    let pawn_targets = enemy | promotion_rank | ep;
    generate_legal_moves_filtered(
        board,
        |piece, _| {
            if piece == Piece::Pawn {
                pawn_targets
            } else {
                enemy
            }
        },
        |moves| listener(moves),
    )
}

fn piece_fen_letter(piece: Piece) -> char {
    match piece {
        Piece::Pawn => 'p',
        Piece::Knight => 'n',
        Piece::Bishop => 'b',
        Piece::Rook => 'r',
        Piece::Queen => 'q',
        Piece::King => 'k',
    }
}

fn color_fen_letter(color: Color) -> char {
    match color {
        Color::White => 'w',
        Color::Black => 'b',
    }
}

fn castling_fen(board: &Board) -> String {
    let mut out = String::new();
    push_castling_side(&mut out, board, Color::White, true);
    push_castling_side(&mut out, board, Color::White, false);
    push_castling_side(&mut out, board, Color::Black, true);
    push_castling_side(&mut out, board, Color::Black, false);
    if out.is_empty() {
        out.push('-');
    }
    out
}

fn push_castling_side(out: &mut String, board: &Board, color: Color, short: bool) {
    let rights = castle_rights(board, color);
    let Some(file) = (if short { rights.short } else { rights.long }) else {
        return;
    };
    let standard = match (color, short) {
        (Color::White, true) => file == File::H,
        (Color::White, false) => file == File::A,
        (Color::Black, true) => file == File::H,
        (Color::Black, false) => file == File::A,
    };
    let mut letter = if standard {
        if short { 'k' } else { 'q' }
    } else {
        file.to_string().chars().next().unwrap_or('-')
    };
    if color == Color::White {
        letter = letter.to_ascii_uppercase();
    }
    out.push(letter);
}

fn en_passant_fen(board: &Board) -> String {
    match board.en_passant {
        Some(file) => {
            let rank = Rank::Third.relative_to(!board.side_to_move);
            Square::new(file, rank).to_string()
        }
        None => "-".to_owned(),
    }
}

fn has_legal_move(board: &Board) -> bool {
    let mut found = false;
    generate_legal_moves_filtered(board, |_, _| BitBoard::FULL, |_| {
        found = true;
        true
    });
    found
}

fn generate_legal_moves_filtered<F, T>(board: &Board, target_filter: T, mut listener: F)
where
    F: FnMut(PieceMoves) -> bool,
    T: Fn(Piece, Square) -> BitBoard,
{
    let side = board.side_to_move;
    let state = MoveGenState::new(board);
    for piece in ALL_PIECES {
        for from in colored_pieces(board, side, piece) {
            let legal_targets =
                legal_targets_for_piece(board, &state, side, piece, from, target_filter(piece, from));
            if !legal_targets.is_empty()
                && listener(PieceMoves {
                    piece,
                    from,
                    to: legal_targets,
                })
            {
                return;
            }
        }
    }
}

struct MoveGenState {
    occupied: BitBoard,
    own: BitBoard,
    enemy: BitBoard,
    enemy_king: BitBoard,
    checker_count: u32,
    evasion_mask: BitBoard,
    pin_masks: PinMasks,
}

impl MoveGenState {
    fn new(board: &Board) -> Self {
        let side = board.side_to_move;
        let own = colors(board, side);
        let enemy = colors(board, !side);
        let occupied = own | enemy;
        let king_square = king(board, side);
        let king_state = KingSafetyState::new(board, side, king_square, occupied, own, enemy);
        Self {
            occupied,
            own,
            enemy,
            enemy_king: colored_pieces(board, !side, Piece::King),
            checker_count: king_state.checker_count,
            evasion_mask: king_state.evasion_mask,
            pin_masks: king_state.pin_masks,
        }
    }
}

struct KingSafetyState {
    checker_count: u32,
    evasion_mask: BitBoard,
    pin_masks: PinMasks,
}

impl KingSafetyState {
    #[inline]
    fn new(
        board: &Board,
        side: Color,
        king_square: Square,
        occupied: BitBoard,
        own: BitBoard,
        enemy: BitBoard,
    ) -> Self {
        let checkers = attackers_to(board, king_square, !side);
        let checker_count = checkers.len();
        let evasion_mask = if checker_count == 1 {
            let checker = checkers.next_square().expect("single checker exists");
            checker.bitboard() | between_squares(king_square, checker)
        } else {
            BitBoard::FULL
        };
        let pin_masks = if checker_count >= 2 {
            PinMasks::default()
        } else {
            build_pin_masks(board, king_square, occupied, own, enemy)
        };
        Self {
            checker_count,
            evasion_mask,
            pin_masks,
        }
    }
}

fn legal_targets_for_piece(
    board: &Board,
    state: &MoveGenState,
    side: Color,
    piece: Piece,
    from: Square,
    target_filter: BitBoard,
) -> BitBoard {
    if piece == Piece::King {
        return legal_king_targets(board, side, from, target_filter);
    }
    if state.checker_count >= 2 {
        return BitBoard::EMPTY;
    }

    let mut targets = (pseudo_targets(board, state, side, piece, from) - state.enemy_king) & target_filter;
    let ep_targets = en_passant_legal_targets(board, side, piece, from, targets);
    targets -= ep_targets;
    targets &= state.evasion_mask;
    let pin_mask = state.pin_masks.mask_for(from);
    if !pin_mask.is_empty() {
        targets &= pin_mask;
    }
    targets | ep_targets
}

fn legal_king_targets(board: &Board, side: Color, from: Square, target_filter: BitBoard) -> BitBoard {
    let mut legal = BitBoard::EMPTY;
    let own = colors(board, side);
    let enemy_king = colored_pieces(board, !side, Piece::King);
    for to in ((get_king_moves(from) - own) - enemy_king) & target_filter {
        if is_legal_king_step(board, side, from, to) {
            legal |= to.bitboard();
        }
    }
    for to in pseudo_castling_targets(board, side, from) & target_filter {
        let mv = Move {
            from,
            to,
            promotion: None,
        };
        if is_legal(board, mv) {
            legal |= to.bitboard();
        }
    }
    legal
}

fn is_legal_king_step(board: &Board, side: Color, from: Square, to: Square) -> bool {
    let mut parts = BoardParts::from_board(board);
    parts.remove_piece(side, Piece::King, from);
    if board.colors[(!side) as usize].has(to)
        && let Some(captured) = piece_on(board, to)
    {
        parts.remove_piece(!side, captured, to);
    }
    parts.add_piece(side, Piece::King, to);
    !parts.is_square_attacked(to, !side)
}

fn en_passant_legal_targets(
    board: &Board,
    side: Color,
    piece: Piece,
    from: Square,
    targets: BitBoard,
) -> BitBoard {
    if piece != Piece::Pawn {
        return BitBoard::EMPTY;
    }
    let Some(ep_file) = board.en_passant else {
        return BitBoard::EMPTY;
    };
    let ep = Square::new(ep_file, Rank::Sixth.relative_to(side));
    if !targets.has(ep) {
        return BitBoard::EMPTY;
    }
    let mv = Move {
        from,
        to: ep,
        promotion: None,
    };
    if is_legal(board, mv) {
        ep.bitboard()
    } else {
        BitBoard::EMPTY
    }
}

#[derive(Clone, Copy)]
struct PinMasks {
    squares: [Square; 8],
    masks: [BitBoard; 8],
    len: u8,
}

impl Default for PinMasks {
    fn default() -> Self {
        Self {
            squares: [Square::A1; 8],
            masks: [BitBoard::EMPTY; 8],
            len: 0,
        }
    }
}

impl PinMasks {
    #[inline]
    fn push(&mut self, square: Square, mask: BitBoard) {
        debug_assert!((self.len as usize) < self.squares.len());
        let index = self.len as usize;
        self.squares[index] = square;
        self.masks[index] = mask;
        self.len += 1;
    }

    #[inline]
    fn mask_for(&self, square: Square) -> BitBoard {
        for index in 0..self.len as usize {
            if self.squares[index] == square {
                return self.masks[index];
            }
        }
        BitBoard::EMPTY
    }
}

#[inline]
fn build_pin_masks(
    board: &Board,
    king_square: Square,
    occupied: BitBoard,
    own: BitBoard,
    enemy: BitBoard,
) -> PinMasks {
    let mut masks = PinMasks::default();
    let enemy_orthogonal_sliders =
        enemy & (pieces(board, Piece::Rook) | pieces(board, Piece::Queen));
    let enemy_diagonal_sliders =
        enemy & (pieces(board, Piece::Bishop) | pieces(board, Piece::Queen));
    let xray_occupied = occupied - own;
    let snipers = (get_rook_moves(king_square, xray_occupied) & enemy_orthogonal_sliders)
        | (get_bishop_moves(king_square, xray_occupied) & enemy_diagonal_sliders);
    for sniper in snipers {
        let blockers = between_squares(king_square, sniper) & occupied;
        if blockers.len() != 1 {
            continue;
        }
        let pinned = blockers
            .next_square()
            .expect("single blocker is present for pin candidate");
        if own.has(pinned) {
            masks.push(pinned, line_squares(king_square, sniper) - king_square.bitboard());
        }
    }
    masks
}

fn pseudo_targets(
    board: &Board,
    state: &MoveGenState,
    side: Color,
    piece: Piece,
    from: Square,
) -> BitBoard {
    match piece {
        Piece::Pawn => pseudo_pawn_targets(board, state, side, from),
        Piece::Knight => get_knight_moves(from) - state.own,
        Piece::Bishop => get_bishop_moves(from, state.occupied) - state.own,
        Piece::Rook => get_rook_moves(from, state.occupied) - state.own,
        Piece::Queen => get_queen_moves(from, state.occupied) - state.own,
        Piece::King => (get_king_moves(from) - state.own) | pseudo_castling_targets(board, side, from),
    }
}

fn pseudo_pawn_targets(board: &Board, state: &MoveGenState, side: Color, from: Square) -> BitBoard {
    let mut targets = BitBoard::EMPTY;
    let forward = if side == Color::White { 1 } else { -1 };
    let from_rank = from.rank() as i8;
    let one_rank = from_rank + forward;
    if (0..8).contains(&one_rank) {
        let one = Square::new(
            from.file(),
            Rank::try_index(one_rank as usize).expect("pawn single push rank is on board"),
        );
        if !state.occupied.has(one) {
            targets |= one.bitboard();
            let two_rank = from_rank + forward * 2;
            if from.rank() == Rank::Second.relative_to(side) && (0..8).contains(&two_rank) {
                let two = Square::new(
                    from.file(),
                    Rank::try_index(two_rank as usize).expect("pawn double push rank is on board"),
                );
                if !state.occupied.has(two) {
                    targets |= two.bitboard();
                }
            }
        }
    }

    let attacks = get_pawn_attacks(from, side);
    targets |= attacks & state.enemy;
    if let Some(ep_file) = board.en_passant {
        let ep = Square::new(ep_file, Rank::Sixth.relative_to(side));
        if attacks.has(ep) && en_passant_capture_square(board, side, ep).is_some()
        {
            targets |= ep.bitboard();
        }
    }
    targets
}

fn pseudo_castling_targets(board: &Board, side: Color, from: Square) -> BitBoard {
    if from != king(board, side) {
        return BitBoard::EMPTY;
    }
    let rank = Rank::First.relative_to(side);
    let rights = castle_rights(board, side);
    let mut targets = BitBoard::EMPTY;
    if let Some(file) = rights.short {
        targets |= Square::new(file, rank).bitboard();
    }
    if let Some(file) = rights.long {
        targets |= Square::new(file, rank).bitboard();
    }
    targets
}

#[derive(Clone, Copy)]
struct CastleMove {
    king_to: Square,
    rook: Square,
}

fn attackers_to(board: &Board, square: Square, attacker: Color) -> BitBoard {
    BoardParts::from_board(board).attackers_to(square, attacker)
}

fn is_square_attacked(board: &Board, square: Square, attacker: Color) -> bool {
    BoardParts::from_board(board).is_square_attacked(square, attacker)
}

fn valid_promotion(piece: Piece, mv: Move) -> bool {
    match mv.promotion {
        Some(Piece::Knight | Piece::Bishop | Piece::Rook | Piece::Queen) => {
            piece == Piece::Pawn && matches!(mv.to.rank(), Rank::First | Rank::Eighth)
        }
        Some(Piece::Pawn | Piece::King) => false,
        None => piece != Piece::Pawn || !matches!(mv.to.rank(), Rank::First | Rank::Eighth),
    }
}

fn is_pseudo_legal(
    board: &Board,
    mv: Move,
    side: Color,
    moving_piece: Piece,
    castle: Option<CastleMove>,
) -> bool {
    if castle.is_some() {
        return true;
    }

    let occupied = occupied(board);
    match moving_piece {
        Piece::Pawn => is_pseudo_legal_pawn_move(board, mv, side),
        Piece::Knight => get_knight_moves(mv.from).has(mv.to),
        Piece::Bishop => get_bishop_moves(mv.from, occupied).has(mv.to),
        Piece::Rook => get_rook_moves(mv.from, occupied).has(mv.to),
        Piece::Queen => get_queen_moves(mv.from, occupied).has(mv.to),
        Piece::King => get_king_moves(mv.from).has(mv.to),
    }
}

fn is_pseudo_legal_pawn_move(board: &Board, mv: Move, side: Color) -> bool {
    let from_file = mv.from.file() as i8;
    let to_file = mv.to.file() as i8;
    let from_rank = mv.from.rank() as i8;
    let to_rank = mv.to.rank() as i8;
    let forward = if side == Color::White { 1 } else { -1 };

    if to_file == from_file {
        if piece_on(board, mv.to).is_some() {
            return false;
        }
        if to_rank - from_rank == forward {
            return true;
        }
        if to_rank - from_rank == forward * 2
            && mv.from.rank() == Rank::Second.relative_to(side)
        {
            let between_rank = Rank::try_index((from_rank + forward) as usize)
                .expect("pawn double push intermediate rank is on board");
            let between = Square::new(mv.from.file(), between_rank);
            return piece_on(board, between).is_none();
        }
        return false;
    }

    if (to_file - from_file).abs() != 1 || to_rank - from_rank != forward {
        return false;
    }
    if color_on(board, mv.to) == Some(!side) {
        return true;
    }
    let ep_target = board
        .en_passant
        .map(|file| Square::new(file, Rank::Sixth.relative_to(side)));
    Some(mv.to) == ep_target && en_passant_capture_square(board, side, mv.to).is_some()
}

fn leaves_king_attacked(
    board: &Board,
    mv: Move,
    side: Color,
    moving_piece: Piece,
    castle: Option<CastleMove>,
) -> bool {
    let mut parts = BoardParts::from_board(board);

    let king_square = if let Some(castle) = castle {
        let rank = Rank::First.relative_to(side);
        let rook_to = Square::new(
            if castle.king_to.file() == File::G {
                File::F
            } else {
                File::D
            },
            rank,
        );
        parts.remove_piece(side, Piece::King, mv.from);
        parts.remove_piece(side, Piece::Rook, mv.to);
        parts.add_piece(side, Piece::King, castle.king_to);
        parts.add_piece(side, Piece::Rook, rook_to);
        castle.king_to
    } else {
        parts.remove_piece(side, moving_piece, mv.from);
        if let Some((captured_piece, captured_square)) =
            captured_piece_and_square(board, mv, side, moving_piece)
        {
            parts.remove_piece(!side, captured_piece, captured_square);
        }
        parts.add_piece(side, mv.promotion.unwrap_or(moving_piece), mv.to);
        if moving_piece == Piece::King {
            mv.to
        } else {
            king(board, side)
        }
    };

    parts.is_square_attacked(king_square, !side)
}

fn captured_piece_and_square(
    board: &Board,
    mv: Move,
    side: Color,
    moving_piece: Piece,
) -> Option<(Piece, Square)> {
    if moving_piece == Piece::Pawn
        && piece_on(board, mv.to).is_none()
        && board
            .en_passant
            .map(|file| Square::new(file, Rank::Sixth.relative_to(side)))
            == Some(mv.to)
    {
        let captured_square = en_passant_capture_square(board, side, mv.to)?;
        return Some((Piece::Pawn, captured_square));
    }

    piece_on(board, mv.to).map(|piece| (piece, mv.to))
}

fn en_passant_capture_square(board: &Board, side: Color, ep: Square) -> Option<Square> {
    let captured = Square::new(ep.file(), Rank::Fifth.relative_to(side));
    if piece_on(board, captured) == Some(Piece::Pawn) && color_on(board, captured) == Some(!side) {
        Some(captured)
    } else {
        None
    }
}

fn castling_move(
    board: &Board,
    mv: Move,
    side: Color,
    moving_piece: Piece,
) -> Option<CastleMove> {
    if moving_piece != Piece::King || color_on(board, mv.to) != Some(side) {
        return None;
    }
    let rank = Rank::First.relative_to(side);
    if mv.from != king(board, side) || mv.from.rank() != rank || mv.to.rank() != rank {
        return None;
    }

    let rights = castle_rights(board, side);
    let short = rights.short.map(|file| Square::new(file, rank)) == Some(mv.to);
    let long = rights.long.map(|file| Square::new(file, rank)) == Some(mv.to);
    if !short && !long {
        return None;
    }

    Some(CastleMove {
        king_to: Square::new(if short { File::G } else { File::C }, rank),
        rook: mv.to,
    })
}

fn is_castle_path_safe(board: &Board, side: Color, castle: CastleMove) -> bool {
    let enemy = !side;
    if is_square_attacked(board, king(board, side), enemy) {
        return false;
    }
    if !is_castle_path_clear(board, side, castle) {
        return false;
    }

    let king_from = king(board, side);
    let from = king_from as i8;
    let to = castle.king_to as i8;
    let step = if to > from { 1 } else { -1 };
    let mut square = from;
    loop {
        let Some(transit) = Square::try_index(square as usize) else {
            return false;
        };
        if is_square_attacked(board, transit, enemy) {
            return false;
        }
        if square == to {
            break;
        }
        square += step;
    }
    true
}

fn is_castle_path_clear(board: &Board, side: Color, castle: CastleMove) -> bool {
    let king_from = king(board, side);
    let rook_to = Square::new(
        if castle.king_to.file() == File::G {
            File::F
        } else {
            File::D
        },
        king_from.rank(),
    );
    castle_line_clear(board, king_from, castle.king_to, king_from, castle.rook)
        && castle_line_clear(board, castle.rook, rook_to, king_from, castle.rook)
}

fn castle_line_clear(
    board: &Board,
    from: Square,
    to: Square,
    king_from: Square,
    rook_from: Square,
) -> bool {
    let from_file = from.file() as i8;
    let to_file = to.file() as i8;
    if from_file == to_file {
        return true;
    }
    let step = if to_file > from_file { 1 } else { -1 };
    let rank = from.rank();
    let mut file_idx = from_file + step;
    while file_idx != to_file + step {
        let Some(file) = File::try_index(file_idx as usize) else {
            return false;
        };
        let square = Square::new(file, rank);
        if square != king_from && square != rook_from && piece_on(board, square).is_some() {
            return false;
        }
        file_idx += step;
    }
    true
}

fn parse_piece_placement(board: &mut Board, placement: &str) -> Result<(), FenParseError> {
    let mut rank = 7usize;
    let mut file = 0usize;
    let mut ranks = 1usize;

    for byte in placement.bytes() {
        match byte {
            b'/' => {
                if file != 8 || rank == 0 {
                    return Err(FenParseError);
                }
                rank -= 1;
                file = 0;
                ranks += 1;
            }
            b'1'..=b'8' => {
                file += usize::from(byte - b'0');
                if file > 8 {
                    return Err(FenParseError);
                }
            }
            b'p' | b'n' | b'b' | b'r' | b'q' | b'k' | b'P' | b'N' | b'B' | b'R' | b'Q' | b'K' => {
                if file >= 8 {
                    return Err(FenParseError);
                }
                let color = if byte.is_ascii_uppercase() {
                    Color::White
                } else {
                    Color::Black
                };
                let piece = parse_piece_letter(byte)?;
                let square = Square::new(
                    File::try_index(file).ok_or(FenParseError)?,
                    Rank::try_index(rank).ok_or(FenParseError)?,
                );
                board.add_piece(color, piece, square);
                file += 1;
            }
            _ => return Err(FenParseError),
        }
    }

    if ranks != 8 || file != 8 {
        return Err(FenParseError);
    }

    Ok(())
}

fn parse_piece_letter(byte: u8) -> Result<Piece, FenParseError> {
    match byte.to_ascii_lowercase() {
        b'p' => Ok(Piece::Pawn),
        b'n' => Ok(Piece::Knight),
        b'b' => Ok(Piece::Bishop),
        b'r' => Ok(Piece::Rook),
        b'q' => Ok(Piece::Queen),
        b'k' => Ok(Piece::King),
        _ => Err(FenParseError),
    }
}

fn parse_side_to_move(value: &str) -> Result<Color, FenParseError> {
    match value {
        "w" => Ok(Color::White),
        "b" => Ok(Color::Black),
        _ => Err(FenParseError),
    }
}

fn parse_castling_rights(
    board: &Board,
    value: &str,
    chess960: bool,
) -> Result<[CastleRights; 2], FenParseError> {
    if value == "-" {
        return Ok([CastleRights::default(); 2]);
    }
    if value.is_empty() || value.contains('-') {
        return Err(FenParseError);
    }

    let mut rights = [CastleRights::default(); 2];
    for byte in value.bytes() {
        let (color, file, explicit_side) = match byte {
            b'K' => (Color::White, castling_rook_file(board, Color::White, true, chess960)?, Some(true)),
            b'Q' => (Color::White, castling_rook_file(board, Color::White, false, chess960)?, Some(false)),
            b'k' => (Color::Black, castling_rook_file(board, Color::Black, true, chess960)?, Some(true)),
            b'q' => (Color::Black, castling_rook_file(board, Color::Black, false, chess960)?, Some(false)),
            b'A'..=b'H' => (Color::White, file_from_byte(byte)?, None),
            b'a'..=b'h' => (Color::Black, file_from_byte(byte)?, None),
            _ => return Err(FenParseError),
        };
        let short = match explicit_side {
            Some(short) => short,
            None => infer_castling_side(board, color, file)?,
        };
        let slot = if short {
            &mut rights[color as usize].short
        } else {
            &mut rights[color as usize].long
        };
        if slot.replace(file).is_some() {
            return Err(FenParseError);
        }
    }

    Ok(rights)
}

fn castling_rook_file(
    board: &Board,
    color: Color,
    short: bool,
    chess960: bool,
) -> Result<File, FenParseError> {
    if !chess960 {
        return Ok(if short { File::H } else { File::A });
    }

    let king_file = king(board, color).file();
    let rank = Rank::First.relative_to(color);
    let rooks = colored_pieces(board, color, Piece::Rook);
    let mut candidate = None;
    for rook in rooks {
        if rook.rank() != rank {
            continue;
        }
        if short && rook.file() > king_file {
            candidate = Some(candidate.map_or(rook.file(), |current: File| current.max(rook.file())));
        } else if !short && rook.file() < king_file {
            candidate = Some(candidate.map_or(rook.file(), |current: File| current.min(rook.file())));
        }
    }
    candidate.ok_or(FenParseError)
}

fn infer_castling_side(board: &Board, color: Color, rook_file: File) -> Result<bool, FenParseError> {
    let king_file = king(board, color).file();
    if rook_file > king_file {
        Ok(true)
    } else if rook_file < king_file {
        Ok(false)
    } else {
        Err(FenParseError)
    }
}

fn file_from_byte(byte: u8) -> Result<File, FenParseError> {
    let lower = byte.to_ascii_lowercase();
    if !(b'a'..=b'h').contains(&lower) {
        return Err(FenParseError);
    }
    File::try_index(usize::from(lower - b'a')).ok_or(FenParseError)
}

fn parse_en_passant(value: &str, side_to_move: Color) -> Result<Option<File>, FenParseError> {
    if value == "-" {
        return Ok(None);
    }
    let bytes = value.as_bytes();
    if bytes.len() != 2 {
        return Err(FenParseError);
    }
    let file = file_from_byte(bytes[0])?;
    let rank = match bytes[1] {
        b'3' => Rank::Third,
        b'6' => Rank::Sixth,
        _ => return Err(FenParseError),
    };
    if rank != Rank::Third.relative_to(!side_to_move) {
        return Err(FenParseError);
    }
    Ok(Some(file))
}

fn parse_halfmove_clock(value: &str) -> Result<u8, FenParseError> {
    value.parse().map_err(|_| FenParseError)
}

fn parse_fullmove_number(value: &str) -> Result<u16, FenParseError> {
    match value.parse() {
        Ok(0) => Err(FenParseError),
        Ok(fullmove_number) => Ok(fullmove_number),
        Err(_) => Err(FenParseError),
    }
}

fn compute_hash(board: &Board) -> u64 {
    let mut hash = 0u64;
    for color in [Color::White, Color::Black] {
        for piece in ALL_PIECES {
            for square in colored_pieces(board, color, piece) {
                hash ^= piece_key(color, piece, square);
            }
        }
    }
    if board.side_to_move == Color::Black {
        hash ^= side_key();
    }
    for color in [Color::White, Color::Black] {
        let rights = castle_rights(board, color);
        if let Some(file) = rights.short {
            hash ^= castle_key(color, 0, file);
        }
        if let Some(file) = rights.long {
            hash ^= castle_key(color, 1, file);
        }
    }
    if let Some(file) = board.en_passant {
        hash ^= en_passant_key(file);
    }
    hash
}

#[inline]
fn piece_key(color: Color, piece: Piece, square: Square) -> u64 {
    zobrist_key((color as u64 * 6 + piece as u64) * 64 + square as u64)
}

#[inline]
fn castle_key(color: Color, side: usize, file: File) -> u64 {
    zobrist_key(768 + color as u64 * 16 + side as u64 * 8 + file as u64)
}

#[inline]
fn en_passant_key(file: File) -> u64 {
    zobrist_key(800 + file as u64)
}

#[inline]
fn side_key() -> u64 {
    zobrist_key(808)
}

#[inline]
fn zobrist_key(index: u64) -> u64 {
    splitmix64(index.wrapping_add(0x9e37_79b9_7f4a_7c15))
}

#[inline]
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
