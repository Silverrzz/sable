pub(crate) use cozy_chess::{
    BitBoard, CastleRights, File, PieceMoves, Rank, Square, get_bishop_moves, get_king_moves,
    get_knight_moves, get_pawn_attacks, get_rook_moves,
};
pub use cozy_chess::{Board, Color, GameStatus, Move, Piece};

#[inline]
pub(crate) fn side_to_move(board: &Board) -> Color {
    board.side_to_move()
}

#[inline]
pub(crate) fn status(board: &Board) -> GameStatus {
    board.status()
}

#[inline]
pub(crate) fn piece_on(board: &Board, square: Square) -> Option<Piece> {
    board.piece_on(square)
}

#[inline]
pub(crate) fn color_on(board: &Board, square: Square) -> Option<Color> {
    board.color_on(square)
}

#[inline]
pub(crate) fn pieces(board: &Board, piece: Piece) -> BitBoard {
    board.pieces(piece)
}

#[inline]
pub(crate) fn colors(board: &Board, color: Color) -> BitBoard {
    board.colors(color)
}

#[inline]
pub(crate) fn colored_pieces(board: &Board, color: Color, piece: Piece) -> BitBoard {
    board.colored_pieces(color, piece)
}

#[inline]
pub(crate) fn occupied(board: &Board) -> BitBoard {
    board.occupied()
}

#[inline]
pub(crate) fn en_passant(board: &Board) -> Option<File> {
    board.en_passant()
}

#[inline]
pub(crate) fn castle_rights(board: &Board, color: Color) -> CastleRights {
    *board.castle_rights(color)
}

#[inline]
pub(crate) fn king(board: &Board, color: Color) -> Square {
    board.king(color)
}

#[inline]
pub(crate) fn checkers(board: &Board) -> BitBoard {
    board.checkers()
}

#[inline]
pub(crate) fn halfmove_clock(board: &Board) -> u8 {
    board.halfmove_clock()
}

#[inline]
pub(crate) fn hash(board: &Board) -> u64 {
    board.hash()
}

#[inline]
pub(crate) fn hash_without_ep(board: &Board) -> u64 {
    board.hash_without_ep()
}

#[inline]
pub(crate) fn is_legal(board: &Board, mv: Move) -> bool {
    board.is_legal(mv)
}

#[inline]
pub(crate) fn play(board: &mut Board, mv: Move) {
    board.play(mv);
}

#[inline]
pub(crate) fn play_unchecked(board: &mut Board, mv: Move) {
    board.play_unchecked(mv);
}

#[inline]
pub(crate) fn null_move(board: &Board) -> Option<Board> {
    board.null_move()
}

#[inline]
pub(crate) fn board_from_fen(fen: &str, chess960: bool) -> Result<Board, cozy_chess::FenParseError> {
    Board::from_fen(fen, chess960)
}

#[inline]
pub(crate) fn generate_moves<F>(board: &Board, listener: F)
where
    F: FnMut(PieceMoves) -> bool,
{
    board.generate_moves(listener);
}
