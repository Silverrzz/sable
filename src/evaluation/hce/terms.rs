use crate::{Color, Piece};
use cozy_chess::{get_bishop_moves, get_knight_moves, get_rook_moves};

use super::{
    HCE_BISHOP_MOBILITY_EG, HCE_BISHOP_MOBILITY_MG, HCE_BISHOP_OUTPOST_EG,
    HCE_BISHOP_OUTPOST_MG, HCE_BISHOP_PAIR_EG, HCE_BISHOP_PAIR_MG, HCE_CONNECTED_PAWN_EG,
    HCE_CONNECTED_PAWN_MG, HCE_DOUBLED_PAWN_EG, HCE_DOUBLED_PAWN_MG,
    HCE_ISOLATED_PAWN_EG, HCE_ISOLATED_PAWN_MG, HCE_KING_CLOSE_SHIELD_MG,
    HCE_KING_FAR_SHIELD_MG, HCE_KING_MISSING_SHIELD_MG, HCE_KING_STORM_PENALTY_MG,
    HCE_KNIGHT_MOBILITY_EG, HCE_KNIGHT_MOBILITY_MG, HCE_KNIGHT_OUTPOST_EG,
    HCE_KNIGHT_OUTPOST_MG, HCE_PASSED_PAWN_EG, HCE_PASSED_PAWN_MG,
    HCE_QUEEN_MOBILITY_EG, HCE_QUEEN_MOBILITY_MG, HCE_ROOK_MOBILITY_EG,
    HCE_ROOK_MOBILITY_MG, HCE_ROOK_OPEN_FILE_EG, HCE_ROOK_OPEN_FILE_MG,
    HCE_ROOK_SEMI_OPEN_FILE_EG, HCE_ROOK_SEMI_OPEN_FILE_MG, HCE_ROOK_SEVENTH_EG,
    HCE_ROOK_SEVENTH_MG,
    geometry::{
        is_outpost, is_passed_pawn, relative_rank, square_file, square_mask, square_rank,
    },
    types::{HceInfo, HceScore},
};

#[inline]
pub(super) fn side_hce_terms(info: &HceInfo, side: Color) -> HceScore {
    let mut score = HceScore::default();
    score.add_score(pawn_structure_terms(info, side));
    score.add_score(minor_piece_terms(info, side));
    score.add_score(activity_terms(info, side));
    score.add_score(rook_terms(info, side));
    score.add_score(king_safety_terms(info, side));
    score
}

#[inline]
fn pawn_structure_terms(info: &HceInfo, side: Color) -> HceScore {
    let side_idx = side as usize;
    let pawns = info.piece_color(side, Piece::Pawn);
    let enemy_pawns = info.piece_color(!side, Piece::Pawn);
    let support = info.pawn_attacks[side_idx];
    let own_pawn_files = &info.pawn_files[side_idx];
    let mut score = HceScore::default();

    for file_pawns in own_pawn_files.iter().take(8) {
        let count = file_pawns.len() as i32;
        if count > 1 {
            score.add(
                HCE_DOUBLED_PAWN_MG * (count - 1),
                HCE_DOUBLED_PAWN_EG * (count - 1),
            );
        }
    }

    for square in pawns {
        let file = square_file(square);
        if (file == 0 || own_pawn_files[file - 1].is_empty())
            && (file == 7 || own_pawn_files[file + 1].is_empty())
        {
            score.add(HCE_ISOLATED_PAWN_MG, HCE_ISOLATED_PAWN_EG);
        }

        if support.has(square) {
            score.add(HCE_CONNECTED_PAWN_MG, HCE_CONNECTED_PAWN_EG);
        }

        if is_passed_pawn(square, side, enemy_pawns) {
            let rank = relative_rank(square, side);
            score.add(HCE_PASSED_PAWN_MG[rank], HCE_PASSED_PAWN_EG[rank]);
        }
    }

    score
}

#[inline]
fn minor_piece_terms(info: &HceInfo, side: Color) -> HceScore {
    let mut score = HceScore::default();
    let side_idx = side as usize;
    let bishops = info.piece_color(side, Piece::Bishop);
    let knights = info.piece_color(side, Piece::Knight);

    if bishops.len() >= 2 {
        score.add(HCE_BISHOP_PAIR_MG, HCE_BISHOP_PAIR_EG);
    }

    let support = info.pawn_attacks[side_idx];
    let enemy_pawn_attacks = info.pawn_attacks[(!side) as usize];
    for square in knights {
        if is_outpost(square, side, support, enemy_pawn_attacks) {
            score.add(HCE_KNIGHT_OUTPOST_MG, HCE_KNIGHT_OUTPOST_EG);
        }
    }
    for square in bishops {
        if is_outpost(square, side, support, enemy_pawn_attacks) {
            score.add(HCE_BISHOP_OUTPOST_MG, HCE_BISHOP_OUTPOST_EG);
        }
    }

    score
}

#[inline]
fn activity_terms(info: &HceInfo, side: Color) -> HceScore {
    let mut score = HceScore::default();
    let side_idx = side as usize;
    let enemy_idx = (!side) as usize;
    let own = info.color(side);
    let mobility_area = !own & !info.pawn_attacks[enemy_idx];
    let enemy_king_ring = info.king_zones[enemy_idx];
    let mut king_attack_units = ((info.pawn_attacks[side_idx] & enemy_king_ring).len() as i32) * 2;
    let mut king_attackers = 0;

    for square in info.piece_color(side, Piece::Knight) {
        let attacks = get_knight_moves(square);
        let mobility = (attacks & mobility_area).len() as i32;
        score.add(
            mobility * HCE_KNIGHT_MOBILITY_MG,
            mobility * HCE_KNIGHT_MOBILITY_EG,
        );
        let ring_hits = (attacks & enemy_king_ring).len() as i32;
        if ring_hits > 0 {
            king_attackers += 1;
            king_attack_units += ring_hits * 3;
        }
    }

    for square in info.piece_color(side, Piece::Bishop) {
        let attacks = get_bishop_moves(square, info.occupied);
        let mobility = (attacks & mobility_area).len() as i32;
        score.add(
            mobility * HCE_BISHOP_MOBILITY_MG,
            mobility * HCE_BISHOP_MOBILITY_EG,
        );
        let ring_hits = (attacks & enemy_king_ring).len() as i32;
        if ring_hits > 0 {
            king_attackers += 1;
            king_attack_units += ring_hits * 3;
        }
    }

    for square in info.piece_color(side, Piece::Rook) {
        let attacks = get_rook_moves(square, info.occupied);
        let mobility = (attacks & mobility_area).len() as i32;
        score.add(
            mobility * HCE_ROOK_MOBILITY_MG,
            mobility * HCE_ROOK_MOBILITY_EG,
        );
        let ring_hits = (attacks & enemy_king_ring).len() as i32;
        if ring_hits > 0 {
            king_attackers += 1;
            king_attack_units += ring_hits * 4;
        }
    }

    for square in info.piece_color(side, Piece::Queen) {
        let attacks =
            get_bishop_moves(square, info.occupied) | get_rook_moves(square, info.occupied);
        let mobility = (attacks & mobility_area).len() as i32;
        score.add(
            mobility * HCE_QUEEN_MOBILITY_MG,
            mobility * HCE_QUEEN_MOBILITY_EG,
        );
        let ring_hits = (attacks & enemy_king_ring).len() as i32;
        if ring_hits > 0 {
            king_attackers += 1;
            king_attack_units += ring_hits * 6;
        }
    }

    if king_attackers > 0 {
        score.mg += king_attack_bonus(king_attackers, king_attack_units);
    }

    score
}

#[inline]
fn rook_terms(info: &HceInfo, side: Color) -> HceScore {
    let mut score = HceScore::default();
    let side_idx = side as usize;
    let enemy_idx = (!side) as usize;
    let own_pawn_files = &info.pawn_files[side_idx];
    let enemy_pawn_files = &info.pawn_files[enemy_idx];

    for square in info.piece_color(side, Piece::Rook) {
        let file = square_file(square);
        if own_pawn_files[file].is_empty() {
            if enemy_pawn_files[file].is_empty() {
                score.add(HCE_ROOK_OPEN_FILE_MG, HCE_ROOK_OPEN_FILE_EG);
            } else {
                score.add(HCE_ROOK_SEMI_OPEN_FILE_MG, HCE_ROOK_SEMI_OPEN_FILE_EG);
            }
        }
        if relative_rank(square, side) == 6 {
            score.add(HCE_ROOK_SEVENTH_MG, HCE_ROOK_SEVENTH_EG);
        }
    }

    score
}

#[inline]
fn king_safety_terms(info: &HceInfo, side: Color) -> HceScore {
    let enemy_idx = (!side) as usize;
    let Some(king) = info.king_square(side) else {
        return HceScore::default();
    };
    if relative_rank(king, side) > 2 {
        return HceScore::default();
    }

    let mut score = HceScore::default();
    let king_file = square_file(king);
    let king_rank = square_rank(king) as i32;
    let forward = if side == Color::White { 1 } else { -1 };
    let own_pawns = info.piece_color(side, Piece::Pawn);
    let enemy_pawn_files = &info.pawn_files[enemy_idx];

    let start_file = king_file.saturating_sub(1);
    let end_file = (king_file + 1).min(7);
    let close_rank = king_rank + forward;
    let far_rank = close_rank + forward;
    for file in start_file..=end_file {
        if (0..8).contains(&close_rank) {
            let close = square_mask(file, close_rank as usize);
            if !(own_pawns & close).is_empty() {
                score.mg += HCE_KING_CLOSE_SHIELD_MG;
            } else {
                score.mg += HCE_KING_MISSING_SHIELD_MG;
            }
        }
        if (0..8).contains(&far_rank)
            && !(own_pawns & square_mask(file, far_rank as usize)).is_empty()
        {
            score.mg += HCE_KING_FAR_SHIELD_MG;
        }

        for pawn in enemy_pawn_files[file] {
            let distance = (square_rank(pawn) as i32 - king_rank).abs() as usize;
            if distance < HCE_KING_STORM_PENALTY_MG.len() {
                score.mg -= HCE_KING_STORM_PENALTY_MG[distance];
            }
        }
    }

    score
}

#[inline(always)]
fn king_attack_bonus(attackers: i32, units: i32) -> i32 {
    let pressure = attackers * units;
    (pressure + (units * units) / 4).min(180)
}
