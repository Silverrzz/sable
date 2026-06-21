use anyhow::Result;
use sable_engine::Engine;
use std::io::{self, Write};

pub(super) fn write_uci_identification(stdout: &mut io::Stdout, engine: &Engine) -> Result<()> {
    let release_id = option_env!("SABLER_RELEASE_ID").unwrap_or("dev");
    let git_commit = option_env!("SABLER_GIT_COMMIT").unwrap_or("unknown");
    let target = option_env!("TARGET").unwrap_or(std::env::consts::ARCH);
    let profile = option_env!("PROFILE").unwrap_or("unknown");

    writeln!(stdout, "id name Sable {}", env!("CARGO_PKG_VERSION"))?;
    writeln!(stdout, "id author Ellie Fulterer")?;
    writeln!(stdout, "option name Hash type spin default 16 min 1 max 32768")?;
    writeln!(stdout, "option name Threads type spin default 1 min 1 max 256")?;
    writeln!(stdout, "option name Ponder type check default false")?;
    writeln!(stdout, "option name MultiPV type spin default 1 min 1 max 256")?;
    writeln!(stdout, "option name UCI_Chess960 type check default false")?;
    writeln!(stdout, "option name UCI_ShowWDL type check default false")?;
    writeln!(
        stdout,
        "option name Move Overhead type spin default 100 min 0 max 10000"
    )?;
    writeln!(stdout, "option name Clear Hash type button")?;
    writeln!(
        stdout,
        "option name Eval type combo default {} var hce var nnue",
        engine.eval_mode_option_value().as_uci()
    )?;
    write_eval_file_option(stdout, engine)?;
    writeln!(
        stdout,
        "info string build release_id {release_id} commit {git_commit} target {target} profile {profile}"
    )?;
    for warning in engine.startup_warnings() {
        writeln!(stdout, "info string warning {warning}")?;
    }
    writeln!(stdout, "uciok")?;
    stdout.flush()?;
    Ok(())
}

fn write_eval_file_option(stdout: &mut io::Stdout, engine: &Engine) -> Result<()> {
    if let Some(default_eval) = engine.eval_file_option_value() {
        writeln!(
            stdout,
            "option name Eval File type string default {}",
            default_eval
        )?;
    } else {
        writeln!(stdout, "option name Eval File type string default")?;
    }
    Ok(())
}

pub(super) fn format_uci_info(info: &sable_engine::SearchInfo, show_wdl: bool) -> String {
    let elapsed_ms = info.time_ms.unwrap_or(0);
    let depth = info.depth.unwrap_or(1);
    let seldepth = info.seldepth.unwrap_or(depth);
    let score = format_uci_score(info);
    let multi_pv = info
        .multi_pv
        .map(|idx| format!(" multipv {idx}"))
        .unwrap_or_default();
    let wdl = if show_wdl {
        let (win, draw, loss) = format_wdl(info);
        format!(" wdl {win} {draw} {loss}")
    } else {
        String::new()
    };
    let nodes = info.nodes.unwrap_or(0);
    let nps = info.nps.unwrap_or(0);
    let hashfull = info.hashfull.unwrap_or(0);
    let pv = info.pv_uci.join(" ");
    let pv = if pv.is_empty() {
        String::new()
    } else {
        format!(" pv {pv}")
    };
    format!(
        "info depth {depth} seldepth {seldepth}{multi_pv} {score}{wdl} nodes {nodes} nps {nps} tbhits 0 hashfull {hashfull} time {elapsed_ms}{pv}",
    )
}

fn format_uci_score(info: &sable_engine::SearchInfo) -> String {
    if let Some(score_mate) = info.score_mate {
        format!("score mate {score_mate}")
    } else {
        format!("score cp {}", info.score_cp.unwrap_or(0))
    }
}

pub(super) fn format_static_eval_score(eval: &sable_engine::StaticEval) -> String {
    if let Some(score_mate) = eval.score_mate {
        format!("score mate {score_mate}")
    } else {
        format!("score cp {}", eval.score_cp)
    }
}

pub(super) fn eval_source_label(source: sable_engine::StaticEvalSource) -> &'static str {
    match source {
        sable_engine::StaticEvalSource::Nnue => "nnue",
        sable_engine::StaticEvalSource::Hce => "hce",
        sable_engine::StaticEvalSource::Terminal => "terminal",
    }
}

fn format_wdl(info: &sable_engine::SearchInfo) -> (u32, u32, u32) {
    if let Some(mate) = info.score_mate {
        return if mate > 0 {
            (1000, 0, 0)
        } else if mate < 0 {
            (0, 0, 1000)
        } else {
            (0, 1000, 0)
        };
    }
    let cp = info.score_cp.unwrap_or(0).clamp(-2000, 2000) as f64;
    let decisive = 1.0 / (1.0 + (-cp / 180.0).exp());
    let draw = (350.0 * (-cp.abs() / 400.0).exp()).round() as u32;
    let remaining = 1000_u32.saturating_sub(draw);
    let win = ((remaining as f64) * decisive).round() as u32;
    let loss = 1000_u32.saturating_sub(draw).saturating_sub(win);
    (win, draw, loss)
}

fn piece_letter(piece: sable_engine::Piece, color: sable_engine::Color) -> char {
    let ch = match piece {
        sable_engine::Piece::Pawn => 'p',
        sable_engine::Piece::Knight => 'n',
        sable_engine::Piece::Bishop => 'b',
        sable_engine::Piece::Rook => 'r',
        sable_engine::Piece::Queen => 'q',
        sable_engine::Piece::King => 'k',
    };
    if color == sable_engine::Color::White {
        ch.to_ascii_uppercase()
    } else {
        ch
    }
}

fn piece_material_pawns(piece: sable_engine::Piece) -> f32 {
    match piece {
        sable_engine::Piece::Pawn => 1.00,
        sable_engine::Piece::Knight => 3.20,
        sable_engine::Piece::Bishop => 3.30,
        sable_engine::Piece::Rook => 5.00,
        sable_engine::Piece::Queen => 9.00,
        sable_engine::Piece::King => 0.0,
    }
}

pub(super) fn format_verbose_eval(veval: &sable_engine::VerboseEval) -> String {
    let mut out = String::new();
    let sep = "+-------+-------+-------+-------+-------+-------+-------+-------+\n";
    let header = if veval.piece_contributions.is_empty() {
        " Piece values (material base):\n"
    } else {
        " NNUE derived piece values:\n"
    };
    out.push_str(&format!("\n{header}"));

    for rank in (0..8u8).rev() {
        out.push_str(sep);
        push_piece_rank(&mut out, veval, rank);
        push_value_rank(&mut out, veval, rank);
    }
    out.push_str(sep);
    push_verbose_eval_summary(&mut out, veval);
    out
}

fn push_piece_rank(out: &mut String, veval: &sable_engine::VerboseEval, rank: u8) {
    out.push('|');
    for file in 0..8u8 {
        let sq = (file + rank * 8) as usize;
        match &veval.squares[sq] {
            Some(p) => {
                let ch = piece_letter(p.piece, p.color);
                out.push_str(&format!("   {ch}   |"));
            }
            None => out.push_str("       |"),
        }
    }
    out.push('\n');
}

fn push_value_rank(out: &mut String, veval: &sable_engine::VerboseEval, rank: u8) {
    use sable_engine::{Color, Piece};

    out.push('|');
    for file in 0..8u8 {
        let sq_idx = (file + rank * 8) as usize;
        match &veval.squares[sq_idx] {
            Some(p) if p.piece != Piece::King => {
                let value_pawns = verbose_piece_value_pawns(veval, sq_idx, p.piece, p.color);
                out.push_str(&format!(" {value_pawns:+.2} |"));
            }
            _ => out.push_str("       |"),
        }
    }
    out.push('\n');

    fn verbose_piece_value_pawns(
        veval: &sable_engine::VerboseEval,
        sq_idx: usize,
        piece: Piece,
        color: Color,
    ) -> f32 {
        if let Some(contrib) = veval
            .piece_contributions
            .iter()
            .find(|c| c.square as usize == sq_idx)
        {
            contrib.score_white_cp as f32 / 100.0
        } else {
            let base = piece_material_pawns(piece);
            if color == Color::White { base } else { -base }
        }
    }
}

fn push_verbose_eval_summary(out: &mut String, veval: &sable_engine::VerboseEval) {
    use sable_engine::{Color, StaticEvalSource};

    let wk_file = (b'a' + veval.white_king_square % 8) as char;
    let wk_rank = (b'1' + veval.white_king_square / 8) as char;
    let bk_file = (b'a' + veval.black_king_square % 8) as char;
    let bk_rank = (b'1' + veval.black_king_square / 8) as char;
    let stm = match veval.side_to_move {
        Color::White => "White",
        Color::Black => "Black",
    };
    out.push_str(&format!(
        "\n King squares: white {wk_file}{wk_rank} (bucket {}), black {bk_file}{bk_rank} (bucket {}) -- {stm} to move\n",
        veval.white_king_square,
        veval.black_king_square ^ 56,
    ));

    let mat_pawns = veval.material_score_white_cp as f32 / 100.0;
    out.push_str(&format!(
        "\n Material balance    {mat_pawns:+.2} (white side)\n"
    ));

    if let Some(nnue_cp) = veval.nnue_score_white_cp {
        let nnue_pawns = nnue_cp as f32 / 100.0;
        out.push_str(&format!(
            " NNUE evaluation     {nnue_pawns:+.2} (white side)\n"
        ));
    }

    let final_white_cp = match veval.side_to_move {
        Color::White => veval.final_score_stm_cp,
        Color::Black => -veval.final_score_stm_cp,
    };
    let final_pawns = final_white_cp as f32 / 100.0;
    let source_str = match veval.source {
        StaticEvalSource::Nnue => "with NNUE",
        StaticEvalSource::Hce => "with HCE",
        StaticEvalSource::Terminal => "terminal",
    };
    out.push_str(&format!(
        " Final evaluation    {final_pawns:+.2} (white side) [{source_str}]\n\n"
    ));
}
