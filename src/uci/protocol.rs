use anyhow::Result;
use sable_engine::{
    Engine, SPSA_UCI_OPTIONS_ENABLED, embedded_eval_hash, embedded_eval_label, has_embedded_eval,
    spsa_parameters,
};
use std::io::{self, Write};

pub(super) fn write_uci_identification(stdout: &mut io::Stdout, engine: &Engine) -> Result<()> {
    let release_id = option_env!("SABLE_RELEASE_ID").unwrap_or("dev");
    let git_commit = option_env!("SABLE_GIT_COMMIT").unwrap_or("unknown");
    let target = option_env!("TARGET").unwrap_or(std::env::consts::ARCH);
    let profile = option_env!("PROFILE").unwrap_or("unknown");

    writeln!(stdout, "id name Sable {}", env!("CARGO_PKG_VERSION"))?;
    writeln!(stdout, "id author Ellie Fulterer")?;
    writeln!(
        stdout,
        "option name Hash type spin default 16 min 1 max 32768"
    )?;
    writeln!(
        stdout,
        "option name Threads type spin default 1 min 1 max 256"
    )?;
    writeln!(stdout, "option name Ponder type check default false")?;
    writeln!(
        stdout,
        "option name MultiPV type spin default 1 min 1 max 256"
    )?;
    writeln!(stdout, "option name UseSoftNodes type check default false")?;
    writeln!(stdout, "option name UCI_Chess960 type check default false")?;
    writeln!(stdout, "option name UCI_ShowWDL type check default false")?;
    writeln!(
        stdout,
        "option name UCI_ShowUncertainty type check default false"
    )?;
    if SPSA_UCI_OPTIONS_ENABLED {
        for parameter in spsa_parameters() {
            writeln!(
                stdout,
                "option name {} type spin default {} min {} max {}",
                parameter.name, parameter.default, parameter.min, parameter.max
            )?;
        }
    }
    writeln!(
        stdout,
        "option name Move Overhead type spin default 100 min 0 max 10000"
    )?;
    writeln!(stdout, "option name Clear Hash type button")?;
    write_eval_file_option(stdout, engine)?;
    writeln!(
        stdout,
        "info string build release_id {release_id} commit {git_commit} target {target} profile {profile}"
    )?;
    write_eval_identity(stdout, engine)?;
    for warning in engine.startup_warnings() {
        writeln!(stdout, "info string warning {warning}")?;
    }
    writeln!(stdout, "uciok")?;
    stdout.flush()?;
    Ok(())
}

fn write_eval_identity(stdout: &mut io::Stdout, engine: &Engine) -> Result<()> {
    let embedded = if has_embedded_eval() { "true" } else { "false" };
    let source = embedded_eval_label().unwrap_or("none");
    let hash = embedded_eval_hash().unwrap_or("none");
    let arch = engine
        .loaded_nnue_architecture_id()
        .map(|id| id.as_str())
        .unwrap_or("none");
    writeln!(
        stdout,
        "info string eval embedded {embedded} source {source} hash {hash} arch {arch}"
    )?;
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

pub(crate) fn format_uci_info(
    engine: &Engine,
    info: &sable_engine::SearchInfo,
    show_wdl: bool,
    show_uncertainty: bool,
) -> String {
    let elapsed_ms = info.time_ms;
    let depth = info.depth;
    let seldepth = info.seldepth;
    let score = format_uci_score(info);
    let multi_pv = info
        .multi_pv
        .map(|idx| format!(" multipv {idx}"))
        .unwrap_or_default();
    let leaf_output = (show_wdl || show_uncertainty)
        .then(|| engine.pv_leaf_output(&info.pv))
        .flatten();
    let wdl = show_wdl
        .then(|| format_wdl(info, leaf_output))
        .flatten()
        .map(|[win, draw, loss]| format!(" wdl {win} {draw} {loss}"))
        .unwrap_or_default();
    let uncertainty = show_uncertainty
        .then_some(leaf_output)
        .flatten()
        .and_then(|output| match output {
            sable_engine::PvLeafOutput::Nnue(output) => Some(output.uncertainty_cp()),
            sable_engine::PvLeafOutput::Terminal(_) => None,
        })
        .map(|uncertainty| format!(" unc {uncertainty}"))
        .unwrap_or_default();
    let nodes = info.nodes;
    let nps = info.nps;
    let hashfull = info.hashfull;
    let pv = engine.format_uci_pv(&info.pv).join(" ");
    let pv = if pv.is_empty() {
        String::new()
    } else {
        format!(" pv {pv}")
    };
    format!(
        "info depth {depth} seldepth {seldepth}{multi_pv} {score}{wdl}{uncertainty} nodes {nodes} nps {nps} tbhits 0 hashfull {hashfull} time {elapsed_ms}{pv}",
    )
}

fn format_uci_score(info: &sable_engine::SearchInfo) -> String {
    if let Some(score_mate) = info.score_mate {
        format!("score mate {score_mate}")
    } else {
        format!("score cp {}", normalize_uci_cp(info.score_cp))
    }
}

pub(super) fn format_static_eval_score(eval: &sable_engine::StaticEval) -> String {
    if let Some(score_mate) = eval.score_mate {
        format!("score mate {score_mate}")
    } else {
        format!("score cp {}", normalize_uci_cp(eval.score_cp))
    }
}

fn normalize_uci_cp(score: i32) -> i32 {
    let magnitude = score.unsigned_abs();
    if magnitude <= 100 {
        return score;
    }
    score.signum() * ((f64::from(magnitude) * 100.0).sqrt().round() as i32)
}

pub(super) fn eval_source_label(source: sable_engine::StaticEvalSource) -> &'static str {
    match source {
        sable_engine::StaticEvalSource::Nnue => "nnue",
        sable_engine::StaticEvalSource::Terminal => "terminal",
    }
}

fn format_wdl(
    info: &sable_engine::SearchInfo,
    leaf_output: Option<sable_engine::PvLeafOutput>,
) -> Option<[u32; 3]> {
    if let Some(mate) = info.score_mate {
        return Some(if mate > 0 {
            [1000, 0, 0]
        } else if mate < 0 {
            [0, 0, 1000]
        } else {
            [0, 1000, 0]
        });
    }
    leaf_output.map(|output| match output {
        sable_engine::PvLeafOutput::Nnue(output) => output.wdl_permille(),
        sable_engine::PvLeafOutput::Terminal(wdl) => wdl,
    })
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

pub(super) fn format_verbose_eval(veval: &sable_engine::VerboseEval) -> String {
    let mut out = String::new();
    let sep = "+-------+-------+-------+-------+-------+-------+-------+-------+\n";
    let header = if veval.piece_contributions.is_empty() {
        " NNUE remove-piece values unavailable:\n"
    } else {
        " NNUE piece values (current - without piece):\n"
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
    use sable_engine::Piece;

    out.push('|');
    for file in 0..8u8 {
        let sq_idx = (file + rank * 8) as usize;
        match &veval.squares[sq_idx] {
            Some(p) if p.piece != Piece::King => {
                if let Some(value_pawns) = verbose_piece_value_pawns(veval, sq_idx) {
                    out.push_str(&format!(" {value_pawns:+.2} |"));
                } else {
                    out.push_str("       |");
                }
            }
            _ => out.push_str("       |"),
        }
    }
    out.push('\n');

    fn verbose_piece_value_pawns(veval: &sable_engine::VerboseEval, sq_idx: usize) -> Option<f32> {
        veval
            .piece_contributions
            .iter()
            .find(|c| c.square as usize == sq_idx)
            .map(|contrib| contrib.score_white_cp as f32 / 100.0)
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

    if let Some(nnue_output) = veval.nnue_output {
        let [win, draw, loss] = nnue_output.wdl_permille();
        out.push_str(&format!(
            " NNUE uncertainty    +/-{} cp (estimated error range, side to move)\n",
            nnue_output.uncertainty_cp()
        ));
        out.push_str(&format!(
            " NNUE WDL            {win} {draw} {loss} (side to move)\n"
        ));
    }

    let final_white_cp = match veval.side_to_move {
        Color::White => veval.final_score_stm_cp,
        Color::Black => -veval.final_score_stm_cp,
    };
    let final_pawns = final_white_cp as f32 / 100.0;
    let source_str = match veval.source {
        StaticEvalSource::Nnue => "with NNUE",
        StaticEvalSource::Terminal => "terminal",
    };
    out.push_str(&format!(
        " Final evaluation    {final_pawns:+.2} (white side) [{source_str}]\n\n"
    ));
}
