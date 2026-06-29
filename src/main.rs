mod uci;

use anyhow::Result;
use clap::{Parser, Subcommand};
use sable_engine::{Engine, SearchLimits, SearchRequest, embedded_eval_label, has_embedded_eval};
use std::env;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about = "Sable chess engine")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    Uci,
    Perft {
        #[arg(long, default_value_t = 5)]
        depth: u32,
        #[arg(long)]
        fen: Option<String>,
    },
    Bench,
    Version,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command.unwrap_or_else(command_from_env) {
        Command::Uci => uci::run_uci_loop(),
        Command::Perft { depth, fen } => run_perft(depth, fen),
        Command::Bench => run_bench(),
        Command::Version => {
            print_version_info();
            Ok(())
        }
    }
}

fn command_from_env() -> Command {
    match env::var("SABLER_MODE")
        .unwrap_or_else(|_| "uci".to_owned())
        .to_lowercase()
        .as_str()
    {
        "perft" => Command::Perft {
            depth: 5,
            fen: None,
        },
        "bench" => Command::Bench,
        "version" => Command::Version,
        _ => Command::Uci,
    }
}

fn print_version_info() {
    let release_id = option_env!("SABLER_RELEASE_ID").unwrap_or("dev");
    let git_commit = option_env!("SABLER_GIT_COMMIT").unwrap_or("unknown");
    let target = option_env!("TARGET").unwrap_or(std::env::consts::ARCH);
    let profile = option_env!("PROFILE").unwrap_or("unknown");
    let default_eval_mode = option_env!("SABLE_ENGINE_DEFAULT_EVAL_MODE").unwrap_or("hce");
    let default_eval = if has_embedded_eval() {
        embedded_eval_label().unwrap_or("embedded")
    } else {
        "none"
    };
    println!("name=Sable");
    println!("pkg_version={}", env!("CARGO_PKG_VERSION"));
    println!("release_id={release_id}");
    println!("git_commit={git_commit}");
    println!("target={target}");
    println!("profile={profile}");
    println!("embedded_eval={}", if has_embedded_eval() { "true" } else { "false" });
    println!("default_eval_mode={default_eval_mode}");
    println!("default_eval_source={default_eval}");
}

fn nodes_per_second(nodes: u64, elapsed_ms: u64) -> u64 {
    nodes.saturating_mul(1000).checked_div(elapsed_ms).unwrap_or(0)
}

fn run_perft(depth: u32, fen: Option<String>) -> Result<()> {
    let mut engine = Engine::default();
    if let Some(fen) = fen {
        engine.set_fen_with_moves(&fen, &[])?;
    }
    let start = Instant::now();
    let nodes = engine.perft(depth);
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let nps = nodes_per_second(nodes, elapsed_ms);
    println!("perft depth={depth} nodes={nodes} time_ms={elapsed_ms} nps={nps}");
    Ok(())
}

fn run_bench() -> Result<()> {
    const BENCH_DEPTH: u32 = 15;

    let positions = [
        "startpos",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "4rrk1/p1pb1ppp/1p1p1n2/8/2PP4/2N1P1P1/PP3PBP/R2R2K1 w - - 0 1",
        "2r2rk1/pp3ppp/2n1bn2/q2p4/3P4/2P1PN2/PP1NBPPP/R2Q1RK1 w - - 0 10",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "2k4r/8/5p2/p2p1P2/P2P4/P7/8/4K1R1 w - - 0 1"
    ];

    let request = SearchRequest {
        limits: SearchLimits {
            depth: Some(BENCH_DEPTH),
            ..Default::default()
        },
        ..Default::default()
    };

    let start_all = Instant::now();
    let mut total_nodes = 0_u64;
    for position in positions {
        let mut engine = Engine::default();
        if position != "startpos" {
            engine.set_fen_with_moves(position, &[])?;
        }
        let result = engine.search(&request)?;
        let nodes = result.info.nodes.unwrap_or(0);
        total_nodes = total_nodes.saturating_add(nodes);
    }
    let all_ms = start_all.elapsed().as_millis() as u64;
    println!("{} nodes {} nps", total_nodes, nodes_per_second(total_nodes, all_ms));
    Ok(())
}
