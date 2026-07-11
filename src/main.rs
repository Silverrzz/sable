mod genfens;
mod uci;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use sable_engine::{
    Engine, SearchLimits, SearchRequest, embedded_eval_hash, embedded_eval_label,
    has_embedded_eval, runtime_simd_backend,
};
use std::env;
use std::io::{self, Write};
use std::time::Instant;

const BENCH_DEPTH: u32 = 15;
const BENCH_POSITIONS: [&str; 6] = [
    "startpos",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "4rrk1/p1pb1ppp/1p1p1n2/8/2PP4/2N1P1P1/PP3PBP/R2R2K1 w - - 0 1",
    "2r2rk1/pp3ppp/2n1bn2/q2p4/3P4/2P1PN2/PP1NBPPP/R2Q1RK1 w - - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "2k4r/8/5p2/p2p1P2/P2P4/P7/8/4K1R1 w - - 0 1",
];

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
    /// Run bench repeatedly with an inline progress display
    Vbench {
        /// Number of benchmark runs
        #[arg(long, default_value_t = 15)]
        count: usize,
    },
    Bmt5k,
    Version,
}

fn main() -> Result<()> {
    if let Some(commands) = protocol_script_commands() {
        return run_protocol_script(commands);
    }

    let cli = Cli::parse();
    match cli.command.unwrap_or_else(command_from_env) {
        Command::Uci => uci::run_uci_loop(),
        Command::Perft { depth, fen } => run_perft(depth, fen),
        Command::Bench => run_bench(),
        Command::Vbench { count } => run_verbose_bench(count),
        Command::Bmt5k => run_bmt5k(),
        Command::Version => {
            print_version_info();
            Ok(())
        }
    }
}

fn protocol_script_commands() -> Option<Vec<String>> {
    let commands = env::args().skip(1).collect::<Vec<_>>();
    if commands
        .iter()
        .any(|command| is_protocol_script_command(command.trim()))
    {
        Some(commands)
    } else {
        None
    }
}

fn is_protocol_script_command(command: &str) -> bool {
    command == "quit"
        || command == "isready"
        || command.starts_with("setoption ")
        || command.starts_with("genfens ")
}

fn run_protocol_script(commands: Vec<String>) -> Result<()> {
    let mut engine = Engine::default();
    for command in commands {
        let command = command.trim();
        if command.is_empty() {
            continue;
        }
        if command == "quit" {
            break;
        }
        if command == "isready" {
            println!("readyok");
            continue;
        }
        if command.starts_with("setoption ") {
            apply_script_setoption(command, &mut engine)?;
            continue;
        }
        if command.starts_with("genfens ") {
            genfens::run_command(command, &mut engine)?;
            continue;
        }
        bail!("unsupported protocol script command: {command}");
    }
    Ok(())
}

fn apply_script_setoption(command: &str, engine: &mut Engine) -> Result<()> {
    let rest = command
        .strip_prefix("setoption ")
        .context("malformed setoption command")?;
    let (name, value) = parse_script_setoption(rest)?;
    engine
        .set_option(&name, value.as_deref())
        .with_context(|| format!("failed to apply setoption: {name}"))?;
    Ok(())
}

fn parse_script_setoption(rest: &str) -> Result<(String, Option<String>)> {
    let rest = rest
        .strip_prefix("name ")
        .context("setoption command is missing name")?;
    let Some((name, value)) = rest.split_once(" value ") else {
        return Ok((rest.trim().to_owned(), None));
    };
    let name = name.trim();
    if name.is_empty() {
        bail!("setoption command has an empty name");
    }
    Ok((name.to_owned(), Some(value.trim().to_owned())))
}

fn command_from_env() -> Command {
    match env::var("SABLE_MODE")
        .unwrap_or_else(|_| "uci".to_owned())
        .to_lowercase()
        .as_str()
    {
        "perft" => Command::Perft {
            depth: 5,
            fen: None,
        },
        "bench" => Command::Bench,
        "vbench" => Command::Vbench { count: 15 },
        "bmt5k" => Command::Bmt5k,
        "version" => Command::Version,
        _ => Command::Uci,
    }
}

fn print_version_info() {
    let release_id = option_env!("SABLE_RELEASE_ID").unwrap_or("dev");
    let git_commit = option_env!("SABLE_GIT_COMMIT").unwrap_or("unknown");
    let target = option_env!("TARGET").unwrap_or(std::env::consts::ARCH);
    let profile = option_env!("PROFILE").unwrap_or("unknown");
    let engine = Engine::default();
    let default_eval = if has_embedded_eval() {
        embedded_eval_label().unwrap_or("embedded")
    } else {
        "none"
    };
    let embedded_eval_hash = embedded_eval_hash().unwrap_or("none");
    let embedded_eval_arch = engine
        .loaded_nnue_architecture_id()
        .map(|id| id.as_str())
        .unwrap_or("none");
    println!("name=Sable");
    println!("pkg_version={}", env!("CARGO_PKG_VERSION"));
    println!("release_id={release_id}");
    println!("git_commit={git_commit}");
    println!("target={target}");
    println!("profile={profile}");
    println!("embedded_eval={}", if has_embedded_eval() { "true" } else { "false" });
    println!("embedded_eval_hash={embedded_eval_hash}");
    println!("embedded_eval_arch={embedded_eval_arch}");
    println!("default_eval_source={default_eval}");
    println!("simd_backend={}", runtime_simd_backend());
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
    let result = run_bench_once()?;
    println!(
        "bench depth={BENCH_DEPTH} positions={} simd_backend={} eval_arch={} eval_file={}",
        BENCH_POSITIONS.len(),
        runtime_simd_backend(),
        result.eval_arch,
        result.eval_file,
    );
    println!(
        "{} nodes {} nps search_ms={} setup_ms={}",
        result.nodes,
        result.nps(),
        result.search_ms,
        result.setup_ms,
    );
    Ok(())
}

struct BenchResult {
    nodes: u64,
    search_ms: u64,
    setup_ms: u64,
    eval_arch: String,
    eval_file: String,
}

impl BenchResult {
    fn nps(&self) -> u64 {
        nodes_per_second(self.nodes, self.search_ms)
    }
}

fn run_bench_once() -> Result<BenchResult> {

    let request = SearchRequest {
        limits: SearchLimits {
            depth: Some(BENCH_DEPTH),
            ..Default::default()
        },
        ..Default::default()
    };

    let mut engine = Engine::default();
    let eval_arch = engine
        .active_nnue_architecture_id()
        .map(|id| id.as_str())
        .unwrap_or("none");
    let eval_file = engine.eval_file_option_value().unwrap_or("none").to_owned();

    let mut total_nodes = 0_u64;
    let mut total_search_ms = 0_u64;
    let start_setup = Instant::now();
    for position in BENCH_POSITIONS {
        engine.reset();
        if position == "startpos" {
            engine.set_startpos_with_moves(&[])?;
        } else {
            engine.set_fen_with_moves(position, &[])?;
        }
        let start_search = Instant::now();
        let result = engine.search(&request)?;
        let search_ms = start_search.elapsed().as_millis() as u64;
        let nodes = result.info.nodes.unwrap_or(0);
        total_nodes = total_nodes.saturating_add(nodes);
        total_search_ms = total_search_ms.saturating_add(search_ms);
    }
    let total_elapsed_ms = start_setup.elapsed().as_millis() as u64;
    let setup_ms = total_elapsed_ms.saturating_sub(total_search_ms);
    Ok(BenchResult {
        nodes: total_nodes,
        search_ms: total_search_ms,
        setup_ms,
        eval_arch: eval_arch.to_owned(),
        eval_file,
    })
}

fn run_verbose_bench(runs: usize) -> Result<()> {
    const BAR_WIDTH: usize = 15;

    if runs == 0 {
        bail!("vbench count must be greater than zero");
    }

    println!(
        "Sable verbose bench | {runs} runs | depth {BENCH_DEPTH} | {} positions",
        BENCH_POSITIONS.len()
    );

    let started = Instant::now();
    let mut nps_values = Vec::with_capacity(runs);
    draw_bench_progress(0, runs, BAR_WIDTH, 0, 0, 0, None)?;

    for completed in 1..=runs {
        let result = run_bench_once()?;
        let current_nps = result.nps();
        nps_values.push(current_nps);
        let average_nps = nps_values.iter().map(|&nps| u128::from(nps)).sum::<u128>()
            / nps_values.len() as u128;
        let elapsed_secs = started.elapsed().as_secs();
        let eta_secs = elapsed_secs
            .saturating_mul((runs - completed) as u64)
            .checked_div(completed as u64)
            .unwrap_or(0);
        draw_bench_progress(
            completed,
            runs,
            BAR_WIDTH,
            current_nps,
            average_nps as u64,
            elapsed_secs,
            Some(eta_secs),
        )?;
    }

    println!();
    let average_nps = nps_values.iter().map(|&nps| u128::from(nps)).sum::<u128>()
        / nps_values.len() as u128;
    let min_nps = nps_values.iter().copied().min().unwrap_or(0);
    let max_nps = nps_values.iter().copied().max().unwrap_or(0);
    println!(
        "Average: {} nps | Min: {} | Max: {} | Total time: {}",
        format_number(average_nps as u64),
        format_number(min_nps),
        format_number(max_nps),
        format_duration(started.elapsed().as_secs()),
    );
    Ok(())
}

fn draw_bench_progress(
    completed: usize,
    runs: usize,
    bar_width: usize,
    current_nps: u64,
    average_nps: u64,
    elapsed_secs: u64,
    eta_secs: Option<u64>,
) -> Result<()> {
    let filled = completed.saturating_mul(bar_width) / runs;
    let bar = format!("{}{}", "#".repeat(filled), "-".repeat(bar_width - filled));
    let eta = eta_secs.map(format_duration).unwrap_or_else(|| "--:--".to_owned());
    print!(
        "\r\x1b[2K[{bar}] {completed:>2}/{runs} | Current: {:>13} nps | Average: {:>13} nps | Elapsed: {} | ETA: {eta}",
        format_number(current_nps),
        format_number(average_nps),
        format_duration(elapsed_secs),
    );
    io::stdout().flush().context("failed to update bench display")
}

fn format_number(value: u64) -> String {
    let digits = value.to_string();
    let mut formatted = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            formatted.push(',');
        }
        formatted.push(character);
    }
    formatted
}

fn format_duration(seconds: u64) -> String {
    format!("{:02}:{:02}", seconds / 60, seconds % 60)
}

fn run_bmt5k() -> Result<()> {
    const MOVE_TIME_MS: u64 = 5_000;

    let engine = Engine::default();
    let request = SearchRequest {
        limits: SearchLimits {
            move_time_ms: Some(MOVE_TIME_MS),
            ..Default::default()
        },
        ..Default::default()
    };

    let result = engine.search_with_observer(&request, None, |info| {
        println!("{}", uci::format_uci_info(info, false));
    })?;

    match result.best_move {
        Some(best) => {
            if let Some(ponder) = result.ponder_move {
                println!("bestmove {best} ponder {ponder}");
            } else {
                println!("bestmove {best}");
            }
        }
        None => println!("bestmove 0000"),
    }
    Ok(())
}
