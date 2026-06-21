use anyhow::{Context, Result, bail};
use sable_engine::{Engine, GameStatus};
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

const DEFAULT_PLIES: u32 = 8;
const MAX_GENERATION_ATTEMPTS: usize = 256;

pub(crate) fn run_command(command: &str, engine: &mut Engine) -> Result<()> {
    let args = parse_command(command)?;
    let mut rng = SplitMix64::new(args.seed);
    let starts = load_start_positions(&args.book)?;

    for _ in 0..args.count {
        let fen = generate_opening(&starts, args.plies, &mut rng, engine)?;
        println!("info string genfens {fen}");
    }

    Ok(())
}

struct GenfensArgs {
    count: usize,
    seed: u64,
    book: String,
    plies: PlyRange,
}

#[derive(Clone, Copy)]
struct PlyRange {
    min: u32,
    max: u32,
}

impl PlyRange {
    fn exact(plies: u32) -> Self {
        Self {
            min: plies,
            max: plies,
        }
    }

    fn sample(self, rng: &mut SplitMix64) -> u32 {
        if self.min == self.max {
            return self.min;
        }
        self.min + rng.index((self.max - self.min + 1) as usize) as u32
    }
}

fn parse_command(command: &str) -> Result<GenfensArgs> {
    let tokens = command.split_whitespace().collect::<Vec<_>>();
    if tokens.len() < 6 || tokens[0] != "genfens" {
        bail!("malformed genfens command");
    }

    let count = parse_nonzero_usize(tokens[1], "count")?;
    if tokens[2] != "seed" {
        bail!("expected seed after genfens count");
    }
    let seed = tokens[3]
        .parse::<u64>()
        .with_context(|| format!("invalid genfens seed: {}", tokens[3]))?;
    if tokens[4] != "book" {
        bail!("expected book after genfens seed");
    }

    let mut plies = PlyRange::exact(DEFAULT_PLIES);
    for token in &tokens[6..] {
        if let Some(raw) = token.strip_prefix("plies=") {
            plies = parse_ply_range(raw)?;
        } else if !token.is_empty() {
            bail!("unknown genfens argument: {token}");
        }
    }

    Ok(GenfensArgs {
        count,
        seed,
        book: tokens[5].to_owned(),
        plies,
    })
}

fn parse_nonzero_usize(raw: &str, label: &str) -> Result<usize> {
    let value = raw
        .parse::<usize>()
        .with_context(|| format!("invalid genfens {label}: {raw}"))?;
    if value == 0 {
        bail!("genfens {label} must be greater than zero");
    }
    Ok(value)
}

fn parse_ply_range(raw: &str) -> Result<PlyRange> {
    let Some((min, max)) = raw.split_once('-') else {
        let plies = raw
            .parse::<u32>()
            .with_context(|| format!("invalid plies value: {raw}"))?;
        if plies == 0 {
            bail!("plies must be greater than zero");
        }
        return Ok(PlyRange::exact(plies));
    };

    let min = min
        .parse::<u32>()
        .with_context(|| format!("invalid minimum plies value: {raw}"))?;
    let max = max
        .parse::<u32>()
        .with_context(|| format!("invalid maximum plies value: {raw}"))?;
    if min == 0 || max == 0 || min > max {
        bail!("invalid plies range: {raw}");
    }
    Ok(PlyRange { min, max })
}

fn load_start_positions(book: &str) -> Result<Vec<String>> {
    if book.eq_ignore_ascii_case("none") {
        return Ok(vec!["startpos".to_owned()]);
    }

    let file = File::open(Path::new(book)).with_context(|| format!("failed to open book: {book}"))?;
    let reader = BufReader::new(file);
    let mut starts = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        starts.push(parse_book_line(trimmed)?);
    }

    if starts.is_empty() {
        bail!("book contains no usable positions: {book}");
    }
    Ok(starts)
}

fn parse_book_line(line: &str) -> Result<String> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.len() >= 6 && tokens[4].parse::<u32>().is_ok() && tokens[5].parse::<u32>().is_ok() {
        return Ok(tokens[..6].join(" "));
    }
    if tokens.len() < 4 {
        bail!("malformed book line: {line}");
    }

    let halfmove = epd_integer_operand(&tokens, "hmvc").unwrap_or(0);
    let fullmove = epd_integer_operand(&tokens, "fmvn").unwrap_or(1).max(1);
    Ok(format!(
        "{} {} {} {} {halfmove} {fullmove}",
        tokens[0], tokens[1], tokens[2], tokens[3]
    ))
}

fn epd_integer_operand(tokens: &[&str], opcode: &str) -> Option<u32> {
    tokens.windows(2).find_map(|window| {
        if window[0] != opcode {
            return None;
        }
        window[1].trim_end_matches(';').parse::<u32>().ok()
    })
}

fn generate_opening(
    starts: &[String],
    plies: PlyRange,
    rng: &mut SplitMix64,
    engine: &mut Engine,
) -> Result<String> {
    for _ in 0..MAX_GENERATION_ATTEMPTS {
        let start = &starts[rng.index(starts.len())];
        set_start(engine, start)?;
        play_random_plies(engine, plies.sample(rng), rng);
        if engine.status() == GameStatus::Ongoing {
            return Ok(engine.full_fen());
        }
    }

    bail!("failed to generate a non-terminal opening");
}

fn set_start(engine: &mut Engine, start: &str) -> Result<()> {
    if start == "startpos" {
        engine.set_startpos_with_moves(&[])?;
    } else {
        engine.set_fen_with_moves(start, &[])?;
    }
    Ok(())
}

fn play_random_plies(engine: &mut Engine, plies: u32, rng: &mut SplitMix64) {
    for _ in 0..plies {
        if engine.status() != GameStatus::Ongoing {
            return;
        }
        let legal_moves = engine.legal_moves();
        if legal_moves.is_empty() {
            return;
        }
        let mv = legal_moves[rng.index(legal_moves.len())];
        engine.play_move(mv);
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn index(&mut self, len: usize) -> usize {
        debug_assert!(len > 0);
        (self.next_u64() % len as u64) as usize
    }
}
