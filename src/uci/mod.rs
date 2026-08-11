mod command;
mod input;
mod protocol;
mod worker;

use anyhow::Result;
use command::{PositionBase, UciCommand, parse_uci_command};
use input::spawn_stdin_reader;
pub(crate) use protocol::format_uci_info;
use protocol::{
    eval_source_label, format_static_eval_score, format_verbose_eval, write_uci_identification,
};
use sable_engine::Engine;
use std::{
    io::{self, Write},
    sync::mpsc::RecvTimeoutError,
    time::Duration,
};
use worker::SearchWorker;

pub fn run_uci_loop() -> Result<()> {
    let line_rx = spawn_stdin_reader();
    let mut worker = SearchWorker::new();
    let mut stdout = io::stdout();
    let mut engine = Engine::default();
    let mut state = UciLoopState::default();

    while state.running {
        worker.drain_events(&mut stdout, &mut state.active_search_id)?;

        match line_rx.recv_timeout(Duration::from_millis(20)) {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
                let command = parse_uci_command(input);
                handle_command(
                    command,
                    input,
                    &mut engine,
                    &mut worker,
                    &mut stdout,
                    &mut state,
                )?;
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                state.running = false;
            }
        }
    }

    worker.shutdown();
    worker.drain_events(&mut stdout, &mut state.active_search_id)?;

    Ok(())
}

#[derive(Debug)]
struct UciLoopState {
    debug_enabled: bool,
    running: bool,
    next_search_id: u64,
    active_search_id: Option<u64>,
}

impl Default for UciLoopState {
    fn default() -> Self {
        Self {
            debug_enabled: false,
            running: true,
            next_search_id: 1,
            active_search_id: None,
        }
    }
}

fn handle_command(
    command: UciCommand,
    input: &str,
    engine: &mut Engine,
    worker: &mut SearchWorker,
    stdout: &mut io::Stdout,
    state: &mut UciLoopState,
) -> Result<()> {
    match command {
        UciCommand::Uci => write_uci_identification(stdout, engine)?,
        UciCommand::IsReady => write_ready(stdout)?,
        UciCommand::UciNewGame => {
            worker.stop();
            state.active_search_id = None;
            engine.reset();
        }
        UciCommand::Position(position) => {
            worker.stop();
            state.active_search_id = None;
            let result = match position.base {
                PositionBase::StartPos => engine.set_startpos_with_moves(&position.moves),
                PositionBase::Fen(fen) => engine.set_fen_with_moves(&fen, &position.moves),
            };
            if let Err(err) = result {
                writeln!(stdout, "info string position error: {err}")?;
                stdout.flush()?;
            }
        }
        UciCommand::Go(request) => {
            let search_id = state.next_search_id;
            state.next_search_id = state.next_search_id.saturating_add(1);
            worker.start(engine.clone(), request, search_id);
            state.active_search_id = Some(search_id);
        }
        UciCommand::Stop => worker.stop(),
        UciCommand::PonderHit => worker.ponder_hit(),
        UciCommand::SetOption { name, value } => {
            handle_setoption(engine, stdout, state.debug_enabled, name, value)?
        }
        UciCommand::Register { name, code, later } => {
            handle_register(stdout, state.debug_enabled, name, code, later)?
        }
        UciCommand::Debug(enabled) => {
            state.debug_enabled = enabled;
        }
        UciCommand::Eval => write_static_eval(stdout, engine)?,
        UciCommand::VerboseEval => write_verbose_eval(stdout, engine)?,
        UciCommand::Quit => {
            worker.stop();
            state.running = false;
        }
        UciCommand::Unknown => {
            if state.debug_enabled {
                writeln!(stdout, "info string ignored unknown command: {input}")?;
                stdout.flush()?;
            }
        }
        UciCommand::ParseError(err) => {
            writeln!(stdout, "info string {err}")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

fn write_ready(stdout: &mut io::Stdout) -> Result<()> {
    writeln!(stdout, "readyok")?;
    stdout.flush()?;
    Ok(())
}

fn handle_setoption(
    engine: &mut Engine,
    stdout: &mut io::Stdout,
    debug_enabled: bool,
    name: String,
    value: Option<String>,
) -> Result<()> {
    match engine.set_option(&name, value.as_deref()) {
        Ok(()) => write_setoption_success(stdout, debug_enabled, name, value)?,
        Err(err) => {
            writeln!(stdout, "info string setoption error: {err}")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

fn write_setoption_success(
    stdout: &mut io::Stdout,
    debug_enabled: bool,
    name: String,
    value: Option<String>,
) -> Result<()> {
    let normalized_name = name.to_ascii_lowercase().replace(' ', "");
    if normalized_name == "evalfile" {
        writeln!(
            stdout,
            "info string eval file loaded: {}",
            value.clone().unwrap_or_default()
        )?;
        stdout.flush()?;
    }
    if debug_enabled {
        writeln!(
            stdout,
            "info string setoption applied name={} value={}",
            name,
            value.unwrap_or_default()
        )?;
        stdout.flush()?;
    }
    Ok(())
}

fn handle_register(
    stdout: &mut io::Stdout,
    debug_enabled: bool,
    name: Option<String>,
    code: Option<String>,
    later: bool,
) -> Result<()> {
    if debug_enabled {
        writeln!(
            stdout,
            "info string register name={} code={} later={}",
            name.unwrap_or_default(),
            code.unwrap_or_default(),
            later
        )?;
        stdout.flush()?;
    }
    Ok(())
}

fn write_static_eval(stdout: &mut io::Stdout, engine: &Engine) -> Result<()> {
    match engine.static_eval() {
        Ok(eval) => {
            let score = format_static_eval_score(&eval);
            writeln!(stdout, "info string eval {score}")?;
            writeln!(
                stdout,
                "info string eval source {}",
                eval_source_label(eval.source)
            )?;
        }
        Err(err) => {
            writeln!(stdout, "info string eval error: {err}")?;
        }
    }
    stdout.flush()?;
    Ok(())
}

fn write_verbose_eval(stdout: &mut io::Stdout, engine: &Engine) -> Result<()> {
    match engine.verbose_eval() {
        Ok(veval) => {
            let output = format_verbose_eval(&veval);
            write!(stdout, "{output}")?;
        }
        Err(err) => {
            writeln!(stdout, "info string eval error: {err}")?;
        }
    }
    stdout.flush()?;
    Ok(())
}
