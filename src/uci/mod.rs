mod command;
mod input;
mod position;
mod protocol;
mod worker;

use anyhow::Result;
use command::{UciCommand, parse_uci_command};
use input::spawn_stdin_reader;
use position::apply_position;
use protocol::{
    eval_source_label, format_static_eval_score, format_verbose_eval, write_uci_identification,
};
use sable_engine::Engine;
use std::{
    io::{self, Write},
    sync::mpsc::RecvTimeoutError,
    time::Duration,
};
use worker::{WorkerCommand, cancel_active_search, drain_worker_events, spawn_search_worker};

pub fn run_uci_loop() -> Result<()> {
    let line_rx = spawn_stdin_reader();
    let (worker_tx, worker_rx) = spawn_search_worker();
    let mut stdout = io::stdout();
    let mut engine = Engine::default();
    let mut state = UciLoopState::default();

    while state.running {
        drain_worker_events(
            &worker_rx,
            &mut stdout,
            &mut state.search_in_progress,
            &mut state.active_search_id,
        )?;

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
                    &worker_tx,
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

    drain_worker_events(
        &worker_rx,
        &mut stdout,
        &mut state.search_in_progress,
        &mut state.active_search_id,
    )?;

    Ok(())
}

#[derive(Debug)]
struct UciLoopState {
    debug_enabled: bool,
    running: bool,
    search_in_progress: bool,
    next_search_id: u64,
    active_search_id: Option<u64>,
}

impl Default for UciLoopState {
    fn default() -> Self {
        Self {
            debug_enabled: false,
            running: true,
            search_in_progress: false,
            next_search_id: 1,
            active_search_id: None,
        }
    }
}

fn handle_command(
    command: UciCommand,
    input: &str,
    engine: &mut Engine,
    worker_tx: &std::sync::mpsc::Sender<WorkerCommand>,
    stdout: &mut io::Stdout,
    state: &mut UciLoopState,
) -> Result<()> {
    match command {
        UciCommand::Uci => write_uci_identification(stdout, engine)?,
        UciCommand::IsReady => write_ready(stdout)?,
        UciCommand::UciNewGame => {
            cancel_search(worker_tx, state);
            engine.reset();
        }
        UciCommand::Position(position) => {
            cancel_search(worker_tx, state);
            if let Err(err) = apply_position(position, engine) {
                writeln!(stdout, "info string position error: {err}")?;
                stdout.flush()?;
            }
        }
        UciCommand::Go(request) => start_search(worker_tx, stdout, state, engine, request)?,
        UciCommand::Stop => {
            let _ = worker_tx.send(WorkerCommand::Stop);
        }
        UciCommand::PonderHit => {
            let _ = worker_tx.send(WorkerCommand::PonderHit);
        }
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
            let _ = worker_tx.send(WorkerCommand::Quit);
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

fn cancel_search(
    worker_tx: &std::sync::mpsc::Sender<WorkerCommand>,
    state: &mut UciLoopState,
) {
    cancel_active_search(
        worker_tx,
        &mut state.search_in_progress,
        &mut state.active_search_id,
    );
}

fn start_search(
    worker_tx: &std::sync::mpsc::Sender<WorkerCommand>,
    stdout: &mut io::Stdout,
    state: &mut UciLoopState,
    engine: &Engine,
    request: sable_engine::SearchRequest,
) -> Result<()> {
    let search_id = state.next_search_id;
    state.next_search_id = state.next_search_id.saturating_add(1);
    if worker_tx
        .send(WorkerCommand::Start {
            engine: Box::new(engine.clone()),
            request: Box::new(request),
            search_id,
        })
        .is_err()
    {
        writeln!(stdout, "info string search worker unavailable")?;
        writeln!(stdout, "bestmove 0000")?;
        stdout.flush()?;
    } else {
        state.search_in_progress = true;
        state.active_search_id = Some(search_id);
    }
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
        Ok(()) => write_setoption_success(engine, stdout, debug_enabled, name, value)?,
        Err(err) => {
            writeln!(stdout, "info string setoption error: {err}")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

fn write_setoption_success(
    engine: &Engine,
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
    if normalized_name == "eval" || normalized_name == "evaluation" {
        writeln!(
            stdout,
            "info string eval mode set: {}",
            engine.eval_mode_option_value().as_uci()
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
    let eval = engine.static_eval();
    let score = format_static_eval_score(&eval);
    writeln!(stdout, "info string eval {score}")?;
    writeln!(stdout, "info string eval source {}", eval_source_label(eval.source))?;
    stdout.flush()?;
    Ok(())
}

fn write_verbose_eval(stdout: &mut io::Stdout, engine: &Engine) -> Result<()> {
    let veval = engine.verbose_eval();
    let output = format_verbose_eval(&veval);
    write!(stdout, "{output}")?;
    stdout.flush()?;
    Ok(())
}
