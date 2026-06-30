use std::{
    env,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SABLE_RELEASE_ID");
    println!("cargo:rerun-if-env-changed=SABLE_GIT_COMMIT");
    println!("cargo:rerun-if-env-changed=SABLE_EVAL_FILE");
    println!("cargo:rerun-if-env-changed=SABLE_EVAL_LABEL");
    println!("cargo:rerun-if-env-changed=SABLE_DEFAULT_EVAL");

    if let Ok(release_id) = env::var("SABLE_RELEASE_ID")
        && !release_id.trim().is_empty()
    {
        println!("cargo:rustc-env=SABLE_RELEASE_ID={release_id}");
    }

    let git_commit = env::var("SABLE_GIT_COMMIT")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(git_commit);
    if let Some(git_commit) = git_commit {
        println!("cargo:rustc-env=SABLE_GIT_COMMIT={git_commit}");
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR should be set by Cargo"));
    generate_attack_tables(&out_dir);

    let source = workspace_default_weights();
    let has_weights = source.exists();
    let default_eval_mode = default_eval_mode(has_weights);
    println!("cargo:rustc-env=SABLE_ENGINE_DEFAULT_EVAL_MODE={default_eval_mode}");

    if !has_weights {
        let embedded_path = out_dir.join("embedded-default-eval-empty.bin");
        fs::write(&embedded_path, []).unwrap_or_else(|error| {
            panic!(
                "Failed to write empty embedded eval placeholder '{}': {error}",
                embedded_path.display()
            )
        });
        println!("cargo:rerun-if-changed={}", source.display());
        println!("cargo:rustc-env=SABLE_ENGINE_HAS_EMBEDDED_EVAL=0");
        println!(
            "cargo:rustc-env=SABLE_ENGINE_EMBEDDED_EVAL_PATH={}",
            embedded_path.display()
        );
        println!("cargo:rustc-env=SABLE_ENGINE_EMBEDDED_EVAL_LABEL=none");
        println!("cargo:rustc-env=SABLE_ENGINE_EMBEDDED_EVAL_HASH=none");
        return;
    }

    println!("cargo:rerun-if-changed={}", source.display());
    let bytes = fs::read(&source).unwrap_or_else(|error| {
        panic!(
            "Failed to read embedded eval from '{}': {error}",
            source.display()
        )
    });
    let embedded_hash = fnv1a64(&bytes);
    let embedded_path = out_dir.join(format!("embedded-default-eval-{embedded_hash:016x}.bin"));
    fs::write(&embedded_path, bytes).unwrap_or_else(|error| {
        panic!(
            "Failed to write embedded eval from '{}' to '{}': {error}",
            source.display(),
            embedded_path.display()
        )
    });
    println!("cargo:rustc-env=SABLE_ENGINE_HAS_EMBEDDED_EVAL=1");
    println!(
        "cargo:rustc-env=SABLE_ENGINE_EMBEDDED_EVAL_PATH={}",
        embedded_path.display()
    );
    println!(
        "cargo:rustc-env=SABLE_ENGINE_EMBEDDED_EVAL_LABEL={}",
        display_label(&source)
    );
    println!("cargo:rustc-env=SABLE_ENGINE_EMBEDDED_EVAL_HASH={embedded_hash:016x}");
}

const ROOK_TABLE_SIZE: usize = 4096;
const BISHOP_TABLE_SIZE: usize = 512;

fn generate_attack_tables(out_dir: &Path) {
    let path = out_dir.join("attack_tables.rs");
    let rook_masks = slider_masks(false);
    let bishop_masks = slider_masks(true);
    let mut out = String::with_capacity(6_000_000);
    write_u64_array(&mut out, "ROOK_RELEVANT_MASKS", &rook_masks);
    write_u64_array(&mut out, "BISHOP_RELEVANT_MASKS", &bishop_masks);
    write_attack_table(
        &mut out,
        "ROOK_ATTACKS",
        "ROOK_TABLE_SIZE",
        ROOK_TABLE_SIZE,
        &rook_masks,
        false,
    );
    write_attack_table(
        &mut out,
        "BISHOP_ATTACKS",
        "BISHOP_TABLE_SIZE",
        BISHOP_TABLE_SIZE,
        &bishop_masks,
        true,
    );
    fs::write(&path, out).unwrap_or_else(|error| {
        panic!(
            "Failed to write generated attack tables '{}': {error}",
            path.display()
        )
    });
}

fn write_u64_array(out: &mut String, name: &str, values: &[u64; 64]) {
    writeln!(out, "const {name}: [u64; 64] = [").expect("writing to String cannot fail");
    for value in values {
        writeln!(out, "    0x{value:016x},").expect("writing to String cannot fail");
    }
    writeln!(out, "];").expect("writing to String cannot fail");
}

fn write_attack_table(
    out: &mut String,
    name: &str,
    size_name: &str,
    size: usize,
    masks: &[u64; 64],
    bishop: bool,
) {
    writeln!(out, "const {name}: [[u64; {size_name}]; 64] = [")
        .expect("writing to String cannot fail");
    for (square, mask) in masks.iter().enumerate() {
        writeln!(out, "    [").expect("writing to String cannot fail");
        for index in 0..size {
            let occupied = occupancy_from_index(index, *mask);
            let attacks = slider_attacks_from(square, occupied, bishop);
            writeln!(out, "        0x{attacks:016x},").expect("writing to String cannot fail");
        }
        writeln!(out, "    ],").expect("writing to String cannot fail");
    }
    writeln!(out, "];").expect("writing to String cannot fail");
}

fn slider_masks(bishop: bool) -> [u64; 64] {
    let mut masks = [0u64; 64];
    for (square, mask) in masks.iter_mut().enumerate() {
        *mask = slider_relevant_mask(square, bishop);
    }
    masks
}

fn slider_relevant_mask(square: usize, bishop: bool) -> u64 {
    if bishop {
        relevant_ray(square, 1, 1)
            | relevant_ray(square, -1, 1)
            | relevant_ray(square, 1, -1)
            | relevant_ray(square, -1, -1)
    } else {
        relevant_ray(square, 1, 0)
            | relevant_ray(square, -1, 0)
            | relevant_ray(square, 0, 1)
            | relevant_ray(square, 0, -1)
    }
}

fn relevant_ray(square: usize, df: i8, dr: i8) -> u64 {
    let mut ray = 0u64;
    let mut file = (square as i8 & 7) + df;
    let mut rank = (square as i8 >> 3) + dr;
    while (0..8).contains(&file) && (0..8).contains(&rank) {
        let next_file = file + df;
        let next_rank = rank + dr;
        if !(0..8).contains(&next_file) || !(0..8).contains(&next_rank) {
            break;
        }
        ray |= 1u64 << (rank as usize * 8 + file as usize);
        file = next_file;
        rank = next_rank;
    }
    ray
}

fn occupancy_from_index(mut index: usize, mut mask: u64) -> u64 {
    let mut occupied = 0u64;
    while mask != 0 {
        let bit = mask & mask.wrapping_neg();
        if index & 1 != 0 {
            occupied |= bit;
        }
        index >>= 1;
        mask ^= bit;
    }
    occupied
}

fn slider_attacks_from(square: usize, occupied: u64, bishop: bool) -> u64 {
    if bishop {
        ray_attacks_from(square, occupied, 1, 1)
            | ray_attacks_from(square, occupied, -1, 1)
            | ray_attacks_from(square, occupied, 1, -1)
            | ray_attacks_from(square, occupied, -1, -1)
    } else {
        ray_attacks_from(square, occupied, 1, 0)
            | ray_attacks_from(square, occupied, -1, 0)
            | ray_attacks_from(square, occupied, 0, 1)
            | ray_attacks_from(square, occupied, 0, -1)
    }
}

fn ray_attacks_from(square: usize, occupied: u64, df: i8, dr: i8) -> u64 {
    let mut attacks = 0u64;
    let mut file = (square as i8 & 7) + df;
    let mut rank = (square as i8 >> 3) + dr;
    while (0..8).contains(&file) && (0..8).contains(&rank) {
        let bit = 1u64 << (rank as usize * 8 + file as usize);
        attacks |= bit;
        if occupied & bit != 0 {
            break;
        }
        file += df;
        rank += dr;
    }
    attacks
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn default_eval_mode(has_weights: bool) -> String {
    let raw = env::var("SABLE_DEFAULT_EVAL").unwrap_or_else(|_| {
        if has_weights {
            "nnue".to_owned()
        } else {
            "hce".to_owned()
        }
    });
    let mut key = raw.to_ascii_lowercase();
    key.retain(|ch| ch != ' ' && ch != '-');
    match key.as_str() {
        "" | "hce" | "handcrafted" | "classical" | "material" => "hce".to_owned(),
        "nnue" => "nnue".to_owned(),
        _ => panic!("SABLE_DEFAULT_EVAL must be 'hce' or 'nnue', got '{raw}'"),
    }
}

fn workspace_default_weights() -> PathBuf {
    if let Ok(path) = env::var("SABLE_EVAL_FILE")
        && !path.trim().is_empty()
    {
        return PathBuf::from(path);
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_default = manifest_dir.join("data").join("quantised.bin");
    if repo_default.exists() {
        return repo_default;
    }

    let workspace_default = manifest_dir.join("..").join("data").join("quantised.bin");
    if workspace_default.exists() {
        return workspace_default;
    }

    repo_default
}

fn display_label(source: &Path) -> String {
    if let Ok(label) = env::var("SABLE_EVAL_LABEL")
        && !label.trim().is_empty()
    {
        return label;
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Ok(workspace) = manifest_dir.join("..").canonicalize()
        && let Ok(source) = source.canonicalize()
        && let Ok(relative) = source.strip_prefix(&workspace)
    {
        return relative.display().to_string().replace('\\', "/");
    }

    source
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "embedded".to_owned())
}

fn git_commit() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let commit = String::from_utf8(output.stdout).ok()?;
    let commit = commit.trim();
    if commit.is_empty() {
        None
    } else {
        Some(commit.to_owned())
    }
}

