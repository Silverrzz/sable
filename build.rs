use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SABLER_RELEASE_ID");
    println!("cargo:rerun-if-env-changed=SABLER_GIT_COMMIT");
    println!("cargo:rerun-if-env-changed=SABLER_EVAL_FILE");
    println!("cargo:rerun-if-env-changed=SABLER_EVAL_LABEL");
    println!("cargo:rerun-if-env-changed=SABLER_DEFAULT_EVAL");

    if let Ok(release_id) = env::var("SABLER_RELEASE_ID")
        && !release_id.trim().is_empty()
    {
        println!("cargo:rustc-env=SABLER_RELEASE_ID={release_id}");
    }

    let git_commit = env::var("SABLER_GIT_COMMIT")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(git_commit);
    if let Some(git_commit) = git_commit {
        println!("cargo:rustc-env=SABLER_GIT_COMMIT={git_commit}");
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR should be set by Cargo"));
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
    let raw = env::var("SABLER_DEFAULT_EVAL").unwrap_or_else(|_| {
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
        _ => panic!("SABLER_DEFAULT_EVAL must be 'hce' or 'nnue', got '{raw}'"),
    }
}

fn workspace_default_weights() -> PathBuf {
    if let Ok(path) = env::var("SABLER_EVAL_FILE")
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
    if let Ok(label) = env::var("SABLER_EVAL_LABEL")
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

