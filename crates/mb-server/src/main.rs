use std::path::PathBuf;

use clap::{Parser, Subcommand};
use mb_server::bootstrap;
use mb_server::config::AppConfig;

#[derive(Parser)]
#[command(name = "mb", about = "model-bridge LLM API gateway")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Path to the configuration file.
    #[arg(short, long, default_value = "config.toml", global = true)]
    config: PathBuf,
}

#[derive(Subcommand)]
enum Command {
    /// Validate configuration file and exit.
    Validate,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Validate) => {
            run_validate(&cli.config);
        }
        None => {
            // Gateway startup (placeholder until handler tasks are done)
            run_validate(&cli.config);
            println!("Gateway startup not yet implemented.");
        }
    }
}

fn run_validate(path: &std::path::Path) {
    let config = match AppConfig::from_file(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading config: {e}");
            std::process::exit(1);
        }
    };

    match bootstrap::into_runtime(config) {
        Ok(_runtime) => {
            println!("Config valid: {}", path.display());
        }
        Err(e) => {
            eprintln!("Config invalid: {e}");
            std::process::exit(1);
        }
    }
}
