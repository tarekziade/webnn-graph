use clap::{Parser, Subcommand};
use std::fs;

use webnn_graph::ast::GraphJson;
use webnn_graph::emit_js::emit_builder_js;
use webnn_graph::parser::parse_wg_text;
use webnn_graph::validate::{validate_graph, validate_weights};
use webnn_graph::weights::WeightsManifest;

#[derive(Parser)]
#[command(name = "webnn-graph")]
#[command(about = "WebNN Graph DSL tools", long_about = None)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    Parse {
        path: String,
    },
    Validate {
        path: String,
        #[arg(long)]
        weights_manifest: Option<String>,
    },
    EmitJs {
        path: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.cmd {
        Command::Parse { path } => {
            let txt = fs::read_to_string(path)?;
            let g = parse_wg_text(&txt)?;
            println!("{}", serde_json::to_string_pretty(&g)?);
        }
        Command::Validate {
            path,
            weights_manifest,
        } => {
            let txt = fs::read_to_string(path)?;
            let g: GraphJson = serde_json::from_str(&txt)?;
            validate_graph(&g)?;

            if let Some(mpath) = weights_manifest {
                let mtxt = fs::read_to_string(mpath)?;
                let m: WeightsManifest = serde_json::from_str(&mtxt)?;
                validate_weights(&g, &m)?;
            }
            eprintln!("OK");
        }
        Command::EmitJs { path } => {
            let txt = fs::read_to_string(path)?;
            let g: GraphJson = serde_json::from_str(&txt)?;
            validate_graph(&g)?;
            let js = emit_builder_js(&g);
            print!("{js}");
        }
    }

    Ok(())
}
