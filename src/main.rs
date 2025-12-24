use clap::{Parser, Subcommand};
use std::fs;

use webnn_graph::ast::GraphJson;
use webnn_graph::emit_js::{emit_builder_js, emit_weights_loader_js};
use webnn_graph::parser::parse_wg_text;
use webnn_graph::validate::{validate_graph, validate_weights};
use webnn_graph::weights::WeightsManifest;
use webnn_graph::weights_io::{create_manifest, pack_weights, unpack_weights};

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
    PackWeights {
        #[arg(long)]
        manifest: String,
        #[arg(long)]
        input_dir: String,
        #[arg(long)]
        output: String,
    },
    UnpackWeights {
        #[arg(long)]
        weights: String,
        #[arg(long)]
        manifest: String,
        #[arg(long)]
        output_dir: String,
    },
    CreateManifest {
        #[arg(long)]
        input_dir: String,
        #[arg(long)]
        output: String,
        #[arg(long, default_value = "little")]
        endianness: String,
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

            // Emit WeightsFile helper class
            print!("{}", emit_weights_loader_js());
            println!();

            // Emit buildGraph function
            let js = emit_builder_js(&g);
            print!("{js}");
        }
        Command::PackWeights {
            manifest,
            input_dir,
            output,
        } => {
            pack_weights(&manifest, &input_dir, &output)?;
        }
        Command::UnpackWeights {
            weights,
            manifest,
            output_dir,
        } => {
            unpack_weights(&weights, &manifest, &output_dir)?;
        }
        Command::CreateManifest {
            input_dir,
            output,
            endianness,
        } => {
            create_manifest(&input_dir, &output, &endianness)?;
        }
    }

    Ok(())
}
