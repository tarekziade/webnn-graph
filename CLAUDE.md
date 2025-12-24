# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

webnn-graph is a Rust implementation for a WebNN-oriented graph DSL. It provides a complete pipeline:
1. Parse WebNN graph text files (.webnn) into JSON representation
2. Validate graph structure and weights manifests
3. Emit WebNN JavaScript builder code

## Build and Test Commands

Build the project:
```bash
make build
```

Run the CLI tool:
```bash
make run
```

Run tests:
```bash
make test
```

Run a specific test:
```bash
cargo test <test_name>
```

Format code:
```bash
make fmt
```

Check code formatting:
```bash
make fmt-check
```

Run clippy linter:
```bash
make lint
```

Quick compile check:
```bash
make check
```

Clean build artifacts:
```bash
make clean
```

View all available commands:
```bash
make help
```

## CLI Usage

The binary is named `webnn-graph` with three subcommands:

Parse graph text to JSON:
```bash
make parse
# Or directly:
webnn-graph parse examples/resnet_head.webnn > graph.json
```

Validate JSON (with optional weights manifest):
```bash
make validate
# Or directly:
webnn-graph validate graph.json --weights-manifest examples/weights.manifest.json
```

Emit JavaScript builder code:
```bash
make emit-js
# Or directly:
webnn-graph emit-js graph.json > buildGraph.js
```

## Architecture

### Module Structure

- **ast.rs**: Core data structures for the graph JSON format
  - `GraphJson`: Top-level structure containing inputs, consts, nodes, outputs
  - `OperandDesc`: Describes tensor shape and data type
  - `DataType`: Enum for f32, f16, i32, u32, i64, u64, i8, u8
  - `ConstDecl`: Constant declarations with initialization (weights, scalar, or inline bytes)
  - `Node`: Represents operations with inputs, options, and outputs

- **parser.rs**: Pest-based parser for WG text format
  - Uses `wg.pest` grammar file
  - `parse_wg_text()`: Main entry point converting WG text to `GraphJson`
  - Handles four main blocks: inputs, consts, nodes, outputs

- **validate.rs**: Graph validation logic
  - `validate_graph()`: Checks format version, output presence, reference validity
  - `validate_weights()`: Validates weights manifest against const declarations

- **weights.rs**: Weights manifest handling
  - `WeightsManifest`: External weights file structure
  - `TensorEntry`: Individual tensor metadata (dataType, shape, byteOffset, byteLength)
  - Helper functions: `dtype_size()`, `numel()`

- **emit_js.rs**: JavaScript code generation
  - `emit_builder_js()`: Generates WebNN MLGraphBuilder code from `GraphJson`
  - Outputs async function that builds the graph using WebNN API

- **main.rs**: CLI interface using clap
  - Defines three subcommands: Parse, Validate, EmitJs

### WebNN Graph Language Format

Graph files have a structured format with four main sections:

```
webnn_graph "name" v1 {
  inputs {
    x: f32[1, 2048];
  }
  consts {
    W: f32[2048, 1000] @weights("W");
  }
  nodes {
    result = operation(input1, input2, option=value);
  }
  outputs { result; }
}
```

### Data Flow

1. Parse phase: WG text → `GraphJson`
2. Validate phase: Check structure + optional weights manifest
3. Emit phase: `GraphJson` → WebNN JavaScript code

### Key Invariants

- All node inputs must reference previously defined inputs, consts, or node outputs
- Graph outputs must reference valid node results
- Weights manifest entries must match const declarations in type and shape
- Node IDs must be unique

## Test Coverage

The project has comprehensive test coverage across all modules:

- **ast.rs**: Tests for DataType conversion, OperandDesc equality, ConstInit variants
- **weights.rs**: Tests for dtype_size, numel, and large value handling
- **parser.rs**: Tests for parsing inputs, consts, nodes, outputs, and error handling
- **validate.rs**: Tests for graph validation and weights validation
- **emit_js.rs**: Tests for JavaScript code generation with various graph configurations

Run tests with:
```bash
make test
```

Generate coverage report (requires cargo-tarpaulin):
```bash
make coverage
```

Install development dependencies:
```bash
make dev-deps
```

## Development Notes

- This is a reference scaffold - op semantics should be extended as needed
- The parser uses Pest grammar (src/wg.pest) for parsing WG text
- JSON format uses camelCase for WebNN API compatibility (dataType, byteOffset, etc.)
- Const initializations support three modes: weights references, scalar values, inline bytes
- All code should be formatted with `make fmt` before committing
- Run `make lint` to check for common issues
