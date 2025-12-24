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

The binary is named `webnn-graph` with six subcommands:

### Graph Operations

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

Emit JavaScript builder code (includes WeightsFile helper):
```bash
make emit-js
# Or directly:
webnn-graph emit-js graph.json > buildGraph.js
```

### Weights Management

Pack tensor files into binary .weights file:
```bash
webnn-graph pack-weights \
  --manifest weights.manifest.json \
  --input-dir ./tensors/ \
  --output model.weights
```

Unpack binary weights for inspection:
```bash
webnn-graph unpack-weights \
  --weights model.weights \
  --manifest weights.manifest.json \
  --output-dir ./extracted/
```

Create manifest from tensor directory:
```bash
webnn-graph create-manifest \
  --input-dir ./tensors/ \
  --output weights.manifest.json \
  --endianness little
```

## Complete Workflow Example

The DSL is designed for **complete separation of concerns**: graph structure is reusable, weights are external, and input data is provided at runtime.

### 1. Define Graph (Data-Agnostic)

Create `model.webnn`:
```webnn
webnn_graph "resnet_head" v1 {
  inputs {
    x: f32[1, 2048];  // Shape only, no actual data
  }
  consts {
    W: f32[2048, 1000] @weights("W");  // Reference to external weights
    b: f32[1000]       @weights("b");
  }
  nodes {
    logits0 = matmul(x, W);
    logits  = add(logits0, b);
    probs   = softmax(logits, axis=1);
  }
  outputs { probs; }
}
```

### 2. Prepare Weights

Pack your trained weights:
```bash
# Assuming you have W.bin and b.bin tensor files with W.meta.json and b.meta.json
webnn-graph pack-weights \
  --manifest weights.manifest.json \
  --input-dir ./trained_weights/ \
  --output model.weights
```

### 3. Generate JavaScript

Build the runtime code:
```bash
webnn-graph parse model.webnn | \
  webnn-graph emit-js /dev/stdin > buildGraph.js
```

This generates both the `WeightsFile` helper class and the `buildGraph()` function.

### 4. Runtime Usage (Browser/Node.js)

Use the graph with different input data:
```javascript
import { WeightsFile, buildGraph } from './buildGraph.js';

// One-time setup: load graph structure + weights
const weights = await WeightsFile.load('model.weights', 'weights.manifest.json');
const context = await navigator.ml.createContext();
const graph = await buildGraph(context, weights);

// Reuse graph with different input data (many times!)
const input1 = new Float32Array(2048).fill(1.0);
const result1 = await context.compute(graph, { x: input1 });

const input2 = new Float32Array(2048).fill(2.0);
const result2 = await context.compute(graph, { x: input2 });

// Same graph, different data - no rebuilding needed!
```

## Architecture

### Module Structure

- **ast.rs**: Core data structures for the graph JSON format
  - `GraphJson`: Top-level structure containing inputs, consts, nodes, outputs
  - `OperandDesc`: Describes tensor shape and data type
  - `DataType`: Enum for f32, f16, i32, u32, i64, u64, i8, u8
  - `ConstDecl`: Constant declarations with initialization (weights, scalar, or inline bytes)
  - `Node`: Represents operations with inputs, options, and outputs

- **parser.rs**: Pest-based parser for WebNN text format
  - Uses `wg.pest` grammar file
  - `parse_wg_text()`: Main entry point converting WebNN text to `GraphJson`
  - Handles four main blocks: inputs, consts, nodes, outputs

- **validate.rs**: Graph validation logic
  - `validate_graph()`: Checks format version, output presence, reference validity
  - `validate_weights()`: Validates weights manifest against const declarations

- **weights.rs**: Weights manifest handling
  - `WeightsManifest`: External weights file structure
  - `TensorEntry`: Individual tensor metadata (dataType, shape, byteOffset, byteLength)
  - Helper functions: `dtype_size()`, `numel()`

- **weights_io.rs**: Binary weights packing/unpacking
  - `pack_weights()`: Combine tensor files into binary .weights format
  - `unpack_weights()`: Extract tensors from binary weights file
  - `create_manifest()`: Generate manifest from tensor directory
  - Binary format: 4-byte magic "WGWT", 4-byte version, concatenated tensor data

- **emit_js.rs**: JavaScript code generation
  - `emit_weights_loader_js()`: Generates WeightsFile helper class
  - `emit_builder_js()`: Generates WebNN MLGraphBuilder code from `GraphJson`
  - Outputs complete module with loading, validation, and graph building

- **main.rs**: CLI interface using clap
  - Defines six subcommands: Parse, Validate, EmitJs, PackWeights, UnpackWeights, CreateManifest

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

1. Parse phase: WebNN text (.webnn) → `GraphJson` (.json)
2. Validate phase: Check structure + optional weights manifest
3. Emit phase: `GraphJson` → WebNN JavaScript code

### Key Invariants

- All node inputs must reference previously defined inputs, consts, or node outputs
- Graph outputs must reference valid node results
- Weights manifest entries must match const declarations in type and shape
- Node IDs must be unique

## Test Coverage

The project has comprehensive test coverage across all modules (40 tests total):

- **ast.rs**: Tests for DataType conversion, OperandDesc equality, ConstInit variants
- **weights.rs**: Tests for dtype_size, numel, and large value handling
- **weights_io.rs**: Tests for pack/unpack roundtrip, validation, and error handling
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
- The parser uses Pest grammar (src/wg.pest) for parsing WebNN text (.webnn files)
- JSON format uses camelCase for WebNN API compatibility (dataType, byteOffset, etc.)
- Const initializations support three modes: weights references, scalar values, inline bytes
- All code should be formatted with `make fmt` before committing
- Run `make lint` to check for common issues
