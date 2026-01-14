# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Please keep prose line lengths at or below 120 characters when editing markdown/text files.

## Project Overview

webnn-graph is a Rust implementation for a WebNN-oriented graph DSL. It provides a complete pipeline:
1. **Convert ONNX models** to WebNN format (NEW!)
2. Parse WebNN graph text files (.webnn) into JSON representation
3. Serialize JSON back to WebNN text format (full round-trip support)
4. Validate graph structure and weights manifests
5. Emit WebNN JavaScript builder code

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

The binary is named `webnn-graph` with ten subcommands. **Most commands accept both .webnn and .json
formats** (auto-detected).

### Graph Operations

Validate graph structure (accepts .webnn or .json):
```bash
make validate
# Or directly:
webnn-graph validate examples/resnet_head.webnn
webnn-graph validate graph.json --weights-manifest examples/weights.manifest.json
```

Emit JavaScript builder code (accepts .webnn or .json):
```bash
make emit-js
# Or directly:
webnn-graph emit-js examples/resnet_head.webnn > buildGraph.js
webnn-graph emit-js graph.json > buildGraph.js  # Also works
```

Generate interactive HTML visualizer (accepts .webnn or .json):
```bash
make emit-html
# Or directly:
webnn-graph emit-html examples/resnet_head.webnn > visualizer.html
open visualizer.html

# Features:
# - Interactive graph layout with zoom/pan
# - Node details sidebar on click
# - Export to PNG/SVG
# - Light/dark theme toggle
# - Works completely offline (no network requests)
```

Parse graph text to JSON (explicit conversion):
```bash
webnn-graph parse examples/resnet_head.webnn > graph.json
```

Serialize JSON back to WebNN text format (explicit conversion):
```bash
webnn-graph serialize graph.json > model.webnn

# Complete round-trip:
webnn-graph parse model.webnn | webnn-graph serialize /dev/stdin > model_copy.webnn
```

### ONNX Conversion

**Built-in Constant Folding**

The converter includes built-in constant folding (enabled with `--optimize`) that automatically handles dynamic
shape patterns. WebNN does not support dynamic shapes at runtime, so the converter resolves all dynamic dimensions
at conversion time.

**Why this is necessary**: WebNN's `reshape` operation requires the shape parameter to be a constant, not a
dynamically computed value. ONNX models with dynamic shapes use `Shape→Gather→Concat→Reshape` patterns that
must be resolved to static constants at conversion time.

Convert ONNX models to WebNN format with constant folding:
```bash
# Basic conversion with optimization (recommended)
webnn-graph convert-onnx --input model.onnx --optimize \
  --override-dim batch_size=1 \
  --override-dim sequence_length=128

# Output: model.webnn + model.weights + model.manifest.json

# Custom output paths
webnn-graph convert-onnx \
  --input model.onnx \
  --optimize \
  --output graph.webnn \
  --weights graph.weights \
  --manifest graph.manifest.json \
  --override-dim batch_size=1 \
  --override-dim sequence_length=128

# Inline weights for small models
webnn-graph convert-onnx --input model.onnx --optimize --inline-weights \
  --override-dim batch_size=1

# Output to JSON format
webnn-graph convert-onnx --input model.onnx --optimize --output model.json \
  --override-dim batch_size=1 \
  --override-dim sequence_length=128
```

The `--optimize` flag performs constant folding, which:
- Evaluates Shape, Gather, Concat operations at conversion time
- Eliminates dynamic shape computation patterns
- Reduces model size by 40-50% for transformer models
- Makes all reshape operations use static constants

**Supported operators** (NLP/Transformer focused):
- **MatMul, Gemm**: Matrix multiplication with options
- **Add, Sub, Mul, Div, Pow**: Element-wise operations
- **LayerNormalization, Softmax**: Normalization operations
- **Reshape, Transpose, Concat, Split**: Tensor manipulation
- **Constant folding**: Shape, Gather, Concat, Unsqueeze, Squeeze, Cast, Constant

**Full pipeline example**:
```bash
# Step 1: Convert ONNX → WebNN with constant folding
webnn-graph convert-onnx --input bert-base.onnx --optimize \
  --override-dim batch_size=1 \
  --override-dim sequence_length=128

# Step 2: Generate JavaScript
webnn-graph emit-js bert-base.webnn > buildGraph.js

# Example results for BERT models with --optimize:
# - Original: 637 nodes with Shape operations
# - After constant folding: 317 nodes (50% reduction), no Shape operations
# - All reshape operations use static constants
```

**See also**: [Dynamic Dimensions Guide](docs/dynamic-dimensions-guide.md) for help choosing dimension override values.

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

The DSL uses **.webnn as the primary format** (10x smaller than JSON). The DSL is designed for **complete
separation of concerns**: graph structure is reusable, weights are external, and input data is provided at
runtime.

### 1. Define Graph (Data-Agnostic)

Create `model.webnn` (primary format):
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

Build the runtime code directly from .webnn:
```bash
webnn-graph emit-js model.webnn > buildGraph.js
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
  - `GraphJson`: Top-level structure containing inputs, consts, nodes, outputs, and optional quantized flag
  - `OperandDesc`: Describes tensor shape and data type
  - `DataType`: Enum for f32, f16, i4, u4, i32, u32, i64, u64, i8, u8
  - `ConstDecl`: Constant declarations with initialization (weights, scalar, or inline bytes)
  - `Node`: Represents operations with inputs, options, and outputs

- **parser.rs**: Pest-based parser for WebNN text format
  - Uses `wg.pest` grammar file
  - `parse_wg_text()`: Main entry point converting WebNN text to `GraphJson`
  - Handles four main blocks: inputs, consts, nodes, outputs
  - Extracts and stores graph name from header

- **serialize.rs**: WebNN text format serializer
  - `serialize_graph_to_wg_text()`: Main entry point converting `GraphJson` to WebNN text
  - Generates properly formatted .webnn files with 2-space indentation
  - Handles all const initializations (weights, scalar, inline bytes)
  - Supports multi-output nodes and options serialization
  - `SerializeError`: Error type for serialization failures

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

**Note**: For quantized graphs, add the `@quantized` annotation in the header:
```
webnn_graph "name" v1 @quantized {
  ...
}
```

This sets the `quantized` field to `true` in the JSON representation, indicating the graph contains quantized
weights or operations using Int4/Uint4 data types.

### Data Flow

**Primary workflow** (.webnn is 10x smaller than JSON):
1. Author: Write `.webnn` text files (human-readable, compact)
2. Validate: `webnn-graph validate model.webnn` (parses internally)
3. Emit: `webnn-graph emit-js model.webnn` (generates JavaScript)

**Optional conversions**:
- Parse: `.webnn` → `.json` (for programmatic manipulation)
- Serialize: `.json` → `.webnn` (for human editing)

The tool supports full round-tripping: text ↔ JSON, preserving semantics.

### Key Invariants

- All node inputs must reference previously defined inputs, consts, or node outputs
- Graph outputs must reference valid node results
- Weights manifest entries must match const declarations in type and shape
- Node IDs must be unique
- Graph name is optional in JSON but will default to "graph" when serializing

## Test Coverage

The project has comprehensive test coverage across all modules (50 tests total):

- **ast.rs**: Tests for DataType conversion, OperandDesc equality, ConstInit variants
- **weights.rs**: Tests for dtype_size, numel, and large value handling
- **weights_io.rs**: Tests for pack/unpack roundtrip, validation, and error handling
- **parser.rs**: Tests for parsing inputs, consts, nodes, outputs, and error handling
- **serialize.rs**: Tests for serialization, round-trip preservation, string escaping, all data types
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
