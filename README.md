# webnn-graph

Rust implementation for a WebNN-oriented graph DSL:

- Parse WebNN graph text (`.webnn`) to canonical JSON
- Validate structure plus optional weights manifest
- Emit WebNN JavaScript builder code (WebNN MLGraphBuilder calls)

This is meant to be a small, hackable reference scaffold: keep the language
surface close to WebNN, but express graphs declaratively.

## Install

### From source (local dev)

Clone and build:

```bash
git clone https://github.com/tarekziade/webnn-graph
cd webnn-graph
make build
```

Then run the CLI:

```bash
make run
# Or for help:
webnn-graph --help
```

### Install the CLI with Cargo

From the repo root:

```bash
cargo install --path .
```

Then:

```bash
webnn-graph --help
```

## Formats

### Text format: .webnn

The text format is block-based and declarative:

- inputs {} declares typed inputs
- consts {} declares typed constants (usually weights)
- nodes {} lists operator calls in order
- outputs {} declares the named graph outputs

Types are written as dtype[dim0, dim1, ...] where dtype is one of:
f32, f16, i32, u32, i64, u64, i8, u8.

### JSON format: .json

The JSON format is canonical (stable for tooling, easy to diff). The text format is sugar over JSON.

## Small example

### Text (examples/resnet_head.webnn)


```
webnn_graph "resnet_head" v1 {
  inputs {
    x: f32[1, 2048];
  }

  consts {
    W: f32[2048, 1000] @weights("W");
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

### JSON (examples/resnet_head.json)


```json
{
  "format": "webnn-graph-json",
  "version": 1,
  "inputs": {
    "x": { "dataType": "float32", "shape": [1, 2048] }
  },
  "consts": {
    "W": {
      "dataType": "float32",
      "shape": [2048, 1000],
      "init": { "kind": "weights", "ref": "W" }
    },
    "b": {
      "dataType": "float32",
      "shape": [1000],
      "init": { "kind": "weights", "ref": "b" }
    }
  },
  "nodes": [
    { "id": "logits0", "op": "matmul", "inputs": ["x", "W"], "options": {} },
    { "id": "logits",  "op": "add",    "inputs": ["logits0", "b"], "options": {} },
    { "id": "probs",   "op": "softmax", "inputs": ["logits"], "options": { "axis": 1 } }
  ],
  "outputs": { "probs": "probs" }
}
```

## Weights manifest (optional)

If you use @weights("key") in .webnn, you can validate the mapping using a manifest such as examples/weights.manifest.json:

```json
{
  "format": "wg-weights-manifest",
  "version": 1,
  "endianness": "little",
  "tensors": {
    "W": {
      "dataType": "float32",
      "shape": [2048, 1000],
      "byteOffset": 0,
      "byteLength": 8192000,
      "layout": "row-major"
    },
    "b": {
      "dataType": "float32",
      "shape": [1000],
      "byteOffset": 8192000,
      "byteLength": 4000,
      "layout": "row-major"
    }
  }
}

```

## CLI

### Graph Operations

Parse graph text (.webnn) to JSON:

```bash
make parse
# Or directly:
webnn-graph parse examples/resnet_head.webnn > graph.json
```

Validate JSON:

```bash
webnn-graph validate graph.json
```

Validate JSON plus weights manifest consistency:

```bash
make validate
# Or directly:
webnn-graph validate graph.json --weights-manifest examples/weights.manifest.json
```

Emit WebNN JS builder code (includes WeightsFile helper):

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

## Complete Workflow

The DSL enables **complete separation of concerns**: graph structure is reusable, weights are external, and input data is provided at runtime.

### 1. Define Graph (Data-Agnostic Template)

Your `.webnn` file contains only structure - no actual data:

```webnn
webnn_graph "resnet_head" v1 {
  inputs {
    x: f32[1, 2048];  // Shape only, no data!
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

### 2. Pack Weights

Combine your trained model weights into a binary file:

```bash
webnn-graph pack-weights \
  --manifest weights.manifest.json \
  --input-dir ./trained_weights/ \
  --output model.weights
```

### 3. Generate Runtime Code

Build the JavaScript module:

```bash
webnn-graph parse model.webnn | \
  webnn-graph emit-js /dev/stdin > buildGraph.js
```

This generates:
- `WeightsFile` class for loading and validating weights
- `buildGraph()` function to construct the WebNN graph

### 4. Use in Browser/Node.js

Reuse the same graph with different input data:

```javascript
import { WeightsFile, buildGraph } from './buildGraph.js';

// One-time setup
const weights = await WeightsFile.load('model.weights', 'weights.manifest.json');
const context = await navigator.ml.createContext();
const graph = await buildGraph(context, weights);

// Reuse with different inputs (no rebuilding!)
const input1 = new Float32Array(2048).fill(1.0);
const result1 = await context.compute(graph, { x: input1 });

const input2 = new Float32Array(2048).fill(2.0);
const result2 = await context.compute(graph, { x: input2 });
```

## Binary Weights Format

The `.weights` file uses a simple binary format:

```
┌─────────────────────────────────────┐
│ Magic: "WGWT" (4 bytes)            │
├─────────────────────────────────────┤
│ Version: 1 (4 bytes, little-endian)│
├─────────────────────────────────────┤
│ Tensor data (concatenated)          │
│ - Tensor 1: bytes at offset         │
│ - Tensor 2: bytes at offset         │
│ ...                                  │
└─────────────────────────────────────┘
```

The `.manifest.json` describes where each tensor lives in the binary file.

## Notes

- The parser and validator are intentionally lightweight and geared toward fast iteration.
- Operator semantics are not deeply validated yet (beyond basic structural checks).
- Extending support is straightforward: add option checks in validation or keep it pass-through and let WebNN runtime validation handle it.

## License

APLv2

