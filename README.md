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

## Development

### Building and Testing

Build the project:
```bash
make build
```

Run tests:
```bash
make test
```

Format code:
```bash
make fmt
```

Run linter:
```bash
make lint
```

Clean build artifacts:
```bash
make clean
```

View all available commands:
```bash
make help
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

### JSON format

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


```jspn
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

Emit WebNN JS builder code:

```bash
make emit-js
# Or directly:
webnn-graph emit-js graph.json > buildGraph.js
```

## Output JS expectations

The generated JS expects a weights helper with:

- weights.buffer (an ArrayBuffer containing concatenated tensor bytes)
- weights.getSlice(key) returning { byteOffset, byteLength }

## Notes

- The parser and validator are intentionally lightweight and geared toward fast iteration.
- Operator semantics are not deeply validated yet (beyond basic structural checks).
- Extending support is straightforward: add option checks in validation or keep it pass-through and let WebNN runtime validation handle it.

## License

APLv2

