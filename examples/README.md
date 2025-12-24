# WebNN Graph Examples

This directory contains a complete working example demonstrating the full workflow of the WebNN graph DSL.

## Contents

- **resnet_head.webnn** - Graph definition (data-agnostic template)
- **weights.manifest.json** - Original manifest (for reference)
- **tensors/** - Example tensor files (W.bin, b.bin with metadata)
- **build_example.sh** - Automated build script
- **browser_example.html** - Interactive browser demo

## Generated Files (after running build script)

- **resnet_head.manifest.json** - Generated weights manifest
- **resnet_head.weights** - Binary weights file (7.8 MB)
- **buildGraph.js** - Generated JavaScript with WeightsFile helper

## Quick Start

### 1. Build the Example

Run the automated build script:

```bash
./examples/build_example.sh
```

This will:
1. Create a manifest from the tensor files
2. Pack the tensors into a binary `.weights` file
3. Generate the JavaScript module

### 2. Run in Browser

Serve the examples directory with a web server:

```bash
# Using Python
python3 -m http.server 8000

# Or using Node.js
npx http-server -p 8000
```

Then open http://localhost:8000/browser_example.html

### 3. Manual Workflow

You can also run each step manually:

```bash
# Step 1: Create manifest from tensors
cargo run -- create-manifest \
  --input-dir examples/tensors \
  --output examples/my_manifest.json \
  --endianness little

# Step 2: Pack weights into binary
cargo run -- pack-weights \
  --manifest examples/my_manifest.json \
  --input-dir examples/tensors \
  --output examples/my_model.weights

# Step 3: Generate JavaScript
cargo run -- parse examples/resnet_head.webnn | \
  cargo run -- emit-js /dev/stdin > examples/my_graph.js

# Optional: Unpack weights for inspection
cargo run -- unpack-weights \
  --weights examples/my_model.weights \
  --manifest examples/my_manifest.json \
  --output-dir examples/unpacked/
```

## Understanding the Example

### Graph Definition (resnet_head.webnn)

The `.webnn` file defines only the **structure** - no actual data:

```webnn
webnn_graph "resnet_head" v1 {
  inputs {
    x: f32[1, 2048];  // Shape only!
  }
  consts {
    W: f32[2048, 1000] @weights("W");  // External reference
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

### Tensor Files (tensors/ directory)

Each weight has:
- **W.bin** - Raw binary data (float32 values)
- **W.meta.json** - Metadata (shape, dataType, layout)

### Generated JavaScript

The `buildGraph.js` contains:

1. **WeightsFile class** - Loads and validates weights
   ```javascript
   const weights = await WeightsFile.load('model.weights', 'manifest.json');
   ```

2. **buildGraph function** - Constructs the WebNN graph
   ```javascript
   const graph = await buildGraph(context, weights);
   ```

### Runtime Usage

The key benefit: **graph reusability**

```javascript
// One-time setup
const weights = await WeightsFile.load('resnet_head.weights', 'manifest.json');
const context = await navigator.ml.createContext();
const graph = await buildGraph(context, weights);

// Run multiple times with different inputs
const result1 = await context.compute(graph, { x: input1 });
const result2 = await context.compute(graph, { x: input2 });
const result3 = await context.compute(graph, { x: input3 });
// ... no rebuilding needed!
```

## File Formats

### Weights Binary Format (.weights)

```
┌─────────────────────────────────────┐
│ Magic: "WGWT" (4 bytes)            │
├─────────────────────────────────────┤
│ Version: 1 (4 bytes, little-endian)│
├─────────────────────────────────────┤
│ W tensor data (8,192,000 bytes)    │
│ b tensor data (4,000 bytes)        │
└─────────────────────────────────────┘
```

### Weights Manifest (.manifest.json)

Describes where each tensor lives in the binary file:

```json
{
  "format": "wg-weights-manifest",
  "version": 1,
  "endianness": "little",
  "tensors": {
    "W": {
      "dataType": "float32",
      "shape": [2048, 1000],
      "byteOffset": 8,
      "byteLength": 8192000
    },
    "b": { ... }
  }
}
```

## Creating Your Own Tensors

To add your own weights:

1. Create binary files (e.g., using Python/NumPy):
   ```python
   import numpy as np

   # Create tensor
   W = np.random.randn(2048, 1000).astype(np.float32)
   W.tofile('W.bin')

   # Create metadata
   import json
   with open('W.meta.json', 'w') as f:
       json.dump({
           "dataType": "float32",
           "shape": [2048, 1000],
           "byteOffset": 8,  # After header
           "byteLength": W.nbytes,
           "layout": "row-major"
       }, f)
   ```

2. Run the build workflow to pack and generate code

## Troubleshooting

**"WebNN API not supported"**
- Use a browser with WebNN support (Chrome/Edge with experimental features enabled)
- Visit `chrome://flags` and enable "Experimental Web Platform features"

**"Failed to load weights"**
- Make sure you're serving the files via HTTP (not file://)
- Check that all generated files exist in the examples directory

**"Invalid magic bytes"**
- Re-run `build_example.sh` to regenerate the weights file
- Ensure the manifest matches the weights file

## Next Steps

- Modify `resnet_head.webnn` to define your own graph
- Replace tensors with your trained model weights
- Use the generated JavaScript in your web application
