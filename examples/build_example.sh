#!/bin/bash
# Complete workflow example for WebNN graph with weights

set -e

# Change to repository root
cd "$(dirname "$0")/.."

echo "=== WebNN Graph Complete Workflow Example ==="
echo

# Step 1: Create manifest from tensor directory
echo "Step 1: Creating weights manifest from tensors..."
cargo run --quiet -- create-manifest \
  --input-dir examples/tensors \
  --output examples/resnet_head.manifest.json \
  --endianness little
echo

# Step 2: Pack weights into binary file
echo "Step 2: Packing weights into binary format..."
cargo run --quiet -- pack-weights \
  --manifest examples/resnet_head.manifest.json \
  --input-dir examples/tensors \
  --output examples/resnet_head.weights
echo

# Step 3: Parse graph and emit JavaScript
echo "Step 3: Generating JavaScript code..."
cargo run --quiet -- parse examples/resnet_head.webnn | \
  cargo run --quiet -- emit-js /dev/stdin > examples/buildGraph.js
echo "Generated examples/buildGraph.js"
echo

# Step 4: Show file sizes
echo "=== Generated Files ==="
ls -lh examples/resnet_head.manifest.json examples/resnet_head.weights examples/buildGraph.js | awk '{print $9, "-", $5}'
echo

echo "=== Usage ==="
echo "The generated files can be used in a browser:"
echo "  - buildGraph.js: Contains WeightsFile class and buildGraph() function"
echo "  - resnet_head.weights: Binary weights file (8.0 MB)"
echo "  - resnet_head.manifest.json: Weights metadata"
echo
echo "Example JavaScript:"
echo "  import { WeightsFile, buildGraph } from './buildGraph.js';"
echo "  const weights = await WeightsFile.load('resnet_head.weights', 'resnet_head.manifest.json');"
echo "  const context = await navigator.ml.createContext();"
echo "  const graph = await buildGraph(context, weights);"
echo "  const result = await context.compute(graph, { x: inputData });"
echo

echo "=== Optional: Unpack weights for inspection ==="
echo "cargo run -- unpack-weights \\"
echo "  --weights examples/resnet_head.weights \\"
echo "  --manifest examples/resnet_head.manifest.json \\"
echo "  --output-dir examples/unpacked/"
