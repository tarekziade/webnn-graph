# T5 Model Conversion Guide

This guide covers converting T5 (Text-To-Text Transfer Transformer) models from ONNX to WebNN format.

## Prerequisites

- T5 model exported to ONNX format (separate encoder and decoder files)
- webnn-graph built in release mode: `cargo build --release`

## Supported Operations

The converter supports all operations required for T5 models:

**Constant Folding:**
- Range (positional encoding sequences)
- ConstantOfShape (attention masks)
- Shape, Gather, Concat, Unsqueeze, Squeeze, Cast

**Computation:**
- MatMul, Gemm (attention and feed-forward layers)
- Add, Sub, Mul, Div, Pow, Min, Max (elementwise operations)
- LayerNormalization, Softmax (normalization)
- Greater, Less, Equal, GreaterOrEqual, LessOrEqual (comparisons)
- Where (conditional selection)
- Reshape, Transpose, Split, Concat (tensor manipulation)

## T5 Encoder Conversion

Convert the T5 encoder model:

```bash
./target/release/webnn-graph convert-onnx \
  --input encoder_model.onnx \
  --optimize \
  --override-dim batch_size=1 \
  --override-dim encoder_sequence_length=128
```

**Dimension Overrides:**
- `batch_size=1` - Typical for inference workloads
- `encoder_sequence_length=128` - Standard sequence length for T5-small (adjust for longer sequences)

**Output Files:**
- `encoder_model.webnn` - Graph structure (human-readable)
- `encoder_model.weights` - Binary weights (model parameters)
- `encoder_model.manifest.json` - Weights metadata

## T5 Decoder Conversion

Convert the T5 decoder model:

```bash
./target/release/webnn-graph convert-onnx \
  --input decoder_model.onnx \
  --optimize \
  --override-dim batch_size=1 \
  --override-dim decoder_sequence_length=128 \
  --override-dim encoder_sequence_length=128
```

**Dimension Overrides:**
- `batch_size=1` - Typical for inference
- `decoder_sequence_length=128` - Target sequence length (adjust as needed)
- `encoder_sequence_length=128` - Must match encoder output length (for cross-attention)

**Output Files:**
- `decoder_model.webnn` - Graph structure
- `decoder_model.weights` - Binary weights
- `decoder_model.manifest.json` - Weights metadata

## Model Size Notes

T5 models with constant folding enabled achieve ~40-50% size reduction:
- Dynamic shape operations are resolved at conversion time
- No Shape/Gather/Concat operations in final graph
- All reshape operations use static constants

## Common Dimension Override Values

**T5-small:**
- Encoder: `encoder_sequence_length=128` to `512`
- Decoder: `decoder_sequence_length=128` to `512`

**T5-base:**
- Encoder: `encoder_sequence_length=128` to `512`
- Decoder: `decoder_sequence_length=128` to `512`

**T5-large:**
- Encoder: `encoder_sequence_length=128` to `512`
- Decoder: `decoder_sequence_length=128` to `512`

Choose sequence lengths based on your application:
- Short text (summaries, classification): 128-256
- Medium text (translation, Q&A): 256-512
- Long text (documents): 512+

## Validation

After conversion, validate the graph structure:

```bash
./target/release/webnn-graph validate encoder_model.webnn
./target/release/webnn-graph validate decoder_model.webnn
```

## JavaScript Generation

Generate WebNN JavaScript code for browser/Node.js:

```bash
./target/release/webnn-graph emit-js encoder_model.webnn > buildEncoder.js
./target/release/webnn-graph emit-js decoder_model.webnn > buildDecoder.js
```

## Troubleshooting

**Missing operators:** If conversion fails with "unsupported operator" errors, please report the issue with:
- The operator name
- The model variant (T5-small, T5-base, etc.)
- The conversion command used

**Dynamic dimensions:** All dynamic dimensions must be overridden at conversion time. WebNN requires static
shapes for all operations.

See also: [Dynamic Dimensions Guide](dynamic-dimensions-guide.md)
