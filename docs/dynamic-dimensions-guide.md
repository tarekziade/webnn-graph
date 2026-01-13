# Dynamic Dimensions Guide

A practical guide for determining dimension override values when converting ONNX models to WebNN format.

## Table of Contents

- [Understanding Dynamic Dimensions](#understanding-dynamic-dimensions)
- [Inspection Methods](#inspection-methods)
- [Common Values by Model Type](#common-values-by-model-type)
- [Decision-Making Process](#decision-making-process)
- [Troubleshooting](#troubleshooting)

## Understanding Dynamic Dimensions

### What Are Dynamic Dimensions?

ONNX models often use symbolic dimensions (like `batch_size`, `sequence_length`) instead of fixed
numbers. This allows the same model to handle different input sizes. However, WebNN requires all
shapes to be known at compile time.

Example ONNX input shape:
```
input_ids: [batch_size, sequence_length]  # Dynamic
```

Must become:
```
input_ids: [1, 128]  # Static
```

### Why Convert to Static Shapes?

WebNN executes in browsers and edge devices where:
- Memory must be allocated upfront
- Graph compilation requires known shapes
- Performance is optimized for specific sizes

## Inspection Methods

### Method 1: Using Python + ONNX

```python
pip install onnxslim
onnxslim --inspect model.onnx
```

### Method 2: Using Netron

1. Install Netron: `pip install netron`
2. Open model: `netron model.onnx`
3. Click on input nodes to see shape information
4. Look for dimension parameters like `batch_size`, `seq_len`, etc.

### Method 3: Check Model Documentation

Most models on Hugging Face include dimension information:

```bash
# Visit the model page
# https://huggingface.co/<org>/<model-name>

# Look for:
# - "Model Details" section
# - "max_seq_length" in config
# - Example usage code
```

### Method 4: Check the Sidecar File

webnn-graph supports automatic dimension discovery via `.dims.json` files:

```bash
# If model.onnx has a model.dims.json file, check it:
cat model.dims.json
```

Example content:
```json
{
  "freeDimensionOverrides": {
    "batch_size": 1,
    "sequence_length": 128
  }
}
```

## Common Values by Model Type

### Text / NLP Models (Transformers, BERT, GPT)

**Typical dimensions:**
```bash
--override-dim batch_size=1
--override-dim sequence_length=<128|256|512>
```

**Choosing sequence_length:**

| Model Type | Recommended | Max | Use Case |
|------------|-------------|-----|----------|
| Sentence embeddings (MiniLM, MPNet) | 128 | 256 | Sentences, titles, short text |
| Document classification (BERT) | 256 | 512 | Paragraphs, articles |
| Question answering (BERT, RoBERTa) | 384 | 512 | Q&A pairs with context |
| Text generation (GPT-2) | 512 | 1024 | Long-form generation |
| Long-document (Longformer) | 1024 | 4096 | Full documents |

**How to determine:**
```python
# Check tokenizer max length
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("model-name")
print(f"Max length: {tokenizer.model_max_length}")
```

**Common dimension parameter names:**
- `batch_size`, `batch`, `N`, `B`
- `sequence_length`, `seq_len`, `max_len`, `T`, `L`
- `hidden_size`, `hidden_dim` (usually fixed, not dynamic)

### Vision Models (CNNs, Vision Transformers)

**Typical dimensions:**
```bash
--override-dim batch_size=1
--override-dim height=224
--override-dim width=224
```

**Standard image sizes by architecture:**

| Architecture | Size | Notes |
|--------------|------|-------|
| ResNet-50/101 | 224×224 | ImageNet standard |
| EfficientNet-B0 | 224×224 | Scales up with variants |
| EfficientNet-B7 | 600×600 | Higher accuracy, slower |
| Vision Transformer (ViT-B) | 224×224 or 384×384 | Two common variants |
| MobileNet V2/V3 | 224×224 | Mobile-optimized |
| YOLO (object detection) | 416×416 or 640×640 | Detection-specific |
| Semantic segmentation | 512×512 or 1024×1024 | Full-resolution |

**How to determine:**
```python
# Check preprocessing configuration
from transformers import AutoImageProcessor
processor = AutoImageProcessor.from_pretrained("model-name")
print(f"Size: {processor.size}")
```

**Common dimension parameter names:**
- `batch_size`, `batch`, `N`, `B`
- `height`, `H`, `image_height`
- `width`, `W`, `image_width`
- `channels`, `C` (usually 3 for RGB, 1 for grayscale)

### Audio Models (Speech Recognition, Audio Classification)

**Typical dimensions:**
```bash
--override-dim batch_size=1
--override-dim sequence_length=<varies>  # Based on audio duration
```

**Determining sequence_length for audio:**
```python
# Formula: sequence_length = sample_rate * duration_seconds / hop_length

# Example for Wav2Vec2 (16kHz, 10 seconds)
sample_rate = 16000
duration = 10  # seconds
hop_length = 320  # model-specific
sequence_length = (sample_rate * duration) // hop_length
# Result: ~500 for 10-second audio
```

**Common values:**
- Whisper: Processes 30-second chunks → sequence_length based on mel spectrogram frames
- Wav2Vec2: Variable based on audio duration
- Audio classification: Often 16000 samples (1 second at 16kHz)

## Decision-Making Process

### Step-by-Step Guide

#### Step 1: Identify Dynamic Dimensions

Try converting without overrides to see what's needed:

```bash
./webnn-graph convert-onnx --input model.onnx
```

Error output will tell you:
```
Error: Dynamic dimensions require explicit overrides:
 - input 'input_ids' dim 'batch_size': --override-dim batch_size=<value>
 - input 'input_ids' dim 'sequence_length': --override-dim sequence_length=<value>
```

#### Step 2: Determine Model Type

Look at the model filename or documentation:
- `*bert*`, `*roberta*`, `*gpt*` → Text model
- `*resnet*`, `*efficientnet*`, `*vit*` → Vision model
- `*wav2vec*`, `*whisper*` → Audio model

#### Step 3: Start with Conservative Values

**Always start with:**
- `batch_size=1` (single inference)
- Smallest reasonable size for other dimensions

**Why start small?**
- Faster conversion and testing
- Less memory usage
- Easier to debug
- Can always increase later

#### Step 4: Look Up Standard Values

Use the tables in [Common Values by Model Type](#common-values-by-model-type) above.

#### Step 5: Test and Iterate

```bash
# Test with initial values
./webnn-graph convert-onnx \
  --input model.onnx \
  --override-dim batch_size=1 \
  --override-dim sequence_length=128

# If successful, test inference
# If shapes are wrong, adjust and retry
```

### Example Workflows

#### Example 1: BERT Sentence Embeddings

```bash
# Model: sentence-transformers/all-MiniLM-L12-v2
# Task: Generate sentence embeddings

# 1. Check documentation
# Hugging Face says: max_seq_length = 256

# 2. Choose conservative value
# Most sentences < 128 tokens, so start there

# 3. Convert
./webnn-graph convert-onnx \
  --input all-MiniLM-L12-v2.onnx \
  --override-dim batch_size=1 \
  --override-dim sequence_length=128 \
  --optimize

# 4. If you need longer sequences, increase
./webnn-graph convert-onnx \
  --input all-MiniLM-L12-v2.onnx \
  --override-dim batch_size=1 \
  --override-dim sequence_length=256 \
  --optimize
```

#### Example 2: ResNet Image Classification

```bash
# Model: ResNet-50
# Task: Image classification

# 1. Standard size for ResNet is 224×224
# 2. No need to check - this is well-known

# 3. Convert
./webnn-graph convert-onnx \
  --input resnet50.onnx \
  --override-dim batch_size=1 \
  --override-dim height=224 \
  --override-dim width=224
```

#### Example 3: Unknown Custom Model

```bash
# 1. Inspect the model
python -c "
import onnx
model = onnx.load('custom_model.onnx')
for inp in model.graph.input:
    print(inp.name, inp.type.tensor_type.shape)
"

# Output shows: data [N, 3, H, W]
# This is an image model (3 channels)

# 2. Try standard image sizes
./webnn-graph convert-onnx \
  --input custom_model.onnx \
  --override-dim N=1 \
  --override-dim H=224 \
  --override-dim W=224

# 3. If conversion fails, check error messages
# 4. Try other common sizes: 256, 299, 384, 512
```

## Troubleshooting

### Problem: "Dynamic dimensions require explicit overrides"

**Solution:** The model has symbolic dimensions that need concrete values.

```bash
# Error shows which dimensions need values
Error: Dynamic dimensions require explicit overrides:
 - input 'input' dim 'height': --override-dim height=<value>

# Provide the missing dimensions
./webnn-graph convert-onnx \
  --input model.onnx \
  --override-dim height=224 \
  --override-dim width=224
```

### Problem: "Shape inference failed"

**Cause:** The dimension values you provided are incompatible with the model's operations.

**Solution:**

1. Check if the model has specific size requirements:
```python
# Some models only work with specific sizes
# e.g., YOLO expects multiples of 32
```

2. Try multiples of common factors:
```bash
# For CNNs: try 224, 256, 288, 320, 384, 416, 512
# For transformers: try 128, 256, 384, 512
```

3. Enable optimization to help with shape inference:
```bash
./webnn-graph convert-onnx \
  --input model.onnx \
  --optimize \
  --override-dim ...
```

### Problem: "Constant folding failed"

**Cause:** The model uses dynamic operations that can't be resolved with the given dimensions.

**Solution:**

1. Make sure you've provided ALL dynamic dimensions:
```bash
# Check for missing dimensions
python -c "
import onnx
model = onnx.load('model.onnx')
for inp in model.graph.input:
    for dim in inp.type.tensor_type.shape.dim:
        if dim.dim_param:
            print(f'Dynamic: {dim.dim_param}')
"
```

2. Use the `--optimize` flag:
```bash
./webnn-graph convert-onnx \
  --input model.onnx \
  --optimize \
  --override-dim batch_size=1 \
  --override-dim sequence_length=128
```

### Problem: Conversion succeeds but inference fails

**Cause:** The dimension values are too small for your actual input data.

**Solution:**

1. Check your actual input size:
```javascript
// In JavaScript
console.log("Input shape:", inputData.shape);
```

2. Increase dimensions to match:
```bash
# If your inputs are 256 tokens but you used 128
./webnn-graph convert-onnx \
  --input model.onnx \
  --override-dim sequence_length=256  # Increased from 128
```

### Problem: Out of memory during conversion

**Cause:** Dimension values are too large.

**Solution:**

1. Reduce to smaller values:
```bash
# Instead of 512, try 256
# Instead of 256, try 128
```

2. Use batch_size=1 (not higher):
```bash
--override-dim batch_size=1  # Always use 1 for inference
```

## Best Practices

### 1. Always Start with batch_size=1

For inference in WebNN (browsers/edge devices):
```bash
--override-dim batch_size=1
```

Only increase batch size if you're doing batch processing server-side.

### 2. Match Your Use Case

Choose dimensions based on your actual inputs:
- Short texts (tweets, titles): `sequence_length=128`
- Medium texts (articles): `sequence_length=256`
- Long texts (documents): `sequence_length=512+`

### 3. Consider Memory Constraints

Larger dimensions = more memory:
```
Memory ∝ batch_size × sequence_length × hidden_size
```

For browser inference, prefer smaller dimensions.

### 4. Use Standard Values When Possible

Standard sizes are well-tested:
- Images: 224, 256, 384, 512
- Text: 128, 256, 512, 1024
- These are powers of 2 or common multiples

### 5. Document Your Choices

Create a `.dims.json` file alongside your model:

```json
{
  "freeDimensionOverrides": {
    "batch_size": 1,
    "sequence_length": 128
  },
  "notes": "128 tokens handles 95% of our sentences. Max length is 256 if needed."
}
```

### 6. Test with Real Data

After conversion, test with actual inputs:
```javascript
// Verify converted model works with your data
const result = await context.compute(graph, {
  input_ids: actualInputIds,  // Your real input
  attention_mask: actualMask
});
```

## Quick Reference

### Text Models
```bash
--override-dim batch_size=1 --override-dim sequence_length=128
```

### Vision Models
```bash
--override-dim batch_size=1 --override-dim height=224 --override-dim width=224
```

### Unknown Model
```bash
# 1. Try without overrides (see error message)
# 2. Check model documentation
# 3. Start with smallest reasonable values
# 4. Iterate based on errors/results
```

## Additional Resources

- [ONNX Model Zoo](https://github.com/onnx/models) - Standard models with known shapes
- [Hugging Face Model Hub](https://huggingface.co/models) - Model documentation
- [Netron](https://netron.app/) - Visual model inspector
- [WebNN Specification](https://www.w3.org/TR/webnn/) - WebNN requirements

## Summary

**The decision process:**
1. Identify dynamic dimensions (try converting without overrides)
2. Determine model type (text/vision/audio)
3. Look up standard values for that model type
4. Start with conservative (small) values
5. Test and iterate as needed

**Most common pattern:**
```bash
./webnn-graph convert-onnx \
  --input model.onnx \
  --optimize \
  --override-dim batch_size=1 \
  --override-dim <other_dims>=<standard_value>
```

Remember: **batch_size=1** is almost always correct for inference!
