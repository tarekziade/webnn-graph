/**
 * Helper class for loading and managing WebNN graph weights
 */
export class WeightsFile {
  constructor(buffer, manifest) {
    this.buffer = buffer;
    this.manifest = manifest;
  }

  /**
   * Load weights from URL paths
   * @param {string} weightsPath - Path to .weights binary file
   * @param {string} manifestPath - Path to .manifest.json file
   * @returns {Promise<WeightsFile>}
   */
  static async load(weightsPath, manifestPath) {
    const [weightsResponse, manifestResponse] = await Promise.all([
      fetch(weightsPath),
      fetch(manifestPath)
    ]);

    if (!weightsResponse.ok) {
      throw new Error(`Failed to load weights: ${weightsResponse.statusText}`);
    }
    if (!manifestResponse.ok) {
      throw new Error(`Failed to load manifest: ${manifestResponse.statusText}`);
    }

    const buffer = await weightsResponse.arrayBuffer();
    const manifest = await manifestResponse.json();

    // Validate manifest format
    if (manifest.format !== 'wg-weights-manifest') {
      throw new Error(`Invalid manifest format: ${manifest.format}`);
    }
    if (manifest.version !== 1) {
      throw new Error(`Unsupported manifest version: ${manifest.version}`);
    }

    // Validate weights file header
    const view = new DataView(buffer);
    const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 4));
    if (magic !== 'WGWT') {
      throw new Error(`Invalid weights file magic: ${magic}`);
    }
    const version = view.getUint32(4, true); // little-endian
    if (version !== 1) {
      throw new Error(`Unsupported weights file version: ${version}`);
    }

    return new WeightsFile(buffer, manifest);
  }

  /**
   * Get a slice descriptor for a named tensor
   * @param {string} name - Tensor name
   * @returns {Object} Tensor metadata with byteOffset and byteLength
   */
  getSlice(name) {
    const tensor = this.manifest.tensors[name];
    if (!tensor) {
      throw new Error(`Tensor not found in manifest: ${name}`);
    }
    return tensor;
  }

  /**
   * Get the raw data for a named tensor
   * @param {string} name - Tensor name
   * @returns {ArrayBuffer} Tensor data
   */
  getData(name) {
    const tensor = this.getSlice(name);
    return this.buffer.slice(tensor.byteOffset, tensor.byteOffset + tensor.byteLength);
  }

  /**
   * List all available tensor names
   * @returns {string[]}
   */
  getTensorNames() {
    return Object.keys(this.manifest.tensors);
  }
}

/**
 * Build a WebNN MLGraph from the graph definition
 * @param {MLContext} context - WebNN context
 * @param {WeightsFile} weights - Loaded weights file
 * @returns {Promise<MLGraph>}
 */
export async function buildGraph(context, weights) {
  const builder = new MLGraphBuilder(context);
  const env = new Map();

  env.set("x", builder.input("x", { dataType: "float32", shape: [1, 2048] }));

  {
    const sl = weights.getSlice("W");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("W", builder.constant({ dataType: "float32", shape: [2048, 1000] }, buf));
  }
  {
    const sl = weights.getSlice("b");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("b", builder.constant({ dataType: "float32", shape: [1000] }, buf));
  }

  env.set("logits0", builder["matmul"](env.get("x"), env.get("W"), {}));
  env.set("logits", builder["add"](env.get("logits0"), env.get("b"), {}));
  env.set("probs", builder["softmax"](env.get("logits"), {"axis":1}));

  const outputs = {};
  outputs["probs"] = env.get("probs");
  return await builder.build(outputs);
}
