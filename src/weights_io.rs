use crate::weights::{dtype_size, numel, TensorEntry, WeightsManifest};
use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"WGWT";
const VERSION: u32 = 1;

/// Pack tensor files into a binary .weights file according to a manifest
pub fn pack_weights(manifest_path: &str, input_dir: &str, output_path: &str) -> Result<()> {
    let manifest_text =
        fs::read_to_string(manifest_path).context("Failed to read manifest file")?;
    let manifest: WeightsManifest =
        serde_json::from_str(&manifest_text).context("Failed to parse manifest JSON")?;

    // Validate manifest format
    if manifest.format != "wg-weights-manifest" {
        bail!("Invalid manifest format: {}", manifest.format);
    }
    if manifest.version != 1 {
        bail!("Unsupported manifest version: {}", manifest.version);
    }

    let mut output = File::create(output_path).context("Failed to create output file")?;

    // Write header
    output.write_all(MAGIC)?;
    output.write_all(&VERSION.to_le_bytes())?;

    // Collect tensors sorted by byte offset
    let mut sorted_tensors: Vec<(&String, &TensorEntry)> = manifest.tensors.iter().collect();
    sorted_tensors.sort_by_key(|(_, entry)| entry.byte_offset);

    let mut current_offset = 8u64; // After magic + version

    for (name, entry) in sorted_tensors {
        // Validate byte offset matches expectation
        if entry.byte_offset != current_offset {
            bail!(
                "Tensor '{}' has unexpected byte_offset: expected {}, got {}",
                name,
                current_offset,
                entry.byte_offset
            );
        }

        // Calculate expected byte length
        let expected_len = numel(&entry.shape) * dtype_size(&entry.data_type);
        if entry.byte_length != expected_len {
            bail!(
                "Tensor '{}' has incorrect byte_length: expected {}, got {}",
                name,
                expected_len,
                entry.byte_length
            );
        }

        // Read tensor data from input directory
        let tensor_path = Path::new(input_dir).join(format!("{}.bin", name));
        let mut tensor_data = Vec::new();
        File::open(&tensor_path)
            .with_context(|| format!("Failed to open tensor file: {:?}", tensor_path))?
            .read_to_end(&mut tensor_data)?;

        // Validate file size
        if tensor_data.len() as u64 != entry.byte_length {
            bail!(
                "Tensor file '{}' has wrong size: expected {} bytes, got {}",
                name,
                entry.byte_length,
                tensor_data.len()
            );
        }

        // Write tensor data
        output.write_all(&tensor_data)?;
        current_offset += entry.byte_length;
    }

    crate::debug_println!(
        "Packed {} tensors ({} bytes) into {}",
        manifest.tensors.len(),
        current_offset,
        output_path
    );

    Ok(())
}

/// Unpack a binary .weights file into individual tensor files
pub fn unpack_weights(weights_path: &str, manifest_path: &str, output_dir: &str) -> Result<()> {
    let manifest_text =
        fs::read_to_string(manifest_path).context("Failed to read manifest file")?;
    let manifest: WeightsManifest =
        serde_json::from_str(&manifest_text).context("Failed to parse manifest JSON")?;

    let mut weights_file = File::open(weights_path).context("Failed to open weights file")?;

    // Read and validate header
    let mut magic = [0u8; 4];
    weights_file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        bail!("Invalid magic bytes: expected {:?}, got {:?}", MAGIC, magic);
    }

    let mut version_bytes = [0u8; 4];
    weights_file.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version != VERSION {
        bail!("Unsupported weights file version: {}", version);
    }

    // Read entire weights data
    let mut weights_data = Vec::new();
    weights_file.read_to_end(&mut weights_data)?;

    // Create output directory
    fs::create_dir_all(output_dir).context("Failed to create output directory")?;

    // Extract each tensor
    for (name, entry) in &manifest.tensors {
        let start = (entry.byte_offset - 8) as usize; // Subtract header size
        let end = start + entry.byte_length as usize;

        if end > weights_data.len() {
            bail!(
                "Tensor '{}' extends beyond weights file: offset={}, length={}, file_size={}",
                name,
                entry.byte_offset,
                entry.byte_length,
                weights_data.len() + 8
            );
        }

        let tensor_data = &weights_data[start..end];
        let output_path = Path::new(output_dir).join(format!("{}.bin", name));

        fs::write(&output_path, tensor_data)
            .with_context(|| format!("Failed to write tensor file: {:?}", output_path))?;

        crate::debug_println!(
            "Extracted tensor '{}': {} bytes ({:?}, shape={:?})",
            name,
            entry.byte_length,
            entry.data_type,
            entry.shape
        );
    }

    crate::debug_println!(
        "Unpacked {} tensors from {} into {}",
        manifest.tensors.len(),
        weights_path,
        output_dir
    );

    Ok(())
}

/// Create a manifest from a directory of tensor files
pub fn create_manifest(input_dir: &str, output_path: &str, endianness: &str) -> Result<()> {
    let mut tensors = BTreeMap::new();
    let mut current_offset = 8u64; // After magic + version

    // Scan directory for .bin files with .meta.json sidecars
    let entries = fs::read_dir(input_dir).context("Failed to read input directory")?;

    let mut tensor_files: Vec<_> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "bin"))
        .collect();

    tensor_files.sort_by_key(|e| e.file_name());

    for entry in tensor_files {
        let path = entry.path();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .context("Invalid filename")?;

        // Read metadata
        let meta_path = path.with_extension("meta.json");
        let meta_text = fs::read_to_string(&meta_path)
            .with_context(|| format!("Failed to read metadata file: {:?}", meta_path))?;
        let meta: TensorEntry =
            serde_json::from_str(&meta_text).context("Failed to parse metadata JSON")?;

        // Get file size
        let file_size = fs::metadata(&path)?.len();
        let expected_size = numel(&meta.shape) * dtype_size(&meta.data_type);

        if file_size != expected_size {
            bail!(
                "Tensor '{}' file size mismatch: expected {}, got {}",
                name,
                expected_size,
                file_size
            );
        }

        // Create tensor entry
        let tensor_entry = TensorEntry {
            data_type: meta.data_type,
            shape: meta.shape,
            byte_offset: current_offset,
            byte_length: file_size,
            layout: meta.layout.or(Some("row-major".to_string())),
        };

        tensors.insert(name.to_string(), tensor_entry);
        current_offset += file_size;
    }

    let manifest = WeightsManifest {
        format: "wg-weights-manifest".to_string(),
        version: 1,
        endianness: endianness.to_string(),
        tensors,
    };

    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    fs::write(output_path, manifest_json).context("Failed to write manifest file")?;

    crate::debug_println!(
        "Created manifest with {} tensors: {}",
        manifest.tensors.len(),
        output_path
    );

    Ok(())
}

/// Extract inline weights from a graph and save them as external files
pub fn extract_weights(
    graph: &crate::ast::GraphJson,
    output_dir: &str,
    weights_file: &str,
    manifest_file: &str,
) -> Result<crate::ast::GraphJson> {
    use crate::ast::{ConstDecl, ConstInit};

    // Create output directory for tensors
    let tensor_dir = Path::new(output_dir).join("tensors");
    fs::create_dir_all(&tensor_dir).context("Failed to create tensor directory")?;

    let mut manifest_tensors = BTreeMap::new();
    let mut new_consts = BTreeMap::new();
    let mut current_offset = 8u64; // After magic + version

    // Process each const
    for (name, const_decl) in &graph.consts {
        match &const_decl.init {
            ConstInit::InlineBytes { bytes } => {
                // Validate byte count matches expected size
                let expected_size = numel(&const_decl.shape) * dtype_size(&const_decl.data_type);
                if bytes.len() as u64 != expected_size {
                    bail!(
                        "Const '{}' has wrong byte count: expected {}, got {}",
                        name,
                        expected_size,
                        bytes.len()
                    );
                }

                // Write tensor to file
                let tensor_path = tensor_dir.join(format!("{}.bin", name));
                fs::write(&tensor_path, bytes)
                    .with_context(|| format!("Failed to write tensor file: {:?}", tensor_path))?;

                // Add to manifest
                manifest_tensors.insert(
                    name.clone(),
                    TensorEntry {
                        data_type: const_decl.data_type.clone(),
                        shape: const_decl.shape.clone(),
                        byte_offset: current_offset,
                        byte_length: bytes.len() as u64,
                        layout: Some("row-major".to_string()),
                    },
                );

                current_offset += bytes.len() as u64;

                // Create new const with weights reference
                new_consts.insert(
                    name.clone(),
                    ConstDecl {
                        data_type: const_decl.data_type.clone(),
                        shape: const_decl.shape.clone(),
                        init: ConstInit::Weights {
                            r#ref: name.clone(),
                        },
                    },
                );

                crate::debug_println!(
                    "Extracted tensor '{}': {} bytes ({:?}, shape={:?})",
                    name,
                    bytes.len(),
                    const_decl.data_type,
                    const_decl.shape
                );
            }
            _ => {
                // Keep non-inline consts as-is
                new_consts.insert(name.clone(), const_decl.clone());
            }
        }
    }

    // Create manifest
    let manifest = WeightsManifest {
        format: "wg-weights-manifest".to_string(),
        version: 1,
        endianness: "little".to_string(),
        tensors: manifest_tensors,
    };

    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    fs::write(manifest_file, manifest_json).context("Failed to write manifest file")?;

    crate::debug_println!(
        "Created manifest with {} tensors: {}",
        manifest.tensors.len(),
        manifest_file
    );

    // Pack weights
    pack_weights(manifest_file, tensor_dir.to_str().unwrap(), weights_file)?;

    // Return new graph with weights references
    let mut new_graph = graph.clone();
    new_graph.consts = new_consts;

    Ok(new_graph)
}

/// Inline external weights into a graph
pub fn inline_weights(
    graph: &crate::ast::GraphJson,
    weights_file: &str,
    manifest_file: &str,
) -> Result<crate::ast::GraphJson> {
    use crate::ast::{ConstDecl, ConstInit};

    // Read manifest
    let manifest_text =
        fs::read_to_string(manifest_file).context("Failed to read manifest file")?;
    let manifest: WeightsManifest =
        serde_json::from_str(&manifest_text).context("Failed to parse manifest JSON")?;

    // Read weights file
    let mut weights_file_handle =
        File::open(weights_file).context("Failed to open weights file")?;

    // Read and validate header
    let mut magic = [0u8; 4];
    weights_file_handle.read_exact(&mut magic)?;
    if &magic != MAGIC {
        bail!("Invalid magic bytes in weights file");
    }

    let mut version_bytes = [0u8; 4];
    weights_file_handle.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version != VERSION {
        bail!("Unsupported weights file version: {}", version);
    }

    // Read all weight data
    let mut weights_data = Vec::new();
    weights_file_handle.read_to_end(&mut weights_data)?;

    // Process consts
    let mut new_consts = BTreeMap::new();

    for (name, const_decl) in &graph.consts {
        match &const_decl.init {
            ConstInit::Weights { r#ref } => {
                // Look up in manifest
                let entry = manifest.tensors.get(r#ref).with_context(|| {
                    format!("Weight reference '{}' not found in manifest", r#ref)
                })?;

                // Extract bytes from weights file
                let start = (entry.byte_offset - 8) as usize; // Subtract header
                let end = start + entry.byte_length as usize;

                if end > weights_data.len() {
                    bail!("Weight reference '{}' extends beyond weights file", r#ref);
                }

                let bytes = weights_data[start..end].to_vec();

                // Create new const with inline bytes
                new_consts.insert(
                    name.clone(),
                    ConstDecl {
                        data_type: const_decl.data_type.clone(),
                        shape: const_decl.shape.clone(),
                        init: ConstInit::InlineBytes { bytes },
                    },
                );

                crate::debug_println!(
                    "Inlined tensor '{}': {} bytes ({:?}, shape={:?})",
                    name,
                    entry.byte_length,
                    entry.data_type,
                    entry.shape
                );
            }
            _ => {
                // Keep non-weight consts as-is
                new_consts.insert(name.clone(), const_decl.clone());
            }
        }
    }

    // Return new graph with inlined bytes
    let mut new_graph = graph.clone();
    new_graph.consts = new_consts;

    Ok(new_graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DataType;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_pack_unpack_roundtrip() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let input_dir = temp_dir.path().join("input");
        let output_dir = temp_dir.path().join("output");
        fs::create_dir(&input_dir)?;

        // Create test tensor files
        let tensor_data = [1.0f32, 2.0, 3.0, 4.0];
        let tensor_bytes: Vec<u8> = tensor_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        fs::write(input_dir.join("test.bin"), &tensor_bytes)?;

        // Create manifest
        let manifest = WeightsManifest {
            format: "wg-weights-manifest".to_string(),
            version: 1,
            endianness: "little".to_string(),
            tensors: {
                let mut map = BTreeMap::new();
                map.insert(
                    "test".to_string(),
                    TensorEntry {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        byte_offset: 8,
                        byte_length: 16,
                        layout: Some("row-major".to_string()),
                    },
                );
                map
            },
        };

        let manifest_path = temp_dir.path().join("manifest.json");
        fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;

        // Pack weights
        let weights_path = temp_dir.path().join("weights.bin");
        pack_weights(
            manifest_path.to_str().unwrap(),
            input_dir.to_str().unwrap(),
            weights_path.to_str().unwrap(),
        )?;

        // Verify packed file has correct header
        let packed_data = fs::read(&weights_path)?;
        assert_eq!(&packed_data[0..4], MAGIC);
        assert_eq!(
            u32::from_le_bytes([
                packed_data[4],
                packed_data[5],
                packed_data[6],
                packed_data[7]
            ]),
            VERSION
        );
        assert_eq!(&packed_data[8..], &tensor_bytes[..]);

        // Unpack weights
        unpack_weights(
            weights_path.to_str().unwrap(),
            manifest_path.to_str().unwrap(),
            output_dir.to_str().unwrap(),
        )?;

        // Verify unpacked data matches original
        let unpacked_data = fs::read(output_dir.join("test.bin"))?;
        assert_eq!(unpacked_data, tensor_bytes);

        Ok(())
    }

    #[test]
    fn test_pack_validates_byte_offsets() {
        let temp_dir = TempDir::new().unwrap();
        let input_dir = temp_dir.path().join("input");
        fs::create_dir(&input_dir).unwrap();

        fs::write(input_dir.join("test.bin"), [0u8; 16]).unwrap();

        let manifest = WeightsManifest {
            format: "wg-weights-manifest".to_string(),
            version: 1,
            endianness: "little".to_string(),
            tensors: {
                let mut map = BTreeMap::new();
                map.insert(
                    "test".to_string(),
                    TensorEntry {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        byte_offset: 100, // Wrong offset!
                        byte_length: 16,
                        layout: Some("row-major".to_string()),
                    },
                );
                map
            },
        };

        let manifest_path = temp_dir.path().join("manifest.json");
        fs::write(&manifest_path, serde_json::to_string(&manifest).unwrap()).unwrap();

        let weights_path = temp_dir.path().join("weights.bin");
        let result = pack_weights(
            manifest_path.to_str().unwrap(),
            input_dir.to_str().unwrap(),
            weights_path.to_str().unwrap(),
        );

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("unexpected byte_offset"));
    }

    #[test]
    fn test_unpack_validates_magic() {
        let temp_dir = TempDir::new().unwrap();
        let weights_path = temp_dir.path().join("bad.bin");

        // Write invalid magic
        let mut file = File::create(&weights_path).unwrap();
        file.write_all(b"XXXX").unwrap();
        file.write_all(&VERSION.to_le_bytes()).unwrap();

        let manifest = WeightsManifest {
            format: "wg-weights-manifest".to_string(),
            version: 1,
            endianness: "little".to_string(),
            tensors: BTreeMap::new(),
        };

        let manifest_path = temp_dir.path().join("manifest.json");
        fs::write(&manifest_path, serde_json::to_string(&manifest).unwrap()).unwrap();

        let output_dir = temp_dir.path().join("output");
        let result = unpack_weights(
            weights_path.to_str().unwrap(),
            manifest_path.to_str().unwrap(),
            output_dir.to_str().unwrap(),
        );

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid magic bytes"));
    }
}
