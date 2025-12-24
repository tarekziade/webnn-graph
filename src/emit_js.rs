use crate::ast::{DataType, GraphJson};

fn dt_to_js(dt: &DataType) -> &'static str {
    match dt {
        DataType::Float32 => "float32",
        DataType::Float16 => "float16",
        DataType::Int32 => "int32",
        DataType::Uint32 => "uint32",
        DataType::Int64 => "int64",
        DataType::Uint64 => "uint64",
        DataType::Int8 => "int8",
        DataType::Uint8 => "uint8",
    }
}

pub fn emit_builder_js(g: &GraphJson) -> String {
    let mut s = String::new();
    s.push_str("export async function buildGraph(context, weights) {\n");
    s.push_str("  const builder = new MLGraphBuilder(context);\n");
    s.push_str("  const env = new Map();\n\n");

    for (name, d) in &g.inputs {
        let shape = format!("{:?}", d.shape);
        s.push_str(&format!(
            "  env.set({name:?}, builder.input({name:?}, {{ dataType: {dt:?}, shape: {shape} }}));\n",
            name = name,
            dt = dt_to_js(&d.data_type),
            shape = shape,
        ));
    }
    s.push('\n');

    for (name, c) in &g.consts {
        match &c.init {
            crate::ast::ConstInit::Weights { r#ref } => {
                let shape = format!("{:?}", c.shape);
                s.push_str(&format!(
"  {{\n    const sl = weights.getSlice({r:?});\n    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);\n    env.set({name:?}, builder.constant({{ dataType: {dt:?}, shape: {shape} }}, buf));\n  }}\n",
                    r = r#ref,
                    name = name,
                    dt = dt_to_js(&c.data_type),
                    shape = shape,
                ));
            }
            crate::ast::ConstInit::Scalar { value } => {
                s.push_str(&format!(
                    "  env.set({name:?}, builder.constant({dt:?}, {val}));\n",
                    name = name,
                    dt = dt_to_js(&c.data_type),
                    val = value,
                ));
            }
            crate::ast::ConstInit::InlineBytes { bytes } => {
                let shape = format!("{:?}", c.shape);
                s.push_str(&format!(
                    "  env.set({name:?}, builder.constant({{ dataType: {dt:?}, shape: {shape} }}, new Uint8Array({bytes:?}).buffer));\n",
                    name = name,
                    dt = dt_to_js(&c.data_type),
                    shape = shape,
                    bytes = bytes
                ));
            }
        }
    }
    s.push('\n');

    for n in &g.nodes {
        let ins = n
            .inputs
            .iter()
            .map(|x| format!("env.get({:?})", x))
            .collect::<Vec<_>>()
            .join(", ");
        let opts = serde_json::Value::Object(n.options.clone()).to_string();
        if let Some(outs) = &n.outputs {
            s.push_str(&format!(
                "  {{\n    const tmp = builder[{op:?}]({ins}, {opts});\n",
                op = n.op,
                ins = ins,
                opts = opts
            ));
            for (i, o) in outs.iter().enumerate() {
                s.push_str(&format!("    env.set({o:?}, tmp[{i}]);\n", o = o, i = i));
            }
            s.push_str("  }\n");
        } else {
            s.push_str(&format!(
                "  env.set({id:?}, builder[{op:?}]({ins}, {opts}));\n",
                id = n.id,
                op = n.op,
                ins = ins,
                opts = opts
            ));
        }
    }

    s.push_str("\n  const outputs = {};\n");
    for (out, r) in &g.outputs {
        s.push_str(&format!(
            "  outputs[{out:?}] = env.get({r:?});\n",
            out = out,
            r = r
        ));
    }
    s.push_str("  return await builder.build(outputs);\n");
    s.push_str("}\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{new_graph_json, ConstDecl, ConstInit, DataType, Node, OperandDesc};

    #[test]
    fn test_dt_to_js() {
        assert_eq!(dt_to_js(&DataType::Float32), "float32");
        assert_eq!(dt_to_js(&DataType::Float16), "float16");
        assert_eq!(dt_to_js(&DataType::Int32), "int32");
        assert_eq!(dt_to_js(&DataType::Uint32), "uint32");
        assert_eq!(dt_to_js(&DataType::Int64), "int64");
        assert_eq!(dt_to_js(&DataType::Uint64), "uint64");
        assert_eq!(dt_to_js(&DataType::Int8), "int8");
        assert_eq!(dt_to_js(&DataType::Uint8), "uint8");
    }

    #[test]
    fn test_emit_simple_graph() {
        let mut g = new_graph_json();
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: vec![1, 10],
            },
        );
        g.nodes.push(Node {
            id: "result".to_string(),
            op: "relu".to_string(),
            inputs: vec!["x".to_string()],
            options: serde_json::Map::new(),
            outputs: None,
        });
        g.outputs.insert("result".to_string(), "result".to_string());

        let js = emit_builder_js(&g);
        assert!(js.contains("export async function buildGraph"));
        assert!(js.contains("MLGraphBuilder(context)"));
        assert!(js.contains("builder.input(\"x\""));
        assert!(js.contains("builder[\"relu\"]"));
        assert!(js.contains("env.get(\"x\")"));
        assert!(js.contains("outputs[\"result\"]"));
        assert!(js.contains("builder.build(outputs)"));
    }

    #[test]
    fn test_emit_with_weights() {
        let mut g = new_graph_json();
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: vec![1, 10],
            },
        );
        g.consts.insert(
            "W".to_string(),
            ConstDecl {
                data_type: DataType::Float32,
                shape: vec![10, 5],
                init: ConstInit::Weights {
                    r#ref: "W".to_string(),
                },
            },
        );
        g.nodes.push(Node {
            id: "result".to_string(),
            op: "matmul".to_string(),
            inputs: vec!["x".to_string(), "W".to_string()],
            options: serde_json::Map::new(),
            outputs: None,
        });
        g.outputs.insert("result".to_string(), "result".to_string());

        let js = emit_builder_js(&g);
        assert!(js.contains("weights.getSlice(\"W\")"));
        assert!(js.contains("weights.buffer.slice"));
        assert!(js.contains("builder.constant"));
        assert!(js.contains("dataType: \"float32\""));
        assert!(js.contains("shape: [10, 5]"));
    }

    #[test]
    fn test_emit_with_scalar() {
        let mut g = new_graph_json();
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: vec![1],
            },
        );
        g.consts.insert(
            "scale".to_string(),
            ConstDecl {
                data_type: DataType::Float32,
                shape: vec![1],
                init: ConstInit::Scalar {
                    value: serde_json::json!(2.5),
                },
            },
        );
        g.nodes.push(Node {
            id: "result".to_string(),
            op: "mul".to_string(),
            inputs: vec!["x".to_string(), "scale".to_string()],
            options: serde_json::Map::new(),
            outputs: None,
        });
        g.outputs.insert("result".to_string(), "result".to_string());

        let js = emit_builder_js(&g);
        assert!(js.contains("builder.constant(\"float32\", 2.5)"));
    }

    #[test]
    fn test_emit_with_inline_bytes() {
        let mut g = new_graph_json();
        g.consts.insert(
            "data".to_string(),
            ConstDecl {
                data_type: DataType::Uint8,
                shape: vec![4],
                init: ConstInit::InlineBytes {
                    bytes: vec![1, 2, 3, 4],
                },
            },
        );
        g.outputs.insert("data".to_string(), "data".to_string());

        let js = emit_builder_js(&g);
        assert!(js.contains("new Uint8Array([1, 2, 3, 4]).buffer"));
    }

    #[test]
    fn test_emit_with_options() {
        let mut g = new_graph_json();
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: vec![1, 10],
            },
        );

        let mut options = serde_json::Map::new();
        options.insert("axis".to_string(), serde_json::json!(1));

        g.nodes.push(Node {
            id: "result".to_string(),
            op: "softmax".to_string(),
            inputs: vec!["x".to_string()],
            options,
            outputs: None,
        });
        g.outputs.insert("result".to_string(), "result".to_string());

        let js = emit_builder_js(&g);
        assert!(js.contains("builder[\"softmax\"]"));
        assert!(js.contains("\"axis\":1"));
    }

    #[test]
    fn test_emit_with_multi_outputs() {
        let mut g = new_graph_json();
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: vec![10],
            },
        );
        g.nodes.push(Node {
            id: "a".to_string(),
            op: "split".to_string(),
            inputs: vec!["x".to_string()],
            options: serde_json::Map::new(),
            outputs: Some(vec!["a".to_string(), "b".to_string()]),
        });
        g.outputs.insert("a".to_string(), "a".to_string());
        g.outputs.insert("b".to_string(), "b".to_string());

        let js = emit_builder_js(&g);
        assert!(js.contains("const tmp = builder[\"split\"]"));
        assert!(js.contains("env.set(\"a\", tmp[0])"));
        assert!(js.contains("env.set(\"b\", tmp[1])"));
    }

    #[test]
    fn test_emit_multiple_inputs_outputs() {
        let mut g = new_graph_json();
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: vec![1],
            },
        );
        g.inputs.insert(
            "y".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: vec![1],
            },
        );
        g.nodes.push(Node {
            id: "a".to_string(),
            op: "relu".to_string(),
            inputs: vec!["x".to_string()],
            options: serde_json::Map::new(),
            outputs: None,
        });
        g.nodes.push(Node {
            id: "b".to_string(),
            op: "sigmoid".to_string(),
            inputs: vec!["y".to_string()],
            options: serde_json::Map::new(),
            outputs: None,
        });
        g.outputs.insert("out1".to_string(), "a".to_string());
        g.outputs.insert("out2".to_string(), "b".to_string());

        let js = emit_builder_js(&g);
        assert!(js.contains("builder.input(\"x\""));
        assert!(js.contains("builder.input(\"y\""));
        assert!(js.contains("outputs[\"out1\"] = env.get(\"a\")"));
        assert!(js.contains("outputs[\"out2\"] = env.get(\"b\")"));
    }
}
