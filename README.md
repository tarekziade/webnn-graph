# webnn-graph

`webnn-graph` is a small Rust library and CLI that defines a **WebNN-oriented
graph DSL**, parses it into a minimal AST, and enables multiple downstream uses
such as graph validation, serialization, and WebNN graph construction.

The goal is to keep the language surface **very close to WebNN itself**, while
allowing graphs to be expressed declaratively and reused across tooling.

## Conceptual Model

A WebNN graph defined with this project is split across **three distinct files**, each with a single responsibility.

### 1. Graph definition (`.webnn`)

The `.webnn` file describes **only the structure of the graph**:

- Inputs and their types
- Constants and their shapes
- Operator calls and their wiring
- Named outputs

It contains **no actual tensor data**.

This file is intended to be:
- Small
- Human-readable
- Easy to diff and review
- Stable across weight updates

Its EBNF-like grammar:

```
File         ::= Header Block* EOF

Header       ::= "webnn_graph" String "v" Int "{"

Block        ::= InputsBlock
               | ConstsBlock
               | NodesBlock
               | OutputsBlock
               | "}"              (* closes the graph *)

InputsBlock  ::= "inputs"  "{" InputDecl*  "}"
ConstsBlock  ::= "consts"  "{" ConstDecl*  "}"
NodesBlock   ::= "nodes"   "{" Stmt*       "}"
OutputsBlock ::= "outputs" "{" OutputItem* "}"

InputDecl    ::= Ident ":" Type ";"
ConstDecl    ::= Ident ":" Type ConstAnnot* ";"

OutputItem   ::= Ident ("," Ident)* ";"?    (* optional semicolon *)

Stmt         ::= (MultiAssign | Assign) ";"
Assign       ::= Ident "=" Expr
MultiAssign  ::= "[" Ident ("," Ident)* "]" "=" Expr

Expr         ::= Call | Ident | Literal
Call         ::= Ident "(" Args? ")"

Args         ::= Arg ("," Arg)*
Arg          ::= Ident "=" Value | Value

Value        ::= Literal | Ident

Literal      ::= Array | String | Number | Boolean | Null
Array        ::= "[" (Value ("," Value)*)? "]"

Boolean      ::= "true" | "false"
Null         ::= "null"

Type         ::= DType Shape
DType        ::= "f32" | "f16" | "i32" | "u32" | "i64" | "u64" | "i8" | "u8"
Shape        ::= "[" (Int ("," Int)*)? "]"

ConstAnnot   ::= "@weights" "(" String ")"
               | "@scalar"  "(" Number ")"

Ident        ::= (ALPHA | "_") (ALNUM | "_")*
Int          ::= DIGIT+
Number       ::= "-"? DIGIT+ ("." DIGIT+)? (("e"|"E") ("+"|"-")? DIGIT+)? 
String       ::= "\"" ( "\\\"" | "\\\\" | (ANY-but-quote) )* "\""
```

### 2. Weights manifest (`.manifest.json`, optional)

If the graph references external weights using `@weights("key")`, a manifest file can be provided to:

- Describe tensor shapes and data types
- Define offsets and sizes inside a binary weights file
- Validate that referenced weights are well-formed

The manifest is metadata only. It does not contain raw tensor bytes.

### 3. Binary weights file (`.weights`, optional)

The `.weights` file is a simple concatenation of raw tensor data.

It is:
- Compact
- Fast to load
- Independent from graph structure

This separation allows the same graph definition to be reused with different trained weights.


## Core Idea

The library parses the `.webnn` DSL into a **very small, intentionally simple AST**:

- Inputs
- Constants
- Nodes (operator name, inputs, options)
- Outputs

This AST is the **true internal representation** of a graph.

Once parsed, the AST can be:
- Validated
- Serialized
- Transformed
- Used to construct a WebNN graph

## Using the AST

The AST is designed to be easy to consume from other tools. In particular, it can be used to:

- load, save a build an WebNN graph and its weights using **rustnn** or **PyWebNN**
- Generate WebNN JavaScript `MLGraphBuilder` calls
- Perform lightweight graph analysis or transformations

The library does not attempt to deeply re-specify WebNN semantics. Anything not
explicitly checked is passed through and left to the WebNN runtime to validate.

## JSON Serialization (Secondary)

In addition to the text DSL, the AST can be serialized to a **canonical JSON format**.

Important points:

- JSON is **not** the primary authoring format
- It exists as a convenience for programmatic manipulation
- It supports full round-trip conversion back to `.webnn`
- It can store optional metadata such as the graph name

The JSON format is roughly **10x larger** than the `.webnn` DSL and is best suited for tooling, not manual editing.

All CLI commands auto-detect and accept both formats.


## Features

- Parse WebNN graph text (`.webnn`) into a simple AST
- Serialize the AST to canonical JSON
- Serialize JSON back to `.webnn` with full round-trip support
- Validate graph structure and optional weights manifest
- Emit WebNN JavaScript builder code (`MLGraphBuilder` calls)
- Pack and unpack binary weight files

This is intended as a **small, hackable reference scaffold**, not a heavy framework.

## Install

### From source (local dev)

```bash
git clone https://github.com/tarekziade/webnn-graph
cd webnn-graph
make build
make run
# Or:
webnn-graph --help
```

### Install the CLI with Cargo

```bash
cargo install webnn-graph
```

## Formats

### Text format: .webnn

The DSL is block-based and declarative:

- inputs {} declares typed inputs
- consts {} declares typed constants
- nodes {} lists operator calls in order
- outputs {} declares named graph outputs


Types use:
```
dtype[dim0, dim1, ...]
```

Supported dtypes: `f32`, `f16`, `i32`, `u32`,`i64`, `u64`, `i8`, `u8`.

## Examples

Below is the same graph expressed in webnn and JSON.

### Text 

```webnn
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

### JSON

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

## Notes

- Validation is intentionally lightweight and structural.
- Operator semantics are mostly pass-through.
- The design favors simplicity and reuse over completeness.
- The AST is stable and meant to be consumed by other WebNN tooling.
