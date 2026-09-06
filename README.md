# ggml-extractor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![llama.cpp](https://img.shields.io/badge/engine-llama.cpp-green.svg)](https://github.com/ggerganov/llama.cpp)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/ggerganov/llama.cpp)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](include/ggml_extractor/version.hpp)

<p align="center">
  <img src="assets/logo.png" alt="ggml-extractor logo — stacked LLM layers with the selected token extracted into a .npy grid" width="580"/>
</p>

> **Created by Giulio Enzo Donninelli and [Adversal.ai](https://adversal.ai).**

A tiny, neat, open-source C++17 library that extracts hidden states for
**specific tokens at specific layers** from any `llama.cpp` / ggml inference
run — and saves them as NumPy `.npy` files.

You instantiate **one object**, pass it a vector of
*(tensor name → token position)* requests, plug it into a `llama_context`,
decode, and save. Prefill (decode) and autoregressive generation (inference)
are both first-class.

```cpp
#include "ggml_extractor/extractor.hpp"
#include "ggml_extractor/llama_session.hpp"
using namespace ggml_extractor;

HiddenStateExtractor extractor({
    {"inp_scaled", TokenSelector::last()},   // input embeddings, last prompt token
    {"l_out-20",   TokenSelector::last()},   // layer 20 output, last token
    {"l_out-29",   TokenSelector::second_last()},
});

LlamaSession session("model.gguf", extractor);
session.decode(session.tokenize("hello world"));
extractor.require_frame_complete();
extractor.commit_frame();
extractor.save_npy("hidden_states.npy");  // shape: (1, 3, n_embd)
```

---

## Table of contents

- [ggml-extractor](#ggml-extractor)
  - [Table of contents](#table-of-contents)
  - [Why](#why)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Build](#build)
  - [Usage](#usage)
    - [1. Prefill — one prompt](#1-prefill--one-prompt)
    - [2. Batch — many prompts, one file](#2-batch--many-prompts-one-file)
    - [3. Generation — current generated token](#3-generation--current-generated-token)
    - [4. Raw ggml — no llama.cpp wrapper](#4-raw-ggml--no-llamacpp-wrapper)
  - [API reference](#api-reference)
    - [`TokenSelector` — `include/ggml_extractor/token_selector.hpp`](#tokenselector--includeggml_extractortoken_selectorhpp)
    - [`ExtractionRequest` — `include/ggml_extractor/extraction_request.hpp`](#extractionrequest--includeggml_extractorextraction_requesthpp)
    - [`HiddenStateExtractor` — `include/ggml_extractor/extractor.hpp`](#hiddenstateextractor--includeggml_extractorextractorhpp)
    - [`LlamaSession` — `include/ggml_extractor/llama_session.hpp`](#llamasession--includeggml_extractorllama_sessionhpp)
    - [`NpyWriter` — `include/ggml_extractor/npy_writer.hpp`](#npywriter--includeggml_extractornpy_writerhpp)
  - [Specification](#specification)
    - [Tensor naming](#tensor-naming)
    - [Token indexing](#token-indexing)
    - [Shapes](#shapes)
    - [`.npy` format](#npy-format)
    - [Lifecycle contract](#lifecycle-contract)
    - [Validation performed per tensor](#validation-performed-per-tensor)
    - [Context \& positions (generation)](#context--positions-generation)
    - [Threading \& determinism](#threading--determinism)
  - [Examples](#examples)
  - [Project layout](#project-layout)
  - [Provenance](#provenance)
  - [License](#license)
  - [Authors](#authors)

---

## Why

`llama.cpp` exposes per-tensor callbacks (`cb_eval`), and ggml tensors carry
canonical names (e.g. `inp_scaled`, `l_out-20`). Capturing the right row of the
right tensor at the right decode step is only a few dozen lines — but every
project re-implements it slightly differently, with slightly different bugs
(non-contiguous rows, wrong token index, silent overwrites of `.npy`).

`ggml-extractor` is the generalized, tested shape of two proven tools
(`extract_hidden_states`, `extract_query_hidden_states`): exact-name matching,
from-end token indexing, atomic `.npy` output — packaged as a plug-and-play
object-oriented library.

---

## Features

- 🧩 **Object-oriented, plug-and-play** — one `HiddenStateExtractor`
  constructed with `std::vector<ExtractionRequest>`; attach to any
  `llama_context_params` via `attach()`.
- 🎯 **Exact tensor names** — the same strings that describe tensors in the
  ggml file / graph (`inp_scaled`, `l_out-20`, …). Case-sensitive equality,
  no glob magic.
- 🔢 **Per-request token selection** — `last()`, `second_last()`,
  `third_last()`, `from_end(k)`, `generated()`. One decode can capture the
  last token of layer 20 *and* the second-to-last of layer 25.
- 🔁 **Decode + inference** — prefill batches *and* token-by-token generation
  loops with correct KV-cache position tracking (`decode` / `decode_one`).
- 💾 **Safe `.npy` output** — float32 `<f4`, v1.0, C-order, 64-byte aligned
  header, streaming or one-shot API, atomic publish, never overwrites.
- 🧵 **Explicit failure modes** — `noexcept` callback records errors;
  `require_frame_complete()` re-throws with tensor + token details; finite
  checks; overflow checks; width-consistency checks.
- 📦 **Minimal dependencies** — C++17, `llama.cpp` (as `LLAMA_CPP_DIR`),
  threads. No Python, no frameworks.

---

## Requirements

| Requirement | Notes |
|---|---|
| C++17 compiler | Clang 14+, GCC 11+, MSVC 19.30+ |
| CMake ≥ 3.20 | |
| llama.cpp checkout | Any recent revision exposing `cb_eval`, `llama_batch_get_one`, `llama_get_logits`, `llama_synchronize` |
| Threads | `find_package(Threads)` |
| GGUF model | Any architecture (optional `required_architecture` guard) |

---

## Build

```bash
git clone https://github.com/<you>/ggml-extractor.git
cmake -S ggml-extractor -B ggml-extractor/build \
  -DLLAMA_CPP_DIR=/path/to/llama.cpp \
  -DCMAKE_BUILD_TYPE=Release
cmake --build ggml-extractor/build -j
```

This produces:

| Target | Binary |
|---|---|
| `ggml_extractor` | static library |
| `prefill_example` | `build/prefill_example` |
| `batch_example` | `build/batch_example` |
| `generate_example` | `build/generate_example` |
| `raw_ggml_example` | `build/raw_ggml_example` |

Install (optional):

```bash
cmake --install ggml-extractor/build --prefix /usr/local
```

---

## Usage

### 1. Prefill — one prompt

Decode-phase extraction. `from_end(0)` is the last prompt token,
`from_end(1)` the second-to-last, and so on.

```cpp
HiddenStateExtractor extractor({
    {"inp_scaled", TokenSelector::last()},
    {"l_out-20",   TokenSelector::last()},
    {"l_out-25",   TokenSelector::from_end(1)},
});
LlamaSession session("model.gguf", extractor);

session.decode(session.tokenize("The cat sat on the mat."));
extractor.require_frame_complete();  // throws if any row is missing
extractor.commit_frame();            // 1 frame
extractor.save_npy("prefill.npy");   // shape: (1, 3, n_embd)
```

CLI:

```bash
./build/prefill_example -m model.gguf -p "hello world" -o out.npy
# → wrote out.npy shape=(3, 3840)
```

### 2. Batch — many prompts, one file

Each prompt is an independent prefill (KV memory cleared between rows); each
contributes one frame.

```cpp
HiddenStateExtractor extractor({
    {"l_out-20", TokenSelector::last()},
    {"l_out-20", TokenSelector::second_last()},  // same tensor, two rows
    {"l_out-29", TokenSelector::last()},
});
LlamaSession session(model, extractor);

for (auto& prompt : prompts) {
    session.decode(session.tokenize(prompt));
    extractor.require_frame_complete();
    extractor.commit_frame();
}
extractor.save_npy("batch.npy");  // (n_prompts, 3, n_embd)
```

```bash
./build/batch_example -m model.gguf -o batch.npy -p "first" -p "second"
```

### 3. Generation — current generated token

Inference-phase extraction. Requests use `TokenSelector::generated()`; the
session preserves the KV cache and advances the position counter, so each
`decode_one()` captures the hidden state of **the token just generated**.

```cpp
HiddenStateExtractor extractor({
    {"l_out-20", TokenSelector::generated()},
    {"l_out-29", TokenSelector::generated()},
});
LlamaSessionOptions opts;
opts.n_ctx = 4096;  // fixed size → KV cache survives across steps
LlamaSession session(model, extractor, opts);

// Prefill (frame discarded, position retained).
session.decode(session.tokenize(prompt));
extractor.require_frame_complete();
extractor.begin_frame();  // drop prefill capture, keep KV + position

// Autoregressive loop (greedy argmax shown; use any sampler):
const llama_vocab* vocab = llama_model_get_vocab(session.model_handle());
for (int step = 0; step < n_predict; ++step) {
    int32_t next = argmax(llama_get_logits(session.context_handle()));
    session.decode_one(next);          // copies the GENERATED row
    extractor.require_frame_complete();
    extractor.commit_frame();          // 1 frame per generated token
    if (next == llama_vocab_eos(vocab)) break;
}
extractor.save_npy("generated.npy");   // (n_generated, 2, n_embd)
```

```bash
./build/generate_example -m model.gguf -p "The capital of Italy is" -o gen.npy -n 32
```

> `generated()` is encoded as offset 0 (the last row of the current decode),
> identical to `last()` at the tensor level. The distinct factory exists so
> prefill vs. generate intent is visible at the call site.

### 4. Raw ggml — no llama.cpp wrapper

`HiddenStateExtractor` is model-agnostic. Any loop visiting every
`ggml_tensor*` can drive it with the two-phase `ask` protocol:

```cpp
HiddenStateExtractor extractor({
    {"inp_scaled", TokenSelector::last()},
    {"l_out-7",    TokenSelector::from_end(1)},
});

// Ask phase (graph construction): keep requested tensors.
for (ggml_tensor* t : graph_tensors)
    if (extractor.filter(t, /*ask=*/true)) { /* retain t */ }

evaluate_graph();

// Copy phase:
extractor.begin_frame();  // (or before evaluation — flags are per-frame)
for (ggml_tensor* t : graph_tensors)
    extractor.filter(t, /*ask=*/false);

extractor.require_frame_complete();
extractor.commit_frame();
extractor.save_npy("raw.npy");
```

Equivalently, pass `HiddenStateExtractor::callback` directly as
`llama_context_params::cb_eval` (this is what `attach()` does) with
`cb_eval_user_data` pointing at your extractor.

---

## API reference

### `TokenSelector` — `include/ggml_extractor/token_selector.hpp`

| Factory | Meaning |
|---|---|
| `TokenSelector::last()` | last row (`offset 0`) — decode |
| `TokenSelector::second_last()` | `offset 1` |
| `TokenSelector::third_last()` | `offset 2` |
| `TokenSelector::from_end(k)` | `k`-th from end (`0` = last) |
| `TokenSelector::generated()` | current generated token — inference loop |

Accessors: `mode()`, `offset_from_end()`, `is_generated()`, `to_string()`.

### `ExtractionRequest` — `include/ggml_extractor/extraction_request.hpp`

```cpp
struct ExtractionRequest {
    std::string tensor_name;             // exact ggml name, e.g. "l_out-20"
    TokenSelector token = TokenSelector::last();
};
```

One tensor may appear multiple times with different selectors.

### `HiddenStateExtractor` — `include/ggml_extractor/extractor.hpp`

Construct with `std::vector<ExtractionRequest>` (non-empty; exact-duplicate
tensor+token pairs rejected).

| Method | Description |
|---|---|
| `attach(params)` | set `cb_eval` + `user_data` on `llama_context_params` |
| `callback(t, ask, ud)` | static trampoline for `cb_eval` (`noexcept`) |
| `filter(t, ask)` | instance filter for raw ggml loops (`noexcept`) |
| `begin_frame()` | reset flags + error before each decode |
| `frame_complete()` | all requests copied, no error |
| `require_frame_complete()` | throw with missing-tensor details |
| `error()` | last callback error |
| `request_count()` / `request(i)` / `requests()` | introspection |
| `embedding_width(i)` / `common_embedding_width()` | observed widths |
| `frame_row(i)` | current frame row for request `i` |
| `frame_flat()` | concatenated current frame |
| `commit_frame()` | validate (complete + finite + uniform) and append |
| `sequence_frame_count()` / `has_sequence()` / `clear_sequence()` | buffer mgmt |
| `save_npy(path)` | `(frames, requests, embd)` — needs ≥1 commit |
| `save_current_frame_npy(path)` | `(requests, embd)` — current frame only |

### `LlamaSession` — `include/ggml_extractor/llama_session.hpp`

Owns `llama_model` + `llama_context`; borrows the extractor (must outlive it).

| Method | Description |
|---|---|
| `LlamaSession(path, extractor, opts)` | load GGUF; optional arch guard |
| `tokenize(prompt)` | BOS-aware, special-parsing tokenization |
| `decode(tokens)` | prefill: size/create ctx, clear KV, decode, `n_past = N` |
| `decode_one(token)` | generate step: reuse ctx, `pos = n_past`, decode, `n_past++` |
| `n_layer()` / `n_embd()` / `n_ctx_train()` | model introspection |
| `context_handle()` / `model_handle()` | raw handles for sampling/logits |

`LlamaSessionOptions`: `n_gpu_layers` (default `-1`), `n_ctx` (`0` = auto;
set explicitly for generation so the context is not re-created),
`n_batch` (`0` = `n_ctx`), `required_architecture` (empty = any).

### `NpyWriter` — `include/ggml_extractor/npy_writer.hpp`

- `NpyWriter(path, shape)` + `write_values(ptr, n)` + `finish()` (streaming).
- `save_npy(path, values, shape)`, `save_npy_2d`, `save_npy_3d` (one-shot).
- Refuses to overwrite; atomic publish via temp sibling + hard link.

---

## Specification

### Tensor naming

- `tensor_name` is compared with `std::string_view(tensor->name)` using exact,
  case-sensitive equality.
- Typical instrumented llama.cpp names: `inp_scaled` (scaled input embeddings),
  `l_out-<layer>` (block outputs, `0`-based). Any exact name in the evaluated
  graph is valid.
- Non-matching tensors return `false` in the `ask` phase and are ignored.

### Token indexing

- ggml hidden tensors are `(n_embd, n_tokens, 1, 1)`; the token axis is `ne[1]`.
- Selected row: `source_row = ne[1] - 1 - offset_from_end`.
- `offset >= ne[1]` → callback error → `require_frame_complete()` throws
  (e.g. `second_last()` on a 1-token decode).
- `generated()` ≡ offset 0 of the **current** decode batch. In a generation
  loop each decode carries exactly the new token, so this is the current
  generated token's row.

### Shapes

| Output | Shape | Axes |
|---|---|---|
| `save_npy` (sequence) | `(F, R, E)` | frames, requests (ctor order), embd |
| `save_current_frame_npy` | `(R, E)` | requests, embd |
| Prefill single prompt | `(1, R, E)` | |
| Batch of `B` prompts | `(B, R, E)` | input order |
| `G` generated tokens | `(G, R, E)` | generation order |

- `E` must be uniform across requests and frames; otherwise `save_npy` throws.
- Dtype is always float32 (`<f4`), C-order.

### `.npy` format

- Version 1.0 (`\x93NUMPY`, `1.0`, uint16 LE header length).
- Header `{'descr': '<f4', 'fortran_order': False, 'shape': (...), }` padded
  with spaces + `\n` to 64-byte alignment.
- Little-endian hosts write directly; big-endian hosts byte-swap per float.
- Requires 8-bit bytes, 32-bit IEEE-754 float (static asserts).
- Write-then-link: bytes go to `<output>.tmp-<rand>`; `create_hard_link` to
  the final path fails if the output already exists — existing files are never
  truncated.

### Lifecycle contract

```
begin_frame() → llama_decode() → llama_synchronize()
  → require_frame_complete() → commit_frame() → [repeat] → save_npy()
```

- Call `begin_frame()` before **every** decode (both `decode` and
  `decode_one` do this internally — call it manually only for raw ggml loops).
- `commit_frame()` validates completeness, finiteness (`isfinite`), and width
  uniformity, then copies.
- `save_npy()` requires ≥1 committed frame.

### Validation performed per tensor

1. `type == GGML_TYPE_F32`, else error.
2. `ne[2] == 1 && ne[3] == 1`, `ne[0] > 0`, `ne[1] > 0`.
3. Row-major contiguity: `nb[0] == 4`, `nb[1] == ne[0] * 4`.
4. Overflow-checked `source_offset = source_row * nb[1]`;
   `source_offset + row_bytes ≤ ggml_nbytes(tensor)`.
5. First-seen width per request is latched; later widths must match.
6. Copy via `ggml_backend_tensor_get`.

### Context & positions (generation)

- `decode()` clears KV memory and starts positions at 0; afterwards
  `n_past = N`.
- `decode_one()` reuses the context, sets `batch.pos[0] = n_past`, decodes,
  then `n_past++`. KV cache is preserved.
- If no context exists, `decode_one()` creates one from `options.n_ctx`
  (or training length) — but prefer an explicit prefill or `n_ctx` so the
  context is large enough for the full run. Exceeding it throws.

### Threading & determinism

- One extractor per inference stream; concurrent decodes sharing an extractor
  are not supported.
- The `cb_eval` path is `noexcept`; errors surface on the decode thread via
  `require_frame_complete()`.
- No RNG inside the library (only temp-file suffixes). Seeded-model behavior
  may still vary by backend (`cpu` is the conservative choice).

---

## Examples

| File | Shows | Output shape |
|---|---|---|
| `examples/01_prefill.cpp` | single prompt, `inp_scaled` + layers | `(R, E)` via `save_current_frame_npy`-equivalent sequence of 1 |
| `examples/02_batch.cpp` | many prompts, same tensor twice (last + 2nd-last) | `(B, R, E)` |
| `examples/03_generate.cpp` | greedy generation, `generated()` rows | `(G, R, E)` |
| `examples/04_raw_ggml.cpp` | raw graph loop without `LlamaSession` | `(F, R, E)` |

See [Usage](#usage) for the corresponding snippets.

---

## Project layout

```
ggml-extractor/
├── assets/
│   ├── logo.svg              # vector logo
│   └── logo.png              # raster logo (1024 px)
├── examples/
│   ├── 01_prefill.cpp
│   ├── 02_batch.cpp
│   ├── 03_generate.cpp
│   └── 04_raw_ggml.cpp
├── include/ggml_extractor/
│   ├── extractor.hpp         # HiddenStateExtractor
│   ├── extraction_request.hpp
│   ├── token_selector.hpp
│   ├── llama_session.hpp     # LlamaSession + options
│   ├── npy_writer.hpp
│   └── version.hpp           # 0.1.0
├── src/
│   ├── extractor.cpp
│   ├── llama_session.cpp
│   └── npy_writer.cpp
├── CMakeLists.txt
├── LICENSE                   # MIT
└── README.md
```

---

## Provenance

Generalizes the extraction logic previously embedded in
`concept-embeddings`' `cpp/hidden_states/src/main.cpp` (full
`inp_scaled` + all-layer prefill capture) and `query_main.cpp` (selected-layer
query capture with `l_out-{20,25,26,29,35}`). Behavior preserved:

- two-phase `cb_eval` (`ask` → copy),
- last-row (`ne[1]-1-offset`) float32 contiguous copies via
  `ggml_backend_tensor_get`,
- atomic non-overwriting `.npy` output.

What changed: hard-coded layer lists → user-supplied `ExtractionRequest`
vectors; `main.cpp`-specific CSV/model runners → reusable
`HiddenStateExtractor` + `LlamaSession`; generation support with position
tracking.

---

## License

MIT — see [LICENSE](LICENSE).

```
Copyright (c) 2026 Giulio Enzo Donninelli and Adversal.ai
```

---

## Authors

- **Giulio Enzo Donninelli** — design, implementation.
- **[Adversal.ai](https://adversal.ai)** — supporting company.

Logo: `assets/logo.svg` / `assets/logo.png` (this page, top).

If you use this library, a citation or shout-out is appreciated but not
required by the license.
