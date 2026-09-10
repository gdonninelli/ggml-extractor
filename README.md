# ggml-extractor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![llama.cpp](https://img.shields.io/badge/engine-llama.cpp-green.svg)](https://github.com/ggml-org/llama.cpp)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/ggml-org/llama.cpp)
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)](include/ggml_extractor/version.hpp)

<p align="center">
  <img src="assets/logo.png" alt="ggml-extractor logo — stacked LLM layers with the selected token extracted into a .npy grid" width="580"/>
</p>

> **Created by Giulio Enzo Donninelli and [Adversal.ai](https://adversal.ai).**

A small C++17 library that reads **hidden states for specific tokens at
specific layers** out of a live `llama.cpp` inference run — without giving up
ordinary inference on the same model.

Load the model once. Serve normal completions from it. When you need
activations instead, arm a capture for a decode or two and take the result in
memory. Text prompts and image prompts go through the same path.

```cpp
auto model = Model::load("gemma4.gguf");          // once per process
SessionOptions options;
options.n_ctx = 8192;
Session session(model, options);                  // context created once

// ordinary inference — nothing armed, full speed
session.decode(0, model->tokenize("hello"));
int32_t next = session.sample_greedy();

// activations — same model, same context, same KV cache
HiddenStateCapture capture({
    {"inp_scaled", TokenSelector::last()},
    {"l_out-20",   TokenSelector::last()},
    {"l_out-29",   TokenSelector::second_last()},
});
{
    auto armed = session.arm(capture);
    session.decode(0, model->tokenize("The cat sat on the mat."));
    capture.commit_frame();
}
HiddenStates states = capture.take();             // (1, 3, n_embd), in memory
```

---

## Table of contents

- [Why](#why)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Build](#build)
- [Usage](#usage)
  - [1. One prompt](#1-one-prompt)
  - [2. Many prompts, one session](#2-many-prompts-one-session)
  - [3. Generation — every generated token](#3-generation--every-generated-token)
  - [4. Images](#4-images)
  - [5. Raw ggml — no llama_context](#5-raw-ggml--no-llama_context)
- [API reference](#api-reference)
- [Specification](#specification)
- [Examples](#examples)
- [Project layout](#project-layout)
- [Upgrading from 0.1.x](#upgrading-from-01x)
- [License](#license)
- [Authors](#authors)

---

## Why

`llama.cpp` exposes per-tensor callbacks (`cb_eval`), and ggml tensors carry
canonical names (`inp_scaled`, `l_out-20`, …). Copying the right row of the
right tensor at the right decode step is only a few dozen lines — but every
project re-implements it slightly differently, with slightly different bugs
(non-contiguous rows, wrong token index, silent `.npy` overwrites).

Version 0.1 packaged that logic as a library, but assumed a one-shot CLI: the
model, the context and the output buffer were all owned by the same object,
the KV cache was cleared on every prefill, and results only went to disk.
Version 0.2 splits those lifetimes apart so the same model can serve inference
and extraction in one long-running process:

| Layer | Lifetime | Holds |
|---|---|---|
| `Model` | process | the weights — shared by every session |
| `Session` | worker thread | one `llama_context` + KV cache + the installed callback |
| `HiddenStateCapture` | one request | which tensors to grab, and the results |
| `Multimodal` | process | the vision/audio projector, bound to a `Model` |

---

## How it works

`llama.cpp` copies `cb_eval` into the context when the context is created and
offers no setter, so extraction cannot be wired in after the fact. `Session`
therefore installs **its own trampoline once**, up front, and switches
behaviour behind it: `arm()` points the trampoline at a capture, and the
guard's destructor points it back at nothing.

Leaving the callback installed is close to free. In the ggml scheduler, when
the `ask` phase answers "no" for every node, the scan runs to the end of the
graph split and computes it in a *single* submission — so a disarmed session
costs one trivial callback per graph node plus one extra backend
synchronisation per split. Graph reuse and operator fusion are unaffected.

When a capture **is** armed the scheduler chops the split into chunks and
synchronises after each one. That cost is inherent to `cb_eval` and applies
only while you are extracting. Measured on the CPU backend (Gemma 3 1B Q8_0,
64 decode steps per configuration, same process):

| | per-token decode |
|---|---|
| disarmed | 27.8 ms |
| armed, 1 layer | 26.8 ms |
| armed, all 26 layers | 27.0 ms |

i.e. within measurement noise, and flat in the number of layers captured —
on CPU the per-chunk synchronise is nearly free. Expect a real cost on CUDA,
where chunking defeats CUDA graphs and forces genuine device syncs.

Arming is also numerically inert: an armed prefill and an unarmed prefill of
the same prompt produce **bit-identical logits** across all 262,144 vocab
entries, and disarming returns to the same path.

Two contexts (a fast one plus an instrumented one) would avoid even that, but
they cannot share a KV cache: `llama_context_params::ctx_other` is gated to
draft/assistant architectures, so a second context means double the KV memory
and re-prefilling every prompt. One context is the better trade.

---

## Requirements

| Requirement | Notes |
|---|---|
| C++17 compiler | Clang 14+, GCC 11+, MSVC 19.30+ |
| CMake ≥ 3.20 | |
| llama.cpp checkout | Recent revision exposing `cb_eval`, `llama_memory_*`, `llama_n_ctx_seq`, and (for images) `tools/mtmd` with `LLAMA_BUILD_MTMD` |
| GGUF model | Any architecture (optional `required_architecture` guard) |
| `mmproj` GGUF | Only for image/audio input — a separate file, see below |

Verified against llama.cpp `df03399` (2026-09-10).

---

## Build

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ../llama.cpp

# text-only model (no vision projector) — smaller, faster build
cmake -S . -B build \
  -DLLAMA_CPP_DIR=../llama.cpp \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_EXTRACTOR_MULTIMODAL=OFF
cmake --build build -j

# vision model (you have an mmproj-*.gguf) — adds libmtmd
cmake -S . -B build -DLLAMA_CPP_DIR=../llama.cpp \
  -DCMAKE_BUILD_TYPE=Release -DGGML_EXTRACTOR_MULTIMODAL=ON
cmake --build build -j
```

| Option | Default | Effect |
|---|---|---|
| `LLAMA_CPP_DIR` | *(required)* | llama.cpp source checkout |
| `GGML_EXTRACTOR_MULTIMODAL` | `ON` | Build `libmtmd` and `Multimodal`. Turn it `OFF` for text-only models: `libmtmd` is a large library (every vision/audio architecture llama.cpp supports) and is useless without a projector file. |

All five examples build in either configuration —
`service_example`'s image stages are `#ifdef`-guarded on the
`GGML_EXTRACTOR_MULTIMODAL` define the library exports.

```bash
cmake --install build --prefix /usr/local   # optional
```

### Does my model need `GGML_EXTRACTOR_MULTIMODAL`?

Only if you have a **separate** `mmproj-*.gguf`. llama.cpp never packs the
vision tower into the text GGUF: `convert_hf_to_gguf.py --mmproj` writes it to
its own file, and `mtmd_init_from_file()` takes that filename as a required
argument. So:

- text-only model, or a vision model whose projector you did not convert or
  download → `-DGGML_EXTRACTOR_MULTIMODAL=OFF`, and images are unavailable
- text GGUF **and** its `mmproj-*.gguf` → `ON`, and `Multimodal` works

To check what you have, look for a projector type in the file's metadata:

```bash
strings model.gguf | grep -m1 clip.projector_type   # nothing = no vision tower
```

---

## Usage

### 1. One prompt

`from_end(0)` (= `last()`) is the final prompt token, `from_end(1)` the one
before it, and so on.

```cpp
auto model = Model::load("model.gguf");
Session session(model);

HiddenStateCapture capture({
    {"inp_scaled", TokenSelector::last()},
    {"l_out-20",   TokenSelector::last()},
    {"l_out-25",   TokenSelector::from_end(1)},
});
{
    auto armed = session.arm(capture);
    session.decode(0, model->tokenize("The cat sat on the mat."), false);
    capture.commit_frame();
}
HiddenStates states = capture.take();   // (1, 3, n_embd)
states.save_npy("prefill.npy");         // optional
```

```bash
./build/prefill_example -m model.gguf -p "hello world" -o out.npy -l 20,25,29
```

### 2. Many prompts, one session

The model and context are created once; `reset(seq)` frees the sequence's KV
cache between prompts.

```cpp
HiddenStateCapture capture({
    {"l_out-20", TokenSelector::last()},
    {"l_out-20", TokenSelector::second_last()},   // same tensor, two rows
    {"l_out-29", TokenSelector::last()},
});

auto armed = session.arm(capture);
for (const auto& prompt : prompts) {
    session.reset(0);
    session.decode(0, model->tokenize(prompt), false);
    capture.commit_frame();                       // one frame per prompt
}
HiddenStates states = capture.take();             // (n_prompts, 3, n_embd)
```

### 3. Generation — every generated token

```cpp
HiddenStateCapture capture({
    {"l_out-20", TokenSelector::generated()},
    {"l_out-29", TokenSelector::generated()},
});
auto armed = session.arm(capture);

session.decode(0, model->tokenize(prompt));
capture.begin_frame();                  // drop the prefill frame, keep the KV cache

for (int step = 0; step < n_predict; ++step) {
    int32_t next = session.sample_greedy();
    if (model->is_eog(next)) break;
    session.decode_one(0, next);
    capture.commit_frame();             // one frame per generated token
}
HiddenStates states = capture.take();   // (n_generated, 2, n_embd)
```

`generated()` is offset 0 of the current decode — identical to `last()` at the
tensor level. The separate factory exists so prefill and generation intent are
visible at the call site.

### 4. Images

Requires a separate `mmproj-*.gguf` and a build with
`-DGGML_EXTRACTOR_MULTIMODAL=ON` — see
[Does my model need it?](#does-my-model-need-ggml_extractor_multimodal).

`Multimodal` owns only the projector; the text weights stay in the `Model`.
Because `eval()` runs `llama_decode` on the session's context, an armed
capture sees an image prompt exactly as it sees a text prefill.

```cpp
Multimodal vision(model, "mmproj-gemma4.gguf");

std::string prompt = vision.marker() + "\nWhat is in this picture?";

// inference
vision.eval(session, 0, prompt, {"photo.jpg"});
int32_t next = session.sample_greedy();

// activations for the same prompt
session.reset(0);
{
    auto armed = session.arm(capture);
    vision.eval(session, 0, prompt, {"photo.jpg"}, /*logits_last=*/false);
    capture.commit_frame();             // the last prompt token, image attended
}
```

`examples/05_service.cpp` runs all four combinations — text generation, text
extraction, image generation, image extraction — against one loaded model.

### 5. Raw ggml — no `llama_context`

`HiddenStateCapture` only needs the two-phase protocol, so any loop over an
evaluated graph can drive it:

```cpp
capture.begin_frame();
for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
    ggml_tensor* node = ggml_graph_node(graph, i);
    if (capture.filter(node, /*ask=*/true)) {
        capture.filter(node, /*ask=*/false);
    }
}
capture.commit_frame();
HiddenStates states = capture.take();
```

`examples/04_raw_ggml.cpp` is a runnable version on the CPU backend, no model
required.

---

## API reference

### `Model` — `include/ggml_extractor/model.hpp`

| Member | Description |
|---|---|
| `Model::load(path, options)` | load a GGUF, returns `shared_ptr`; initialises backends |
| `tokenize(text, add_special, parse_special)` | tokenize |
| `token_to_piece(token)` / `is_eog(token)` | detokenize one token / end-of-generation test |
| `n_layer()` / `n_embd()` / `n_ctx_train()` / `n_vocab()` | introspection |
| `architecture()` | `general.architecture`, e.g. `"gemma4"` |
| `handle()` / `vocab()` | raw llama.cpp handles |

`ModelOptions`: `n_gpu_layers` (`-1` = auto), `required_architecture`
(empty = any).

### `Session` — `include/ggml_extractor/session.hpp`

| Member | Description |
|---|---|
| `Session(model, options)` | create the context (once) and install the callback |
| `arm(capture)` | `[[nodiscard]]` RAII guard; extraction live while it exists |
| `armed()` | whether a capture is armed |
| `decode(seq, tokens, logits_last)` | decode a batch into `seq`, splitting across `n_batch` |
| `decode_one(seq, token)` | one generation step, with logits |
| `n_past(seq)` | next position for `seq` (read from the KV cache) |
| `reset(seq)` / `reset_all()` | free one sequence / all sequences |
| `sample_greedy()` | argmax over the last logits |
| `model()` / `handle()` | the shared model / raw context handle |
| `n_ctx()` / `n_ctx_seq()` / `n_batch()` | sizes as llama.cpp allocated them |

`SessionOptions`: `n_ctx` (default `4096`), `n_batch`, `n_ubatch`,
`n_seq_max`, `n_threads`, `n_threads_batch`.

### `HiddenStateCapture` — `include/ggml_extractor/capture.hpp`

| Member | Description |
|---|---|
| `HiddenStateCapture(requests)` | non-empty; duplicate (tensor, token) pairs rejected |
| `filter(tensor, ask)` | the callback body (`noexcept`); for raw ggml loops |
| `begin_frame()` | discard the in-progress frame |
| `frame_complete()` / `require_frame_complete()` | test / throw with details |
| `error()` | first error recorded by `filter` |
| `commit_frame()` | validate, append, and begin the next frame |
| `frame_count()` / `requests()` / `embedding_width()` | introspection |
| `take()` | move the committed frames out as `HiddenStates` |

`HiddenStates`: `data`, `n_frames`, `n_requests`, `n_embd`, plus
`row(frame, request)` and `save_npy(path)`.

### `Multimodal` — `include/ggml_extractor/multimodal.hpp`

| Member | Description |
|---|---|
| `Multimodal(model, mmproj_path, options)` | load the projector against a `Model` |
| `supports_vision()` / `supports_audio()` | projector capabilities |
| `marker()` | the media placeholder, default `"<__media__>"` |
| `eval(session, seq, prompt, media_paths, logits_last)` | encode media, decode prompt + media |

`MultimodalOptions`: `use_gpu`, `n_threads`, `print_timings`.

### `TokenSelector` — `include/ggml_extractor/token_selector.hpp`

| Factory | Meaning |
|---|---|
| `last()` | last row (offset 0) |
| `second_last()` / `third_last()` | offset 1 / 2 |
| `from_end(k)` | k-th from end (0 = last) |
| `generated()` | current generated token — inference loop |

### `NpyWriter` — `include/ggml_extractor/npy_writer.hpp`

Streaming `NpyWriter(path, shape)` + `write_values()` + `finish()`, and
one-shot `save_npy`, `save_npy_2d`, `save_npy_3d`. Refuses to overwrite;
publishes atomically.

---

## Specification

### Tensor naming

- `tensor_name` is compared to `tensor->name` with exact, case-sensitive
  equality. Requests are grouped by name at construction, so the `ask` phase
  is one lookup per graph node regardless of how many layers you request.
- llama.cpp names hidden tensors `inp_scaled` (scaled input embeddings) and
  `l_out-<layer>` (block outputs, 0-based), unconditionally, in
  `llama_context::graph_get_cb`. Any exact name in the evaluated graph works.
- These are internal debug names with **no stability guarantee** across
  llama.cpp versions. A rename — or a tensor removed by operator fusion —
  makes `require_frame_complete()` throw rather than return wrong data. Run
  one prompt at startup to fail fast.

### Token indexing

- ggml hidden tensors are `(n_embd, n_tokens, 1, 1)`; the token axis is `ne[1]`.
- Selected row: `source_row = ne[1] - 1 - offset_from_end`.
- **Offsets past the last token need care on long prompts.** A prompt longer
  than `n_ubatch` is evaluated in several passes, and the capture keeps the
  most recent one — so `last()` is always the final prompt token, but
  `from_end(k)` for `k > 0` requires the *last* pass to hold more than `k`
  rows. When it does not, the capture reports it instead of guessing. Raise
  `SessionOptions::n_ubatch` past your prompt length if you need those rows.

### The final layer is special

llama.cpp gathers the last block's output down to only the positions you
requested logits for:

```c
// llama.cpp, e.g. src/models/gemma3.cpp
if (il == n_layer - 1 && inp_out_ids) {
    cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
    inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
}
```

So `l_out-<n_layer-1>` has `ne[1] == n_outputs`, not the batch length. Two
consequences:

- **Extracting the final layer requires `logits_last = true`.** With no
  outputs requested that tensor has zero rows, nothing is captured, and
  `commit_frame()` throws `missing hidden state`. A zero-row tensor is treated
  as "no rows in this pass", not as an error, so a multi-batch prompt still
  works — only the pass carrying the outputs contributes.
- **`from_end(k)` with `k > 0` is unavailable on the final layer**, since it
  holds only the output rows. Ask for it on any earlier layer instead.

Every other layer, and `inp_scaled`, carry the full batch.

### Frames

A frame is one row of the output. Several `llama_decode` calls may feed one
frame — a batched prompt, or an interleaved image + text prompt — with each
overwriting the last, so a frame committed after a whole prefill holds the
final prompt token. `commit_frame()` closes the current frame and opens the
next, which makes a generation loop one commit per sampled token.

### Shapes

| Output | Shape |
|---|---|
| `HiddenStates` from a single prefill | `(1, R, E)` |
| `B` prompts | `(B, R, E)` |
| `G` generated tokens | `(G, R, E)` |

Axes are (frames, requests in construction order, embedding). `E` must be
uniform across requests and frames. Dtype is always float32, C-order.

### `.npy` format

- Version 1.0 (`\x93NUMPY`, `1.0`, uint16 LE header length), `'<f4'`,
  `fortran_order: False`, header padded to 64-byte alignment.
- Little-endian hosts write directly; big-endian hosts byte-swap per float.
- Write-then-link: bytes go to `<output>.tmp-<rand>`, then `create_hard_link`
  to the final path — which fails if the output exists. Existing files are
  never truncated, so a service that writes files should vary the path.

### Validation performed per tensor

1. `type == GGML_TYPE_F32`.
2. `ne[2] == 1 && ne[3] == 1`, `ne[0] > 0`, `ne[1] > 0`.
3. Row-major contiguity: `nb[0] == 4`, `nb[1] == ne[0] * 4`.
4. Overflow-checked `source_offset`, bounded by `ggml_nbytes(tensor)`.
5. First-seen width per request is latched; later widths must match.
6. `commit_frame()` additionally rejects non-finite values.

### Threading

- `Model` is read-only during inference and may back any number of concurrent
  `Session`s.
- A `Session` and a `Capture` are single-threaded: one per worker. For a
  concurrent service, give each worker its own `Session` over the shared
  `Model`, or serialise behind a mutex.
- The `cb_eval` path is `noexcept`; errors surface on the decoding thread via
  `require_frame_complete()`.
- Backend initialisation happens once per process (`ensure_backend_initialized`,
  called by `Model::load`) and is never torn down — llama.cpp's backend
  registry is global state shared by every model and context.

---

## Examples

| File | Shows |
|---|---|
| `examples/01_prefill.cpp` | one prompt, layers chosen on the command line |
| `examples/02_batch.cpp` | many prompts through one session, `reset(seq)` between |
| `examples/03_generate.cpp` | greedy generation, one frame per generated token |
| `examples/04_raw_ggml.cpp` | the two-phase protocol on a raw CPU graph, no model |
| `examples/05_service.cpp` | **one process: text gen, text extraction, image gen, image extraction** |

```bash
./build/service_example -m gemma4.gguf --mmproj mmproj-gemma4.gguf \
    --image photo.jpg --image-prompt "What is in this picture?" \
    -p "The capital of Italy is" -n 48 -l 20,29 -o out
```

---

## Project layout

```
ggml-extractor/
├── assets/
├── examples/
│   ├── 01_prefill.cpp
│   ├── 02_batch.cpp
│   ├── 03_generate.cpp
│   ├── 04_raw_ggml.cpp
│   └── 05_service.cpp
├── include/ggml_extractor/
│   ├── backend.hpp           # process-wide backend init
│   ├── model.hpp             # Model (shared weights)
│   ├── session.hpp           # Session + SessionOptions + arm()
│   ├── capture.hpp           # HiddenStateCapture + HiddenStates
│   ├── multimodal.hpp        # Multimodal (libmtmd)
│   ├── extraction_request.hpp
│   ├── token_selector.hpp
│   ├── npy_writer.hpp
│   └── version.hpp
├── src/
├── CMakeLists.txt
├── LICENSE                   # MIT
└── README.md
```

---

## Upgrading from 0.1.x

`HiddenStateExtractor` and `LlamaSession` are gone; their responsibilities are
split four ways.

| 0.1.x | 0.2.0 |
|---|---|
| `LlamaSession(path, extractor, opts)` | `Model::load(path)` then `Session(model, opts)` |
| `HiddenStateExtractor(requests)` | `HiddenStateCapture(requests)`, one per request |
| `extractor.attach(params)` | automatic — `Session` installs its own trampoline |
| *(no way to turn it off)* | `session.arm(capture)` returns an RAII guard |
| `session.decode(tokens)` | `session.decode(seq, tokens, logits_last)` |
| `session.decode_one(token)` | `session.decode_one(seq, token)` |
| `extractor.require_frame_complete(); extractor.commit_frame();` | `capture.commit_frame()` (it validates first) |
| `extractor.save_npy(path)` | `capture.take()` → `HiddenStates`, then `save_npy(path)` if you want a file |
| `extractor.clear_sequence()` | `capture.take()` (or drop the capture) |

Behavioural changes worth knowing:

- **The KV cache is no longer cleared on every prefill.** `decode()` continues
  from the sequence's current position; call `reset(seq)` for an independent
  prompt.
- **The context is never recreated.** Size it with `SessionOptions::n_ctx` at
  construction; 0.1.x re-created it whenever a longer prompt arrived.
- **Results come back in memory.** `.npy` output is optional.
- **`llama_backend_init` is process-global**, not per session, so several
  sessions can coexist.

---

## Provenance

Generalizes the extraction logic previously embedded in
`concept-embeddings`' `cpp/hidden_states/src/main.cpp` and `query_main.cpp`.
Preserved from those tools and from 0.1.x: two-phase `cb_eval` (`ask` → copy),
last-row (`ne[1]-1-offset`) float32 contiguous copies via
`ggml_backend_tensor_get`, and atomic non-overwriting `.npy` output.

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
