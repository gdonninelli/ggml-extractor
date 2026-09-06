#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml_extractor {

/// @file npy_writer.hpp
/// @brief Minimal NumPy `.npy` (v1.0, `<f4`, C-order) writer.
///
/// Only the subset needed by ggml-extractor is implemented:
/// float32 arrays, C-contiguous, version 1.0 header with 64-byte alignment.
/// Files are written atomically: bytes go to a unique sibling temporary file
/// which is then hard-linked to the final path. Existing outputs are never
/// overwritten.

/// @brief Streaming writer: open once, append row blocks, then finish().
class NpyWriter {
public:
    /// @brief Open a temporary sibling of `output` and write the header.
    /// @param output Final destination path (must not exist yet).
    /// @param shape Array shape, e.g. `{rows, layers, embd}`. Non-empty.
    /// @throws std::runtime_error on invalid shape, unsupported host order,
    ///         or I/O failure.
    NpyWriter(const std::string& output, const std::vector<std::size_t>& shape);

    ~NpyWriter();

    NpyWriter(const NpyWriter&) = delete;
    NpyWriter& operator=(const NpyWriter&) = delete;
    NpyWriter(NpyWriter&&) = delete;
    NpyWriter& operator=(NpyWriter&&) = delete;

    /// @brief Append `count` float32 values.
    /// @throws std::runtime_error if `count` overflows the declared shape.
    void write_values(const float* values, std::size_t count);

    /// @brief Validate the value count, flush, and atomically publish.
    /// @throws std::runtime_error on short/long arrays or publish failure.
    void finish();

    /// @brief True after finish() published successfully.
    bool finished() const { return finished_; }

private:
    struct Impl;
    Impl* impl_ = nullptr;
    bool finished_ = false;
};

/// @brief One-shot convenience wrapper around NpyWriter.
/// @param output Destination path (must not already exist).
/// @param values Row-major data, size must equal product(shape).
/// @param shape Array shape.
/// @param rows_hint Unused, kept for API symmetry (pass shape explicitly).
void save_npy(const std::string& output,
              const std::vector<float>& values,
              const std::vector<std::size_t>& shape);

/// @brief Overload taking a 2-D (rows, cols) shape.
void save_npy_2d(const std::string& output,
                 const std::vector<float>& values,
                 std::size_t rows,
                 std::size_t cols);

/// @brief Overload taking a 3-D (dim0, dim1, dim2) shape.
void save_npy_3d(const std::string& output,
                 const std::vector<float>& values,
                 std::size_t dim0,
                 std::size_t dim1,
                 std::size_t dim2);

}  // namespace ggml_extractor
