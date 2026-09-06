#include "ggml_extractor/npy_writer.hpp"

#include <climits>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace ggml_extractor {
namespace {

namespace fs = std::filesystem;

class TemporaryFileCleanup {
public:
    explicit TemporaryFileCleanup(fs::path path) : path_(std::move(path)) {}

    ~TemporaryFileCleanup() {
        if (active_) {
            std::error_code ignored;
            fs::remove(path_, ignored);
        }
    }

    void release() noexcept { active_ = false; }

private:
    fs::path path_;
    bool active_ = true;
};

fs::path make_temporary_sibling(const fs::path& output) {
    std::random_device random;
    const fs::path parent =
        output.parent_path().empty() ? fs::path(".") : output.parent_path();
    const std::string base = output.filename().string();
    for (int attempt = 0; attempt < 100; ++attempt) {
        const std::uint64_t suffix =
            (static_cast<std::uint64_t>(random()) << 32U) ^ static_cast<std::uint64_t>(random());
        const fs::path candidate = parent / (base + ".tmp-" + std::to_string(suffix));
        if (!fs::exists(candidate)) {
            return candidate;
        }
    }
    throw std::runtime_error("could not create a unique temporary output path");
}

bool host_is_little_endian(bool& little, bool& big) {
    const std::uint32_t probe = 0x01020304U;
    unsigned char order[sizeof(probe)] = {};
    std::memcpy(order, &probe, sizeof(probe));
    little = (order[0] == 0x04U && order[1] == 0x03U && order[2] == 0x02U &&
              order[3] == 0x01U);
    big = (order[0] == 0x01U && order[1] == 0x02U && order[2] == 0x03U &&
           order[3] == 0x04U);
    return little || big;
}

std::string shape_string(const std::vector<std::size_t>& shape) {
    std::string text = "(";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            text += ", ";
        }
        text += std::to_string(shape[i]);
    }
    if (shape.size() == 1) {
        text += ",";
    }
    text += ")";
    return text;
}

}  // namespace

struct NpyWriter::Impl {
    fs::path output;
    fs::path temporary;
    TemporaryFileCleanup cleanup;
    std::ofstream stream;
    std::size_t expected = 0;
    std::size_t written = 0;
    bool little_endian = false;

    Impl(const std::string& out, const std::vector<std::size_t>& shape)
        : output(out), temporary(make_temporary_sibling(output)), cleanup(temporary) {
        static_assert(CHAR_BIT == 8, "NumPy serialization requires 8-bit bytes");
        static_assert(sizeof(float) == 4, "NumPy <f4 needs 32-bit float");
        static_assert(std::numeric_limits<float>::is_iec559,
                      "NumPy <f4 needs IEEE-754 float");

        if (shape.empty()) {
            throw std::runtime_error("NumPy array must have at least one dimension");
        }
        expected = 1;
        for (std::size_t dim : shape) {
            if (dim != 0 && expected > std::numeric_limits<std::size_t>::max() / dim) {
                throw std::runtime_error("NumPy array dimensions overflow");
            }
            expected *= dim;
        }

        bool little = false, big = false;
        if (!host_is_little_endian(little, big)) {
            throw std::runtime_error("unsupported host byte order");
        }
        little_endian = little;

        std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " +
                             shape_string(shape) + ", }";
        constexpr std::size_t preamble = 10;
        const std::size_t padding = (64 - ((preamble + header.size() + 1) % 64)) % 64;
        header.append(padding, ' ');
        header.push_back('\n');
        if (header.size() > std::numeric_limits<std::uint16_t>::max()) {
            throw std::runtime_error("NumPy header exceeds version 1.0 limits");
        }

        stream.open(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("could not open temporary output: " +
                                     temporary.string());
        }
        const char magic[] = {static_cast<char>(0x93), 'N', 'U', 'M', 'P', 'Y'};
        const char version[] = {1, 0};
        const auto header_len = static_cast<std::uint16_t>(header.size());
        const char len_bytes[] = {
            static_cast<char>(header_len & 0xffU),
            static_cast<char>((header_len >> 8U) & 0xffU),
        };
        stream.write(magic, sizeof(magic));
        stream.write(version, sizeof(version));
        stream.write(len_bytes, sizeof(len_bytes));
        stream.write(header.data(), static_cast<std::streamsize>(header.size()));
        if (!stream) {
            throw std::runtime_error("failed to write temporary output: " +
                                     temporary.string());
        }
    }
};

NpyWriter::NpyWriter(const std::string& output, const std::vector<std::size_t>& shape)
    : impl_(new Impl(output, shape)) {}

NpyWriter::~NpyWriter() {
    delete impl_;
}

void NpyWriter::write_values(const float* values, std::size_t count) {
    if (count > impl_->expected - impl_->written) {
        throw std::runtime_error("too many values for NumPy output");
    }
    if (impl_->little_endian) {
        impl_->stream.write(reinterpret_cast<const char*>(values),
                            static_cast<std::streamsize>(count * sizeof(float)));
    } else {
        for (std::size_t i = 0; i < count; ++i) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, values + i, sizeof(bits));
            bits = ((bits & 0x000000ffU) << 24U) | ((bits & 0x0000ff00U) << 8U) |
                   ((bits & 0x00ff0000U) >> 8U) | ((bits & 0xff000000U) >> 24U);
            impl_->stream.write(reinterpret_cast<const char*>(&bits), sizeof(bits));
        }
    }
    if (!impl_->stream) {
        throw std::runtime_error("failed to write temporary output: " +
                                 impl_->temporary.string());
    }
    impl_->written += count;
}

void NpyWriter::finish() {
    if (finished_) {
        return;
    }
    if (impl_->written != impl_->expected) {
        throw std::runtime_error("NumPy output contains an incomplete array");
    }
    impl_->stream.flush();
    impl_->stream.close();
    if (!impl_->stream) {
        throw std::runtime_error("failed to close temporary output: " +
                                 impl_->temporary.string());
    }
    std::error_code publish_error;
    fs::create_hard_link(impl_->temporary, impl_->output, publish_error);
    if (publish_error) {
        if (publish_error == std::errc::file_exists || fs::exists(impl_->output)) {
            throw std::runtime_error("output already exists: " + impl_->output.string());
        }
        throw std::runtime_error("failed to publish output: " + publish_error.message());
    }
    std::error_code remove_error;
    fs::remove(impl_->temporary, remove_error);
    if (remove_error) {
        throw std::runtime_error("failed to remove temporary output link: " +
                                 remove_error.message());
    }
    impl_->cleanup.release();
    finished_ = true;
}

void save_npy(const std::string& output,
              const std::vector<float>& values,
              const std::vector<std::size_t>& shape) {
    if (shape.empty()) {
        throw std::runtime_error("NumPy array must have at least one dimension");
    }
    std::size_t expected = 1;
    for (std::size_t dim : shape) {
        if (dim != 0 && expected > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::runtime_error("NumPy array dimensions overflow");
        }
        expected *= dim;
    }
    if (values.size() != expected) {
        throw std::runtime_error("NumPy data size does not match shape");
    }
    NpyWriter writer(output, shape);
    if (!values.empty()) {
        writer.write_values(values.data(), values.size());
    }
    writer.finish();
}

void save_npy_2d(const std::string& output,
                 const std::vector<float>& values,
                 std::size_t rows,
                 std::size_t cols) {
    save_npy(output, values, {rows, cols});
}

void save_npy_3d(const std::string& output,
                 const std::vector<float>& values,
                 std::size_t dim0,
                 std::size_t dim1,
                 std::size_t dim2) {
    save_npy(output, values, {dim0, dim1, dim2});
}

}  // namespace ggml_extractor
