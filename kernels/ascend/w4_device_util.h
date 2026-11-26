#pragma once

#include "kernel_operator.h"
#include "w4_integer_layout.h"

namespace plait_ascend {

// Assign complete cache lines to a single core even when the buffer begins inside a line.
struct WriteSlices {
    uint64_t length;
    uint64_t items;
    uint64_t prefix;

    __aicore__ WriteSlices(GM_ADDR buffer, uint64_t length, uint32_t item_bytes,
                           uint32_t alignment = PLAIT_ASCEND_WORKSPACE_ALIGNMENT)
        : length(length), items(alignment / item_bytes),
          prefix(((0u - reinterpret_cast<uint64_t>(buffer)) & (alignment - 1)) / item_bytes) {}

    __aicore__ uint64_t start(uint64_t slice) const {
        return slice ? prefix + (slice - 1) * items : 0;
    }

    __aicore__ uint64_t end(uint64_t slice) const {
        const auto limit = prefix + slice * items;
        return limit < length ? limit : length;
    }
};

__aicore__ inline void flush(GM_ADDR address) {
    AscendC::GlobalTensor<uint64_t> written;
    written.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(address));
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                     AscendC::DcciDst::CACHELINE_OUT>(written);
}

__aicore__ inline uint32_t group_count(uint32_t k, uint32_t group_size) {
    return 1 + (k - 1) / group_size;
}

__aicore__ inline void pack_projection_activation(GM_ADDR activation, GM_ADDR packed,
    uint32_t rows, uint32_t k, uint32_t padded_rows, uint32_t depth, uint32_t part_step,
    uint32_t first, uint32_t parts) {
    const auto source = reinterpret_cast<__gm__ const uint64_t *>(activation);
    const auto out = reinterpret_cast<__gm__ uint64_t *>(packed);
    const uint32_t words = depth / 4;
    const uint64_t part_stride = uint64_t(padded_rows) * words;
    const WriteSlices slices(packed, part_stride * parts, sizeof(uint64_t));
    for (uint64_t slice = AscendC::GetBlockIdx(); slices.start(slice) < slices.length;
         slice += AscendC::GetBlockNum()) {
        const uint64_t end = slices.end(slice);
        for (uint64_t pos = slices.start(slice); pos < end; ++pos) {
            const uint32_t part = first + pos / part_stride;
            const uint32_t row = (pos % part_stride) / words;
            const uint32_t inner = part * (part_step / 4) + pos % words;
            out[pos] = row < rows && inner < k / 4 ? source[uint64_t(row) * (k / 4) + inner] : 0;
        }
    }
    flush(packed);
}

}  // namespace plait_ascend
