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

}  // namespace plait_ascend
