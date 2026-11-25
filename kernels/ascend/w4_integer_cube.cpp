#include "kernel_operator.h"
#include "w4_integer_layout.h"

namespace {

constexpr uint32_t tile_m = PLAIT_ASCEND_TILE_M;
constexpr uint32_t tile_n = PLAIT_ASCEND_TILE_N;
constexpr uint32_t cube_k = PLAIT_ASCEND_CUBE_K;
constexpr uint32_t max_group_k = PLAIT_ASCEND_MAX_GROUP_K;

}  // namespace

extern "C" __global__ __aicore__ void plait_ascend_w4_cube(
    GM_ADDR packed_a, GM_ADDR packed_b, GM_ADDR scratch,
    uint32_t m, uint32_t n, uint32_t k, uint32_t groups) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    AscendC::LocalMemAllocator<AscendC::Hardware::L1> l1;
    AscendC::LocalMemAllocator<AscendC::Hardware::L0A> l0a;
    AscendC::LocalMemAllocator<AscendC::Hardware::L0B> l0b;
    AscendC::LocalMemAllocator<AscendC::Hardware::L0C> l0c;
    const auto a1 = l1.Alloc<AscendC::TPosition::A1, int8_t>(tile_m * max_group_k);
    const auto b1 = l1.Alloc<AscendC::TPosition::B1, int8_t>(tile_n * max_group_k);
    const auto a2 = l0a.Alloc<AscendC::TPosition::A2, int8_t>(tile_m * max_group_k);
    const auto b2 = l0b.Alloc<AscendC::TPosition::B2, int8_t>(tile_n * max_group_k);
    const auto c = l0c.Alloc<AscendC::TPosition::CO1, int32_t>(tile_m * tile_n);

    const uint32_t n_tiles = n / tile_n;
    const uint32_t tiles_per_group = (m / tile_m) * n_tiles;
    const uint32_t tiles = groups * tiles_per_group;
    const AscendC::Nd2NzParams a_load(1, tile_m, k, 0, k, tile_m, 1, 0);
    const AscendC::Nd2NzParams b_load(1, tile_n, k, 0, k, tile_n, 1, 0);
    const AscendC::LoadData2DParamsV2 a_move(0, 0, tile_m / 16, k / cube_k, tile_m / 16, tile_m / 16, false, 0);
    const AscendC::LoadData2DParamsV2 b_move(0, 0, tile_n / 16, k / cube_k, tile_n / 16, tile_n / 16, false, 0);
    const AscendC::MmadParams dot(tile_m, tile_n, k, 0, false, true);
    const AscendC::FixpipeParamsV220 store(tile_n, tile_m, tile_m, tile_n, false);

    const auto a = reinterpret_cast<__gm__ int8_t *>(packed_a);
    const auto b = reinterpret_cast<__gm__ int8_t *>(packed_b);
    const auto out = reinterpret_cast<__gm__ int32_t *>(scratch);
    for (uint32_t task = AscendC::GetBlockIdx(); task < tiles; task += AscendC::GetBlockNum()) {
        const uint32_t group = task / tiles_per_group;
        const uint32_t tile = task % tiles_per_group;
        const uint32_t row = tile / n_tiles * tile_m;
        const uint32_t col = tile % n_tiles * tile_n;
        AscendC::GlobalTensor<int8_t> ar, br;
        AscendC::GlobalTensor<int32_t> cr;
        ar.SetGlobalBuffer(a + (uint64_t(group) * m + row) * k, tile_m * k);
        br.SetGlobalBuffer(b + (uint64_t(group) * n + col) * k, tile_n * k);
        cr.SetGlobalBuffer(out + uint64_t(task) * tile_m * tile_n, tile_m * tile_n);

        AscendC::DataCopy(a1, ar, a_load);
        AscendC::DataCopy(b1, br, b_load);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::LoadData(a2, a1, a_move);
        AscendC::LoadData(b2, b1, b_move);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::Mmad(c, a2, b2, dot);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::Fixpipe(cr, c, store);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}
