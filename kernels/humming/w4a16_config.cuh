#pragma once

// Dense W4A16 configuration adapted from the pinned Humming JIT template.
// See NOTICE for the upstream revision and license.
#ifndef PLAIT_HUMMING_SM
#define PLAIT_HUMMING_SM 86
#endif
#ifndef PLAIT_HUMMING_N
#define PLAIT_HUMMING_N 256
#endif
#ifndef PLAIT_HUMMING_K
#define PLAIT_HUMMING_K 512
#endif

#define HUMMING_INPUT_SCALE_GROUP_SIZE 0
#define HUMMING_WEIGHT_SCALE_GROUP_SIZE 128
#define HUMMING_HAS_ZERO_POINT 1
#define HUMMING_HAS_BIAS 0
#define HUMMING_IS_CHANNEL_WEIGHT_SCALE 0
#define HUMMING_IS_GROUP_WEIGHT_SCALE 1
#define HUMMING_IS_CHANNEL_WEIGHT_SCALE_2 0
#define HUMMING_HAS_INPUT_SCALE 0
#define HUMMING_HAS_INPUT_SCALE_2 0
#define HUMMING_IS_GROUP_INPUT_SCALE 0
#define HUMMING_IS_TENSOR_INPUT_SCALE 0
#define HUMMING_IS_TENSOR_INPUT_SCALE_2 0
#define HUMMING_USE_BLOCK_SCALED_MMA 0

#define HUMMING_IS_INDEXED_GEMM 0
#define HUMMING_IS_GROUPED_GEMM 0
#define HUMMING_IS_GROUPED_CONTIGUOUS_GEMM 0

#define HUMMING_OUTPUT_CHUNK_ROWS 0
#define HUMMING_USE_WARP_SPEC 0
#define HUMMING_USE_TMA 0
#define HUMMING_USE_TMA_C 0
#define HUMMING_RASTER_GROUP_M 1
#define HUMMING_MMA_TYPE_ID 0
#define HUMMING_USE_UMMA_SS 0
#define HUMMING_SMEM_REUSE_MODE_ID 2

#define HUMMING_USE_NATIVE_LDMATRIX_DEQUANT 0
#define HUMMING_BLOCK_SHAPE_M 32
#define HUMMING_BLOCK_SHAPE_N 128
#define HUMMING_BLOCK_SHAPE_K 64

#include <humming/kernel/humming.cuh>

class MmaOpClass {
public:
  using ElementB = FloatingPointType<16, 5, 10>;
  static constexpr MmaType kMmaType = MmaType::MMA;
  using MmaShape = Shape<16, 8, 16>;

  using ValTypeC = float;
  using ValTypeD = float;

  static constexpr uint32_t kATypeBits = 16;
  static constexpr uint32_t kBTypeBits = 16;
  static constexpr uint32_t kCTypeBits = 32;
  static constexpr uint32_t kDTypeBits = 32;
  static constexpr bool kNativeMixed = false;

  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];
  using DRegisters = float[4];

  CUDA_INLINE
  static void fma(uint32_t *a, uint32_t *b, float *c, float *d) {
    d[0] = c[0];
    d[1] = c[1];
    d[2] = c[2];
    d[3] = c[3];
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
         "r"(b[0]),  "r"(b[1])
    );
  };
};

template<uint32_t N, uint32_t K>
class LayerConfig {
public:
  static constexpr auto kSmVersion = PLAIT_HUMMING_SM;
  static constexpr auto kShapeN = N;
  static constexpr auto kShapeK = K;
  static constexpr auto kPadShapeN = 0u;
  static constexpr auto kPadShapeK = 0u;
  static constexpr auto kNumExperts = 0u;
  static constexpr auto kInputQuantMode = InputQuantizationMode::Disabled;
  static constexpr auto kInputScaleGroupSize = 0u;
  static constexpr auto kWeightScaleGroupSize = 128u;
  static constexpr auto kWeightScaleGroupSizeN = 0u;
  static constexpr auto kWeightScaleType = WeightScaleType::GROUP;
  static constexpr auto kWeightScale2Type = WeightScale2Type::NONE;
  static constexpr auto kUseIntWeightScale = false;
  static constexpr auto kUseFusedE8m0Scale = false;
  static constexpr auto kHasZeroPoint = true;
  static constexpr auto kIsFpZeroPoint = false;
  static constexpr auto kHasBias = false;
  static constexpr auto kUsePackedKLayout = false;
  static constexpr auto kIsChannelWeightScale = false;
  static constexpr auto kIsBlockWeightScale = false;
  static constexpr auto kIsGroupWeightScale = true;
  static constexpr auto kIsTensorWeightScale = false;
  static constexpr auto kIsChannelWeightScale2 = false;
  static constexpr auto kIsTensorWeightScale2 = false;
  static constexpr auto kHasChannelWeightScale = false;
  static constexpr auto kHasTensorWeightScale = false;
  static constexpr auto kHasInputScale = false;
  static constexpr auto kHasInputScale2 = false;
  static constexpr auto kIsGroupInputScale = false;
  static constexpr auto kIsTokenInputScale = false;
  static constexpr auto kIsTensorInputScale = false;
  static constexpr auto kIsTokenInputScale2 = false;
  static constexpr auto kIsTensorInputScale2 = false;
  static constexpr auto kUseRawWeight = false;
  static constexpr auto kUseFlatWeight = false;
  static constexpr auto kUseLdmatrixDequant = false;
  static constexpr auto kUseNativeDequant = false;
  static constexpr auto kUseBlockScaledMma = false;
  static constexpr auto kMmaScaleGroupSize = 128u;
  static constexpr auto kWeightShapeN = N;
  static constexpr auto kWeightShapeK = K;
};

class ComputeConfig {
public:
  static constexpr auto kUseF16Accum = false;
  static constexpr auto kUseBatchInvariant = false;
  static constexpr auto kUseMMajorInputScale = false;
  static constexpr auto kGemmType = GemmType::DENSE;
  static constexpr auto kGemmTypeId = 0u;
  static constexpr auto kIsIndexedGemm = false;
  static constexpr auto kIsGroupedGemm = false;
  static constexpr auto kIsGroupedContiguousGemm = false;
  static constexpr auto kIsGroupedMaskedGemm = false;
};

class TuningConfig {
public:
  static constexpr auto kMmaType = MmaType::MMA;
  static constexpr auto kUseStreamK = false;
  static constexpr auto kWgmmaUseLateAS = false;
  static constexpr auto kWgmmaSplitIssueWait = false;
  static constexpr auto kNumStages = 3u;
  static constexpr auto kProducerStageUnroll = 3u;
  static constexpr auto kConsumerStageUnroll = 3u;
  static constexpr auto kNumCtasPerSm = 1u;
  static constexpr auto kUmmaNumDequantWarpgroups = 1u;
  static constexpr auto kUmmaCtaGroupSize = 1u;
  static constexpr auto kUmmaSplitAsyncLoads = false;
  static constexpr auto kOutputChunkRows = 0u;
  static constexpr auto kUseWarpSpec = false;
  static constexpr auto kUseTma = false;
  static constexpr auto kUseTmaA = false;
  static constexpr auto kUseTmaAS = false;
  static constexpr auto kUseTmaAS2 = false;
  static constexpr auto kUseTmaB = false;
  static constexpr auto kUseTmaC = false;
  static constexpr auto kUseTmaBS = false;
  static constexpr auto kUseTmaBS2 = false;
  static constexpr auto kUseTmaBZP = false;
  static constexpr auto kUseTmaBias = false;
  static constexpr auto kSmemReuseMode = SmemReuseMode::ALL_STAGES;
  static constexpr auto kMultiCastSizeA = 1u;
  static constexpr auto kMultiCastSizeB = 1u;
  static constexpr auto kUsePdl = false;
  static constexpr auto kRasterGroupM = 1u;
  static constexpr auto kEvictWeightsFirst = false;
  static constexpr auto kMmaTypeId = 0u;
  static constexpr auto kUseUmmaSs = false;
  static constexpr auto kNumThreads = 128u;
  static constexpr auto kNumMathThreads = 128u;
  static constexpr auto kNumLoadThreads = 128u;
  static constexpr auto kSmemReuseModeId = 2u;
};
