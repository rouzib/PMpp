// Optional PM++ CUDA routing kernels.
//
// This translation unit is not part of the portable pmpp wheel.  It is built
// as a small typed-JAX-FFI shared library by cuda/CMakeLists.txt.  The handlers
// only enqueue work on the stream supplied by XLA; they never synchronize,
// allocate with cudaMalloc, create a private stream, or communicate with a
// remote device.

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr int kThreads = 256;
constexpr int kRecordWordsF32 = 8;
constexpr int kRecordWordsF64 = 14;

struct Key64 {
  uint32_t lo;
  uint32_t hi;
};

__host__ __device__ constexpr bool KeyLess(Key64 left, Key64 right) {
  return left.hi < right.hi ||
         (left.hi == right.hi && left.lo < right.lo);
}

__host__ __device__ constexpr bool KeyEqual(Key64 left, Key64 right) {
  return left.lo == right.lo && left.hi == right.hi;
}

__host__ __device__ constexpr bool KeyLessEqual(Key64 left, Key64 right) {
  return KeyLess(left, right) || KeyEqual(left, right);
}

__host__ __device__ constexpr size_t VectorOffset(size_t row,
                                                   size_t component = 0) {
  return row * size_t{3} + component;
}

__host__ __device__ constexpr size_t RecordOffset(size_t row,
                                                   size_t record_words) {
  return row * record_words;
}

__device__ Key64 LoadKey(const uint32_t* words) {
  return Key64{words[0], words[1]};
}

__device__ void StoreKey(uint32_t* words, Key64 key) {
  words[0] = key.lo;
  words[1] = key.hi;
}

__global__ void OffsetProbeKernel(const uint32_t* row_lo,
                                  const uint32_t* row_hi,
                                  int component, int record_words,
                                  uint32_t* output) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  uint64_t row = static_cast<uint64_t>(*row_lo) |
                 (static_cast<uint64_t>(*row_hi) << 32);
  size_t vector_offset = VectorOffset(static_cast<size_t>(row),
                                      static_cast<size_t>(component));
  size_t record_offset = RecordOffset(static_cast<size_t>(row),
                                      static_cast<size_t>(record_words));
  output[0] = static_cast<uint32_t>(vector_offset);
  output[1] = static_cast<uint32_t>(static_cast<uint64_t>(vector_offset) >> 32);
  output[2] = static_cast<uint32_t>(record_offset);
  output[3] = static_cast<uint32_t>(static_cast<uint64_t>(record_offset) >> 32);
}

template <typename Real>
struct RecordTraits;

template <>
struct RecordTraits<float> {
  static constexpr int kValueWords = 1;
  static constexpr int kRecordWords = kRecordWordsF32;

  __device__ static void Store(uint32_t* output, float value) {
    output[0] = __float_as_uint(value);
  }

  __device__ static float Load(const uint32_t* input) {
    return __uint_as_float(input[0]);
  }
};

template <>
struct RecordTraits<double> {
  static constexpr int kValueWords = 2;
  static constexpr int kRecordWords = kRecordWordsF64;

  __device__ static void Store(uint32_t* output, double value) {
    unsigned long long bits = __double_as_longlong(value);
    output[0] = static_cast<uint32_t>(bits);
    output[1] = static_cast<uint32_t>(bits >> 32);
  }

  __device__ static double Load(const uint32_t* input) {
    unsigned long long bits = static_cast<unsigned long long>(input[0]) |
                              (static_cast<unsigned long long>(input[1]) << 32);
    return __longlong_as_double(static_cast<long long>(bits));
  }
};

template <typename Real>
__device__ bool InPeriodicInterval(Real value, int start, int end) {
  // This is deliberately the same comparison/order as
  // pmpp.halo_moving.particles_in_slice_mask: [start, end), with a wrapped
  // interval represented by start > end.
  if (start > end) return value >= static_cast<Real>(start) ||
                            value < static_cast<Real>(end);
  return value >= static_cast<Real>(start) &&
         value < static_cast<Real>(end);
}

template <typename Coord>
__device__ Key64 RaveledKeyWithX(const Coord* pmid, int row, int mesh_x,
                                 int mesh_y, int mesh_z) {
  int64_t x = pmid[VectorOffset(row, 0)] % mesh_x;
  int64_t y = pmid[VectorOffset(row, 1)] % mesh_y;
  int64_t z = pmid[VectorOffset(row, 2)] % mesh_z;
  if (x < 0) x += mesh_x;
  if (y < 0) y += mesh_y;
  if (z < 0) z += mesh_z;
  uint64_t key = (static_cast<uint64_t>(x) * static_cast<uint64_t>(mesh_y) +
                  static_cast<uint64_t>(y)) *
                     static_cast<uint64_t>(mesh_z) +
                 static_cast<uint64_t>(z);
  return Key64{static_cast<uint32_t>(key), static_cast<uint32_t>(key >> 32)};
}

template <typename Real>
__global__ void ClassifyKernel(const Real* x_mod, const uint8_t* valid,
                               uint8_t* classes,
                               uint32_t* block_counts, int n, int global_nmesh,
                               const int32_t* owned_start,
                               const int32_t* owned_end,
                               const int32_t* slice_width, int direction,
                               int num_devices) {
  __shared__ uint32_t block_count;
  if (threadIdx.x == 0) block_count = 0;
  __syncthreads();

  int row = blockIdx.x * blockDim.x + threadIdx.x;

  uint8_t result = 0;  // invalid/padding
  bool choose = false;
  if (row < n && valid[row] != 0) {
    Real x = x_mod[row];
    int start = *owned_start;
    int end = *owned_end;
    int width = *slice_width;
    bool stay = InPeriodicInterval(x, start, end);
    int left_start = (start - width) % global_nmesh;
    if (left_start < 0) left_start += global_nmesh;
    int right_end = (end + width) % global_nmesh;
    bool send_left = InPeriodicInterval(x, left_start, start);
    bool send_right = num_devices == 2
                          ? false
                          : InPeriodicInterval(x, end, right_end);
    if (stay) {
      result = 1;
    } else if (send_left) {
      result = 2;
      choose = direction < 0;
    } else if (send_right) {
      result = 3;
      choose = direction > 0;
    } else {
      result = 4;  // valid but outside the one-hop routing domain
    }
  }
  if (row < n) classes[row] = result;
  if (choose) atomicAdd(&block_count, 1u);
  __syncthreads();
  if (threadIdx.x == 0) block_counts[blockIdx.x] = block_count;
}

template <typename Real>
__global__ void WriteRecordsKernel(const int32_t* pmid, const Real* disp,
                                   const Real* vel, const uint8_t* classes,
                                   const uint32_t* block_offsets,
                                   uint32_t* records,
                                   int n, int mesh_x, int mesh_y, int mesh_z,
                                   int capacity, int direction) {
  using BlockScan = cub::BlockScan<uint32_t, kThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  const uint8_t expected = direction < 0 ? 2 : 3;
  uint32_t selected = row < n && classes[row] == expected ? 1u : 0u;
  uint32_t local_rank = 0;
  uint32_t block_total = 0;
  BlockScan(scan_storage).ExclusiveSum(selected, local_rank, block_total);
  if (row >= n || selected == 0) return;
  uint32_t output = block_offsets[blockIdx.x] + local_rank;
  if (output >= static_cast<uint32_t>(capacity)) return;

  constexpr int value_words = RecordTraits<Real>::kValueWords;
  constexpr int record_words = RecordTraits<Real>::kRecordWords;
  uint32_t* record = records + RecordOffset(output, record_words);
  StoreKey(record, RaveledKeyWithX(pmid, row, mesh_x, mesh_y, mesh_z));
  for (int component = 0; component < 3; ++component) {
    RecordTraits<Real>::Store(record + 2 + component * value_words,
                              disp[VectorOffset(row, component)]);
    RecordTraits<Real>::Store(record + 2 + 3 * value_words +
                                  component * value_words,
                              vel[VectorOffset(row, component)]);
  }
}

__global__ void WriteCountKernel(const uint32_t* block_counts,
                                 const uint32_t* block_offsets, int32_t* count,
                                 int num_blocks) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (num_blocks == 0) {
    count[0] = 0;
  } else {
    int last = num_blocks - 1;
    count[0] = static_cast<int32_t>(block_offsets[last] + block_counts[last]);
  }
}

// The bidirectional path classifies each authoritative slot exactly once.
// Three block-count arrays are then scanned and consumed by one stable write
// kernel, so the left/right payloads and compact stay descriptors all retain
// authoritative input order.
template <typename Real>
__global__ void ClassifyBidirKernel(
    const Real* x_mod, const uint8_t* valid, uint8_t* classes,
    uint32_t* left_block_counts, uint32_t* right_block_counts,
    uint32_t* stay_block_counts, int n, int global_nmesh,
    const int32_t* owned_start_buffer, const int32_t* owned_end_buffer,
    int slice_width, int num_devices) {
  __shared__ uint32_t left_count;
  __shared__ uint32_t right_count;
  __shared__ uint32_t stay_count;
  if (threadIdx.x == 0) {
    left_count = 0;
    right_count = 0;
    stay_count = 0;
  }
  __syncthreads();

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  uint8_t result = 0;
  if (row < n && valid[row] != 0) {
    Real x = fmod(x_mod[row], static_cast<Real>(global_nmesh));
    if (x < static_cast<Real>(0)) x += static_cast<Real>(global_nmesh);
    const int owned_start = *owned_start_buffer;
    const int owned_end = *owned_end_buffer;
    int left_start = (owned_start - slice_width) % global_nmesh;
    if (left_start < 0) left_start += global_nmesh;
    int right_end = (owned_end + slice_width) % global_nmesh;
    bool stay = InPeriodicInterval(x, owned_start, owned_end);
    bool send_left = InPeriodicInterval(x, left_start, owned_start);
    bool send_right = num_devices == 2
                          ? false
                          : InPeriodicInterval(x, owned_end, right_end);
    if (stay) {
      result = 1;
      atomicAdd(&stay_count, 1u);
    } else if (send_left) {
      result = 2;
      atomicAdd(&left_count, 1u);
    } else if (send_right) {
      result = 3;
      atomicAdd(&right_count, 1u);
    } else {
      result = 4;
    }
  }
  if (row < n) classes[row] = result;
  __syncthreads();
  if (threadIdx.x == 0) {
    left_block_counts[blockIdx.x] = left_count;
    right_block_counts[blockIdx.x] = right_count;
    stay_block_counts[blockIdx.x] = stay_count;
  }
}

template <typename Real, typename Coord>
__device__ void WriteRouteRecord(const Coord* pmid, const Real* disp,
                                 const Real* vel, int row, int mesh_x,
                                 int mesh_y, int mesh_z, uint32_t* records,
                                 uint32_t output) {
  constexpr int value_words = RecordTraits<Real>::kValueWords;
  constexpr int record_words = RecordTraits<Real>::kRecordWords;
  uint32_t* record = records + RecordOffset(output, record_words);
  StoreKey(record, RaveledKeyWithX(pmid, row, mesh_x, mesh_y, mesh_z));
  for (int component = 0; component < 3; ++component) {
    RecordTraits<Real>::Store(record + 2 + component * value_words,
                              disp[VectorOffset(row, component)]);
    RecordTraits<Real>::Store(record + 2 + 3 * value_words +
                                  component * value_words,
                              vel[VectorOffset(row, component)]);
  }
}

template <typename Real, typename Coord>
__global__ void WriteBidirRecordsKernel(
    const Coord* pmid, const Real* disp, const Real* vel,
    const uint8_t* classes, const uint32_t* left_offsets,
    const uint32_t* right_offsets, const uint32_t* stay_offsets,
    uint32_t* left_records, uint32_t* right_records,
    int32_t* stay_indices, int n, int mesh_x, int mesh_y, int mesh_z,
    int record_capacity, int stay_capacity) {
  using BlockScan = cub::BlockScan<uint32_t, kThreads>;
  __shared__ typename BlockScan::TempStorage left_scan_storage;
  __shared__ typename BlockScan::TempStorage right_scan_storage;
  __shared__ typename BlockScan::TempStorage stay_scan_storage;

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t left_rank = 0;
  uint32_t right_rank = 0;
  uint32_t stay_rank = 0;
  uint32_t block_total = 0;
  uint32_t selected = row < n && classes[row] == 2 ? 1u : 0u;
  BlockScan(left_scan_storage).ExclusiveSum(selected, left_rank, block_total);
  selected = row < n && classes[row] == 3 ? 1u : 0u;
  BlockScan(right_scan_storage).ExclusiveSum(selected, right_rank, block_total);
  selected = row < n && classes[row] == 1 ? 1u : 0u;
  BlockScan(stay_scan_storage).ExclusiveSum(selected, stay_rank, block_total);

  if (row >= n) return;
  uint8_t classification = classes[row];
  if (classification == 2) {
    uint32_t output = left_offsets[blockIdx.x] + left_rank;
    if (output < static_cast<uint32_t>(record_capacity)) {
      WriteRouteRecord<Real, Coord>(pmid, disp, vel, row, mesh_x, mesh_y,
                                    mesh_z, left_records, output);
    }
  } else if (classification == 3) {
    uint32_t output = right_offsets[blockIdx.x] + right_rank;
    if (output < static_cast<uint32_t>(record_capacity)) {
      WriteRouteRecord<Real, Coord>(pmid, disp, vel, row, mesh_x, mesh_y,
                                    mesh_z, right_records, output);
    }
  } else if (classification == 1) {
    uint32_t output = stay_offsets[blockIdx.x] + stay_rank;
    if (output < static_cast<uint32_t>(stay_capacity)) {
      stay_indices[output] = row;
    }
  }
}

// The production forward-only route computes drift and ownership in CUDA.
// It deliberately emits only block-scale stay metadata: no per-particle
// x-position, class, key, or compact stay-index arrays cross the FFI boundary.
__device__ uint8_t ClassifyDriftRowI16(
    const int16_t* pmid, const float* disp, const float* vel,
    const bool* valid, int row, int global_nmesh, int owned_start,
    int owned_end, int slice_width, int num_devices, float drift_factor,
    float disp_size) {
  if (valid[row] == 0) return 0;
  float drifted_x = disp[VectorOffset(row, 0)] +
                    vel[VectorOffset(row, 0)] * drift_factor;
  float x = static_cast<float>(pmid[VectorOffset(row, 0)]) +
            drifted_x * disp_size;
  x = fmodf(x, static_cast<float>(global_nmesh));
  if (x < 0.0f) x += static_cast<float>(global_nmesh);
  int left_start = (owned_start - slice_width) % global_nmesh;
  if (left_start < 0) left_start += global_nmesh;
  int right_end = (owned_end + slice_width) % global_nmesh;
  if (InPeriodicInterval(x, owned_start, owned_end)) return 1;
  if (InPeriodicInterval(x, left_start, owned_start)) return 2;
  if (num_devices != 2 && InPeriodicInterval(x, owned_end, right_end)) {
    return 3;
  }
  return 4;
}

__global__ void ClassifyDriftBidirCountsKernel(
    const int16_t* pmid, const float* disp, const float* vel,
    const bool* valid, uint32_t* left_block_counts,
    uint32_t* right_block_counts, uint32_t* stay_block_counts,
    uint32_t* invalid_block_counts, int n, int global_nmesh,
    const int32_t* owned_start_buffer, const int32_t* owned_end_buffer,
    int slice_width, int num_devices, const float* drift_factor_buffer,
    const float* disp_size_buffer) {
  __shared__ uint32_t left_count;
  __shared__ uint32_t right_count;
  __shared__ uint32_t stay_count;
  __shared__ uint32_t invalid_count;
  if (threadIdx.x == 0) {
    left_count = 0;
    right_count = 0;
    stay_count = 0;
    invalid_count = 0;
  }
  __syncthreads();

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  uint8_t classification = 0;
  if (row < n) {
    classification = ClassifyDriftRowI16(
        pmid, disp, vel, valid, row, global_nmesh, *owned_start_buffer,
        *owned_end_buffer, slice_width, num_devices, *drift_factor_buffer,
        *disp_size_buffer);
  }
  if (classification == 1) atomicAdd(&stay_count, 1u);
  if (classification == 2) atomicAdd(&left_count, 1u);
  if (classification == 3) atomicAdd(&right_count, 1u);
  if (classification == 4) atomicAdd(&invalid_count, 1u);
  __syncthreads();
  if (threadIdx.x == 0) {
    left_block_counts[blockIdx.x] = left_count;
    right_block_counts[blockIdx.x] = right_count;
    stay_block_counts[blockIdx.x] = stay_count;
    invalid_block_counts[blockIdx.x] = invalid_count;
  }
}

__device__ void WriteDriftRouteRecordI16(
    const int16_t* pmid, const float* disp, const float* vel, int row,
    int mesh_x, int mesh_y, int mesh_z, float drift_factor,
    uint32_t* records, uint32_t output) {
  constexpr int record_words = RecordTraits<float>::kRecordWords;
  uint32_t* record = records + RecordOffset(output, record_words);
  StoreKey(record, RaveledKeyWithX(pmid, row, mesh_x, mesh_y, mesh_z));
  for (int component = 0; component < 3; ++component) {
    float drifted = disp[VectorOffset(row, component)] +
                    vel[VectorOffset(row, component)] * drift_factor;
    RecordTraits<float>::Store(record + 2 + component, drifted);
    RecordTraits<float>::Store(record + 5 + component,
                               vel[VectorOffset(row, component)]);
  }
}

__global__ void WriteDriftBidirRecordsKernelI16(
    const int16_t* pmid, const float* disp, const float* vel,
    const bool* valid, const uint32_t* left_offsets,
    const uint32_t* right_offsets, uint32_t* left_records,
    uint32_t* right_records, int n, int global_nmesh,
    const int32_t* owned_start_buffer, const int32_t* owned_end_buffer,
    int slice_width, int num_devices, const float* drift_factor_buffer,
    const float* disp_size_buffer, int mesh_x, int mesh_y, int mesh_z,
    int record_capacity) {
  using BlockScan = cub::BlockScan<uint32_t, kThreads>;
  __shared__ typename BlockScan::TempStorage left_scan_storage;
  __shared__ typename BlockScan::TempStorage right_scan_storage;

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  uint8_t classification = 0;
  if (row < n) {
    classification = ClassifyDriftRowI16(
        pmid, disp, vel, valid, row, global_nmesh, *owned_start_buffer,
        *owned_end_buffer, slice_width, num_devices, *drift_factor_buffer,
        *disp_size_buffer);
  }
  uint32_t local_rank = 0;
  uint32_t block_total = 0;
  uint32_t selected = classification == 2 ? 1u : 0u;
  BlockScan(left_scan_storage).ExclusiveSum(selected, local_rank, block_total);
  if (classification == 2) {
    uint32_t output = left_offsets[blockIdx.x] + local_rank;
    if (output < static_cast<uint32_t>(record_capacity)) {
      WriteDriftRouteRecordI16(pmid, disp, vel, row, mesh_x, mesh_y,
                               mesh_z, *drift_factor_buffer, left_records,
                               output);
    }
  }
  __syncthreads();
  selected = classification == 3 ? 1u : 0u;
  BlockScan(right_scan_storage).ExclusiveSum(selected, local_rank,
                                               block_total);
  if (classification == 3) {
    uint32_t output = right_offsets[blockIdx.x] + local_rank;
    if (output < static_cast<uint32_t>(record_capacity)) {
      WriteDriftRouteRecordI16(pmid, disp, vel, row, mesh_x, mesh_y,
                               mesh_z, *drift_factor_buffer, right_records,
                               output);
    }
  }
}

__global__ void BuildStayBlockMetadataKernel(
    const uint8_t* stay, uint8_t* local_prefix, uint32_t* block_counts,
    int n) {
  using BlockScan = cub::BlockScan<uint32_t, kThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t selected = row < n && stay[row] != 0 ? 1u : 0u;
  uint32_t local_rank = 0;
  uint32_t block_total = 0;
  BlockScan(scan_storage).ExclusiveSum(selected, local_rank, block_total);
  if (row < n) local_prefix[row] = static_cast<uint8_t>(local_rank);
  if (threadIdx.x == kThreads - 1) block_counts[blockIdx.x] = block_total;
}

__device__ int FindStayIndex(const uint32_t* block_offsets,
                             const uint32_t* block_counts,
                             const uint8_t* local_prefix, const uint8_t* stay,
                             int n, int num_blocks, uint32_t rank) {
  int block_lo = 0;
  int block_hi = num_blocks;
  while (block_lo < block_hi) {
    int mid = block_lo + (block_hi - block_lo) / 2;
    uint32_t end = block_offsets[mid] + block_counts[mid];
    if (end > rank) block_hi = mid;
    else block_lo = mid + 1;
  }
  int block = block_lo < num_blocks ? block_lo : num_blocks - 1;
  uint32_t local_rank = rank - block_offsets[block];
  int block_start = block * kThreads;
  int block_end = min(block_start + kThreads, n);
  int lo = block_start;
  int hi = block_end;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    uint32_t end = static_cast<uint32_t>(local_prefix[mid]) +
                   static_cast<uint32_t>(stay[mid] != 0);
    if (end > local_rank) hi = mid;
    else lo = mid + 1;
  }
  return lo < block_end ? lo : block_end - 1;
}

__device__ Key64 StayKeyAt(const int32_t* pmid,
                           const uint32_t* block_offsets,
                           const uint32_t* block_counts,
                           const uint8_t* local_prefix, const uint8_t* stay,
                           int n, int num_blocks, uint32_t rank,
                           int mesh_x, int mesh_y, int mesh_z) {
  int row = FindStayIndex(block_offsets, block_counts, local_prefix, stay, n,
                          num_blocks, rank);
  return RaveledKeyWithX(pmid, row, mesh_x, mesh_y, mesh_z);
}

template <typename Real>
__device__ Key64 IncomingKeyAt(const uint32_t* records, uint32_t rank) {
  return LoadKey(records + RecordOffset(
                               rank, RecordTraits<Real>::kRecordWords));
}

template <typename Real>
__device__ uint32_t LowerBoundIncoming(const uint32_t* records, uint32_t count,
                                       Key64 key) {
  uint32_t lo = 0;
  uint32_t hi = count;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    if (KeyLess(IncomingKeyAt<Real>(records, mid), key)) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

__device__ uint32_t UpperBoundStay(
    const int32_t* pmid, const uint32_t* block_offsets,
    const uint32_t* block_counts, const uint8_t* local_prefix,
    const uint8_t* stay, int n, int num_blocks, uint32_t count, Key64 key,
    int mesh_x, int mesh_y, int mesh_z) {
  uint32_t lo = 0;
  uint32_t hi = count;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    Key64 candidate = StayKeyAt(
        pmid, block_offsets, block_counts, local_prefix, stay, n, num_blocks,
        mid, mesh_x, mesh_y, mesh_z);
    if (KeyLessEqual(candidate, key)) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

template <typename Real>
__device__ uint32_t StayOutputPosition(const int32_t* pmid,
                                       const uint32_t* block_offsets,
                                       const uint32_t* block_counts,
                                       const uint8_t* local_prefix,
                                       const uint8_t* stay, int n,
                                       int num_blocks,
                                       const uint32_t* records,
                                       uint32_t incoming_count, uint32_t rank,
                                       int mesh_x, int mesh_y, int mesh_z) {
  Key64 key = StayKeyAt(pmid, block_offsets, block_counts, local_prefix,
                        stay, n, num_blocks, rank, mesh_x, mesh_y, mesh_z);
  return rank + LowerBoundIncoming<Real>(records, incoming_count, key);
}

template <typename Real>
__device__ uint32_t IncomingOutputPosition(const int32_t* pmid,
                                            const uint32_t* block_offsets,
                                            const uint32_t* block_counts,
                                            const uint8_t* local_prefix,
                                            const uint8_t* stay, int n,
                                            int num_blocks,
                                            const uint32_t* records,
                                            uint32_t incoming_count,
                                            uint32_t stay_count, uint32_t rank,
                                            int mesh_x, int mesh_y, int mesh_z) {
  Key64 key = IncomingKeyAt<Real>(records, rank);
  return rank + UpperBoundStay(pmid, block_offsets, block_counts, local_prefix,
                               stay, n, num_blocks, stay_count, key, mesh_x,
                               mesh_y, mesh_z);
}

template <bool Auxiliary, typename Real>
__global__ void MergeKernel(const int32_t* pmid, const Real* disp,
                            const Real* vel, const uint8_t* stay,
                            const uint32_t* stay_block_offsets,
                            const uint32_t* stay_block_counts,
                            const uint8_t* stay_local_prefix, int num_blocks,
                            const uint32_t* incoming_records,
                            const int32_t* incoming_count, int n, int mesh_x,
                            int mesh_y, int mesh_z, int capacity,
                            int32_t* out_pmid, Real* out_disp, Real* out_vel,
                            uint8_t* out_valid, uint8_t* out_tag,
                            int32_t* out_index) {
  int output = blockIdx.x * blockDim.x + threadIdx.x;
  if (output >= capacity) return;
  uint32_t stay_count = num_blocks == 0
                            ? 0u
                            : stay_block_offsets[num_blocks - 1] +
                                  stay_block_counts[num_blocks - 1];
  int incoming_signed = *incoming_count;
  uint32_t incoming = incoming_signed <= 0
                          ? 0u
                          : min(static_cast<uint32_t>(incoming_signed),
                                static_cast<uint32_t>(capacity));
  uint32_t total = stay_count + incoming;
  if (static_cast<uint32_t>(output) >= total) return;

  // Find the unique source whose stable merge position is this output slot.
  // Existing/stay entries use a strict lower-bound in the incoming stream;
  // incoming entries use an upper-bound in the stay stream.  Consequently a
  // tie always keeps the existing entry before the incoming entry.
  uint32_t lo = 0;
  uint32_t hi = stay_count;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    uint32_t position = StayOutputPosition<Real>(
        pmid, stay_block_offsets, stay_block_counts, stay_local_prefix, stay, n,
        num_blocks, incoming_records, incoming, mid, mesh_x, mesh_y, mesh_z);
    if (position < static_cast<uint32_t>(output)) lo = mid + 1;
    else hi = mid;
  }
  bool from_stay = lo < stay_count &&
                   StayOutputPosition<Real>(
                       pmid, stay_block_offsets, stay_block_counts,
                       stay_local_prefix, stay, n, num_blocks, incoming_records,
                       incoming, lo, mesh_x, mesh_y, mesh_z) ==
                       static_cast<uint32_t>(output);
  uint32_t source_rank = 0;
  int source_row = -1;
  if (from_stay) {
    source_rank = lo;
    source_row = FindStayIndex(stay_block_offsets, stay_block_counts,
                               stay_local_prefix, stay, n, num_blocks,
                               source_rank);
    out_pmid[VectorOffset(output, 0)] = pmid[VectorOffset(source_row, 0)];
    out_pmid[VectorOffset(output, 1)] = pmid[VectorOffset(source_row, 1)];
    out_pmid[VectorOffset(output, 2)] = pmid[VectorOffset(source_row, 2)];
    for (int component = 0; component < 3; ++component) {
      out_disp[VectorOffset(output, component)] =
          disp[VectorOffset(source_row, component)];
      out_vel[VectorOffset(output, component)] =
          vel[VectorOffset(source_row, component)];
    }
    if constexpr (Auxiliary) {
      out_tag[output] = 0;
      // The adjoint consumes a compact stay stream, so its index is the
      // stream rank rather than the original authoritative slot.  The latter
      // is supplied separately by stay_pos.
      out_index[output] = static_cast<int32_t>(source_rank);
    }
  } else {
    uint32_t blo = 0;
    uint32_t bhi = incoming;
    while (blo < bhi) {
      uint32_t mid = blo + (bhi - blo) / 2;
      uint32_t position = IncomingOutputPosition<Real>(
          pmid, stay_block_offsets, stay_block_counts, stay_local_prefix, stay,
          n, num_blocks, incoming_records, incoming, stay_count, mid, mesh_x,
          mesh_y, mesh_z);
      if (position < static_cast<uint32_t>(output)) blo = mid + 1;
      else bhi = mid;
    }
    source_rank = blo < incoming ? blo : incoming - 1;
    constexpr int value_words = RecordTraits<Real>::kValueWords;
    constexpr int record_words = RecordTraits<Real>::kRecordWords;
    const uint32_t* record =
        incoming_records + RecordOffset(source_rank, record_words);
    Key64 key = LoadKey(record);
    uint64_t raveled = static_cast<uint64_t>(key.lo) |
                       (static_cast<uint64_t>(key.hi) << 32);
    uint64_t yz_stride = static_cast<uint64_t>(mesh_y) * mesh_z;
    out_pmid[VectorOffset(output, 0)] =
        static_cast<int32_t>(raveled / yz_stride);
    uint64_t yz = raveled % yz_stride;
    out_pmid[VectorOffset(output, 1)] =
        static_cast<int32_t>(yz / static_cast<uint64_t>(mesh_z));
    out_pmid[VectorOffset(output, 2)] =
        static_cast<int32_t>(yz % static_cast<uint64_t>(mesh_z));
    for (int component = 0; component < 3; ++component) {
      out_disp[VectorOffset(output, component)] = RecordTraits<Real>::Load(
          record + 2 + component * value_words);
      out_vel[VectorOffset(output, component)] = RecordTraits<Real>::Load(
          record + 2 + 3 * value_words + component * value_words);
    }
    if constexpr (Auxiliary) {
      out_tag[output] = 1;
      out_index[output] = static_cast<int32_t>(source_rank);
    }
  }
  out_valid[output] = 1;
}

template <bool Auxiliary, ffi::DataType DType>
ffi::Error LaunchMerge(
    cudaStream_t stream, ffi::ScratchAllocator scratch,
    ffi::Buffer<ffi::S32> pmid, ffi::Buffer<DType> disp,
    ffi::Buffer<DType> vel, ffi::Buffer<ffi::U8> stay,
    ffi::Buffer<ffi::U32> incoming_records,
    ffi::Buffer<ffi::S32> incoming_count, ffi::ResultBuffer<ffi::S32> out_pmid,
    ffi::ResultBuffer<DType> out_disp, ffi::ResultBuffer<DType> out_vel,
    ffi::ResultBuffer<ffi::U8> out_valid,
    ffi::ResultBuffer<ffi::U8>* out_tag,
    ffi::ResultBuffer<ffi::S32>* out_index, int mesh_x, int mesh_y, int mesh_z,
    int capacity) {
  using Real = ffi::NativeType<DType>;
  static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>);
  int n = static_cast<int>(pmid.element_count() / 3);
  if (n <= 0 || capacity < 0) return ffi::Error::InvalidArgument("invalid route-merge shape");
  int num_blocks = (n + kThreads - 1) / kThreads;
  auto local_prefix_mem = scratch.Allocate(sizeof(uint8_t) * n, alignof(uint8_t));
  if (!local_prefix_mem) {
    return ffi::Error::Internal("unable to allocate route-merge local-prefix scratch");
  }
  auto block_counts_mem = scratch.Allocate(sizeof(uint32_t) * num_blocks,
                                           alignof(uint32_t));
  if (!block_counts_mem) {
    return ffi::Error::Internal("unable to allocate route-merge block-count scratch");
  }
  auto block_offsets_mem = scratch.Allocate(sizeof(uint32_t) * num_blocks,
                                            alignof(uint32_t));
  if (!block_offsets_mem) {
    return ffi::Error::Internal("unable to allocate route-merge block-offset scratch");
  }
  auto* local_prefix = static_cast<uint8_t*>(*local_prefix_mem);
  auto* block_counts = static_cast<uint32_t*>(*block_counts_mem);
  auto* block_offsets = static_cast<uint32_t*>(*block_offsets_mem);
  dim3 blocks((n + kThreads - 1) / kThreads);
  BuildStayBlockMetadataKernel<<<blocks, kThreads, 0, stream>>>(
      stay.typed_data(), local_prefix, block_counts, n);
  if (cudaGetLastError() != cudaSuccess) {
    return ffi::Error::Internal("route-merge metadata launch failed");
  }

  size_t temp_bytes = 0;
  cudaError_t status = cub::DeviceScan::ExclusiveSum(
      nullptr, temp_bytes, block_counts, block_offsets, num_blocks, stream);
  if (status != cudaSuccess) return ffi::Error::Internal("CUB route-merge scan query failed");
  auto temp_mem = scratch.Allocate(temp_bytes, 1);
  if (!temp_mem) {
    return ffi::Error::Internal("unable to allocate CUB route-merge workspace (bytes=" +
                                std::to_string(temp_bytes) + ")");
  }
  status = cub::DeviceScan::ExclusiveSum(
      *temp_mem, temp_bytes, block_counts, block_offsets, num_blocks, stream);
  if (status != cudaSuccess) return ffi::Error::Internal("CUB route-merge scan failed");

  status = cudaMemsetAsync(out_pmid->typed_data(), 0,
                           sizeof(int32_t) * capacity * 3, stream);
  if (status != cudaSuccess) return ffi::Error::Internal("route-merge pmid clear failed");
  status = cudaMemsetAsync(out_disp->typed_data(), 0,
                           sizeof(Real) * capacity * 3, stream);
  if (status != cudaSuccess) return ffi::Error::Internal("route-merge displacement clear failed");
  status = cudaMemsetAsync(out_vel->typed_data(), 0,
                           sizeof(Real) * capacity * 3, stream);
  if (status != cudaSuccess) return ffi::Error::Internal("route-merge velocity clear failed");
  status = cudaMemsetAsync(out_valid->typed_data(), 0,
                           sizeof(uint8_t) * capacity, stream);
  if (status != cudaSuccess) return ffi::Error::Internal("route-merge validity clear failed");
  if constexpr (Auxiliary) {
    status = cudaMemsetAsync((*out_tag)->typed_data(), 0,
                             sizeof(uint8_t) * capacity, stream);
    if (status != cudaSuccess) return ffi::Error::Internal("route-merge tag clear failed");
    // Invalid/padded merge rows must carry a negative provenance index. The
    // transpose kernel rejects negative indices; clearing to zero aliases
    // every padded row onto valid compact-stay slot zero.
    status = cudaMemsetAsync((*out_index)->typed_data(), 0xff,
                             sizeof(int32_t) * capacity, stream);
    if (status != cudaSuccess) return ffi::Error::Internal("route-merge index clear failed");
  }

  blocks = dim3((capacity + kThreads - 1) / kThreads);
  MergeKernel<Auxiliary, Real><<<blocks, kThreads, 0, stream>>>(
      pmid.typed_data(), disp.typed_data(), vel.typed_data(), stay.typed_data(),
      block_offsets, block_counts, local_prefix, num_blocks,
      incoming_records.typed_data(), incoming_count.typed_data(), n, mesh_x,
      mesh_y, mesh_z, capacity, out_pmid->typed_data(),
      out_disp->typed_data(), out_vel->typed_data(), out_valid->typed_data(),
      Auxiliary ? (*out_tag)->typed_data() : nullptr,
      Auxiliary ? (*out_index)->typed_data() : nullptr);
  if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("route-merge kernel launch failed");
  return ffi::Error::Success();
}

template <typename Real>
__device__ Key64 RecordKeyAt(const uint32_t* records, uint32_t rank) {
  return LoadKey(records + RecordOffset(
                               rank, RecordTraits<Real>::kRecordWords));
}

template <typename Coord>
__device__ Key64 CompactStayKeyAt(
    const Coord* pmid, const int32_t* stay_indices, uint32_t rank,
    int mesh_x, int mesh_y, int mesh_z) {
  int row = stay_indices[rank];
  return RaveledKeyWithX(pmid, row, mesh_x, mesh_y, mesh_z);
}

template <typename Coord>
__device__ uint32_t UpperBoundCompactStay(
    const Coord* pmid, const int32_t* stay_indices, uint32_t count,
    Key64 key, int mesh_x, int mesh_y, int mesh_z) {
  uint32_t lo = 0;
  uint32_t hi = count;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    if (KeyLessEqual(CompactStayKeyAt(pmid, stay_indices, mid, mesh_x,
                                     mesh_y, mesh_z), key)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// Route records are eight float32 words or fourteen float64 words wide. The
// merge-path searches below use the first two words as the little-endian
// limbs of the raveled particle key. The remaining words are floating-point
// bit patterns and do not form a sorted key stream.
template <typename Real>
__device__ uint32_t LowerBoundRecordKeys(const uint32_t* records,
                                         uint32_t count, Key64 key) {
  uint32_t lo = 0;
  uint32_t hi = count;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    if (KeyLess(RecordKeyAt<Real>(records, mid), key)) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

template <typename Real>
__device__ uint32_t UpperBoundRecordKeys(const uint32_t* records,
                                         uint32_t count, Key64 key) {
  uint32_t lo = 0;
  uint32_t hi = count;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    if (KeyLessEqual(RecordKeyAt<Real>(records, mid), key)) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

__device__ uint32_t UpperBoundDriftSourceRowsI16(
    const int16_t* pmid, const bool* valid, uint32_t n, Key64 key,
    int mesh_x, int mesh_y, int mesh_z) {
  uint32_t lo = 0;
  uint32_t hi = n;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    bool before = valid[mid] != 0 &&
                  KeyLessEqual(RaveledKeyWithX(pmid, static_cast<int>(mid),
                                               mesh_x, mesh_y, mesh_z),
                               key);
    if (before) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__device__ uint32_t UpperBoundDriftStayI16(
    const int16_t* pmid, const float* disp, const float* vel,
    const bool* valid, const uint32_t* stay_block_counts,
    const uint32_t* stay_block_offsets, int n, Key64 key,
    int global_nmesh, int owned_start, int owned_end, int slice_width,
    int num_devices, float drift_factor, float disp_size, int mesh_x,
    int mesh_y, int mesh_z) {
  uint32_t row_limit = UpperBoundDriftSourceRowsI16(
      pmid, valid, static_cast<uint32_t>(n), key, mesh_x, mesh_y, mesh_z);
  uint32_t full_blocks = row_limit / static_cast<uint32_t>(kThreads);
  uint32_t count = 0;
  if (full_blocks > 0) {
    uint32_t last = full_blocks - 1;
    count = stay_block_offsets[last] + stay_block_counts[last];
  }
  uint32_t partial_start = full_blocks * static_cast<uint32_t>(kThreads);
  for (uint32_t row = partial_start; row < row_limit; ++row) {
    count += ClassifyDriftRowI16(
                 pmid, disp, vel, valid, static_cast<int>(row),
                 global_nmesh, owned_start, owned_end, slice_width,
                 num_devices, drift_factor, disp_size) == 1
                 ? 1u
                 : 0u;
  }
  return count;
}

__global__ void ScatterDriftStayKernelI16(
    const int16_t* pmid, const float* disp, const float* vel,
    const bool* valid, const uint32_t* stay_block_counts,
    const uint32_t* stay_block_offsets, const uint32_t* left_records,
    const int32_t* left_count_value, const uint32_t* right_records,
    const int32_t* right_count_value, int n, int global_nmesh,
    const int32_t* owned_start_buffer, const int32_t* owned_end_buffer,
    int slice_width, int num_devices, const float* drift_factor_buffer,
    const float* disp_size_buffer, int mesh_x, int mesh_y, int mesh_z,
    int record_capacity, int capacity, int16_t* out_pmid, float* out_disp,
    float* out_vel, bool* out_valid) {
  using BlockScan = cub::BlockScan<uint32_t, kThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  uint8_t classification = 0;
  if (row < n) {
    classification = ClassifyDriftRowI16(
        pmid, disp, vel, valid, row, global_nmesh, *owned_start_buffer,
        *owned_end_buffer, slice_width, num_devices, *drift_factor_buffer,
        *disp_size_buffer);
  }
  uint32_t local_rank = 0;
  uint32_t block_total = 0;
  uint32_t selected = classification == 1 ? 1u : 0u;
  BlockScan(scan_storage).ExclusiveSum(selected, local_rank, block_total);
  if (classification != 1) return;

  uint32_t stay_rank = stay_block_offsets[blockIdx.x] + local_rank;
  Key64 key = RaveledKeyWithX(pmid, row, mesh_x, mesh_y, mesh_z);
  int left_signed = *left_count_value;
  int right_signed = *right_count_value;
  uint32_t left_count = left_signed <= 0
                            ? 0u
                            : min(static_cast<uint32_t>(left_signed),
                                  static_cast<uint32_t>(record_capacity));
  uint32_t right_count = right_signed <= 0
                             ? 0u
                             : min(static_cast<uint32_t>(right_signed),
                                   static_cast<uint32_t>(record_capacity));
  uint32_t output =
      stay_rank + LowerBoundRecordKeys<float>(left_records, left_count, key) +
      LowerBoundRecordKeys<float>(right_records, right_count, key);
  if (output >= static_cast<uint32_t>(capacity)) return;

  out_pmid[VectorOffset(output, 0)] = pmid[VectorOffset(row, 0)];
  out_pmid[VectorOffset(output, 1)] = pmid[VectorOffset(row, 1)];
  out_pmid[VectorOffset(output, 2)] = pmid[VectorOffset(row, 2)];
  for (int component = 0; component < 3; ++component) {
    out_disp[VectorOffset(output, component)] =
        disp[VectorOffset(row, component)] +
        vel[VectorOffset(row, component)] * *drift_factor_buffer;
    out_vel[VectorOffset(output, component)] =
        vel[VectorOffset(row, component)];
  }
  out_valid[output] = 1;
}

template <bool RightStream>
__global__ void ScatterDriftIncomingKernelI16(
    const int16_t* pmid, const float* disp, const float* vel,
    const bool* valid, const uint32_t* stay_block_counts,
    const uint32_t* stay_block_offsets, const uint32_t* left_records,
    const int32_t* left_count_value, const uint32_t* right_records,
    const int32_t* right_count_value, int n, int global_nmesh,
    const int32_t* owned_start_buffer, const int32_t* owned_end_buffer,
    int slice_width, int num_devices, const float* drift_factor_buffer,
    const float* disp_size_buffer, int mesh_x, int mesh_y, int mesh_z,
    int record_capacity, int capacity, int16_t* out_pmid, float* out_disp,
    float* out_vel, bool* out_valid) {
  uint32_t rank = static_cast<uint32_t>(blockIdx.x) * blockDim.x +
                  static_cast<uint32_t>(threadIdx.x);
  int left_signed = *left_count_value;
  int right_signed = *right_count_value;
  uint32_t left_count = left_signed <= 0
                            ? 0u
                            : min(static_cast<uint32_t>(left_signed),
                                  static_cast<uint32_t>(record_capacity));
  uint32_t right_count = right_signed <= 0
                             ? 0u
                             : min(static_cast<uint32_t>(right_signed),
                                   static_cast<uint32_t>(record_capacity));
  uint32_t source_count = RightStream ? right_count : left_count;
  if (rank >= source_count) return;

  const uint32_t* source_records =
      RightStream ? right_records : left_records;
  Key64 key = RecordKeyAt<float>(source_records, rank);
  uint32_t stay_before = UpperBoundDriftStayI16(
      pmid, disp, vel, valid, stay_block_counts, stay_block_offsets, n, key,
      global_nmesh, *owned_start_buffer, *owned_end_buffer, slice_width,
      num_devices, *drift_factor_buffer, *disp_size_buffer, mesh_x, mesh_y,
      mesh_z);
  uint32_t other_before =
      RightStream ? UpperBoundRecordKeys<float>(left_records, left_count, key)
                  : LowerBoundRecordKeys<float>(right_records, right_count,
                                                key);
  uint32_t output = rank + stay_before + other_before;
  if (output >= static_cast<uint32_t>(capacity)) return;

  constexpr int record_words = RecordTraits<float>::kRecordWords;
  const uint32_t* record =
      source_records + RecordOffset(rank, record_words);
  uint64_t raveled = static_cast<uint64_t>(key.lo) |
                     (static_cast<uint64_t>(key.hi) << 32);
  uint64_t yz_stride = static_cast<uint64_t>(mesh_y) * mesh_z;
  out_pmid[VectorOffset(output, 0)] =
      static_cast<int16_t>(raveled / yz_stride);
  uint64_t yz = raveled % yz_stride;
  out_pmid[VectorOffset(output, 1)] =
      static_cast<int16_t>(yz / static_cast<uint64_t>(mesh_z));
  out_pmid[VectorOffset(output, 2)] =
      static_cast<int16_t>(yz % static_cast<uint64_t>(mesh_z));
  for (int component = 0; component < 3; ++component) {
    out_disp[VectorOffset(output, component)] =
        RecordTraits<float>::Load(record + 2 + component);
    out_vel[VectorOffset(output, component)] =
        RecordTraits<float>::Load(record + 5 + component);
  }
  out_valid[output] = 1;
}

template <typename Real, typename Coord>
__device__ uint32_t StayPosition(const Coord* pmid,
                                 const int32_t* stay_indices,
                                 uint32_t stay_count,
                                 const uint32_t* left_records,
                                 uint32_t left_count,
                                 const uint32_t* right_records,
                                 uint32_t right_count, uint32_t rank,
                                 int mesh_x, int mesh_y, int mesh_z) {
  Key64 key = CompactStayKeyAt(pmid, stay_indices, rank, mesh_x, mesh_y,
                               mesh_z);
  return rank + LowerBoundRecordKeys<Real>(left_records, left_count, key) +
         LowerBoundRecordKeys<Real>(right_records, right_count, key);
}

template <typename Real, typename Coord>
__device__ uint32_t LeftPosition(const Coord* pmid,
                                 const int32_t* stay_indices,
                                 uint32_t stay_count,
                                const uint32_t* left_records,
                                uint32_t left_count,
                                const uint32_t* right_records,
                                uint32_t right_count, uint32_t rank,
                                int mesh_x, int mesh_y, int mesh_z) {
  Key64 key = RecordKeyAt<Real>(left_records, rank);
  return rank + UpperBoundCompactStay(pmid, stay_indices, stay_count, key,
                                      mesh_x, mesh_y, mesh_z) +
         LowerBoundRecordKeys<Real>(right_records, right_count, key);
}

template <typename Real, typename Coord>
__device__ uint32_t RightPosition(const Coord* pmid,
                                  const int32_t* stay_indices,
                                  uint32_t stay_count,
                                 const uint32_t* left_records,
                                 uint32_t left_count,
                                 const uint32_t* right_records,
                                 uint32_t right_count, uint32_t rank,
                                 int mesh_x, int mesh_y, int mesh_z) {
  Key64 key = RecordKeyAt<Real>(right_records, rank);
  return rank + UpperBoundCompactStay(pmid, stay_indices, stay_count, key,
                                      mesh_x, mesh_y, mesh_z) +
         UpperBoundRecordKeys<Real>(left_records, left_count, key);
}

// Each output thread computes the co-rank of its output diagonal in the three
// sorted streams.  The strict/lower and non-strict/upper bounds encode the
// canonical tie order stay < left < right without materializing a concatenated
// candidate array.
template <bool Metadata, typename Real, typename Coord>
__global__ void MergePathBidirKernel(
    const Coord* pmid, const Real* disp, const Real* vel,
    const int32_t* stay_indices,
    const uint32_t* left_records, const uint32_t* right_records,
    const int32_t* stay_count_value, const int32_t* left_count_value,
    const int32_t* right_count_value,
    int mesh_x, int mesh_y, int mesh_z, int capacity, Coord* out_pmid,
    Real* out_disp, Real* out_vel, uint8_t* out_valid, uint8_t* out_tag,
    int32_t* out_index, uint32_t* out_key) {
  int output = blockIdx.x * blockDim.x + threadIdx.x;
  if (output >= capacity) return;
  int stay_signed = *stay_count_value;
  int left_signed = *left_count_value;
  int right_signed = *right_count_value;
  uint32_t stay_count = stay_signed <= 0
                            ? 0u
                            : min(static_cast<uint32_t>(stay_signed),
                                  static_cast<uint32_t>(capacity));
  uint32_t left_count = left_signed <= 0
                            ? 0u
                            : min(static_cast<uint32_t>(left_signed),
                                  static_cast<uint32_t>(capacity));
  uint32_t right_count = right_signed <= 0
                             ? 0u
                             : min(static_cast<uint32_t>(right_signed),
                                   static_cast<uint32_t>(capacity));
  uint32_t total = stay_count + left_count + right_count;
  if (static_cast<uint32_t>(output) >= total) return;

  uint32_t source_rank = 0;
  uint8_t source_tag = 0;
  bool found = false;

  // Find a stay source on the output diagonal.
  uint32_t lo = 0;
  uint32_t hi = stay_count;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    uint32_t position = StayPosition<Real>(
        pmid, stay_indices, stay_count, left_records, left_count,
        right_records, right_count, mid, mesh_x, mesh_y, mesh_z);
    if (position < static_cast<uint32_t>(output)) lo = mid + 1;
    else hi = mid;
  }
  if (lo < stay_count &&
      StayPosition<Real>(pmid, stay_indices, stay_count, left_records,
                         left_count, right_records, right_count, lo, mesh_x,
                         mesh_y, mesh_z) ==
          static_cast<uint32_t>(output)) {
    source_rank = lo;
    source_tag = 0;
    found = true;
  }

  // If no stay source owns the diagonal, locate a left source.
  if (!found) {
    lo = 0;
    hi = left_count;
    while (lo < hi) {
      uint32_t mid = lo + (hi - lo) / 2;
      uint32_t position = LeftPosition<Real>(
          pmid, stay_indices, stay_count, left_records, left_count,
          right_records, right_count, mid, mesh_x, mesh_y, mesh_z);
      if (position < static_cast<uint32_t>(output)) lo = mid + 1;
      else hi = mid;
    }
    if (lo < left_count &&
        LeftPosition<Real>(pmid, stay_indices, stay_count, left_records,
                           left_count, right_records, right_count, lo, mesh_x,
                           mesh_y, mesh_z) ==
            static_cast<uint32_t>(output)) {
      source_rank = lo;
      source_tag = 1;
      found = true;
    }
  }

  // Remaining diagonals belong to the right stream.
  if (!found) {
    lo = 0;
    hi = right_count;
    while (lo < hi) {
      uint32_t mid = lo + (hi - lo) / 2;
      uint32_t position = RightPosition<Real>(
          pmid, stay_indices, stay_count, left_records, left_count,
          right_records, right_count, mid, mesh_x, mesh_y, mesh_z);
      if (position < static_cast<uint32_t>(output)) lo = mid + 1;
      else hi = mid;
    }
    if (lo >= right_count ||
        RightPosition<Real>(pmid, stay_indices, stay_count, left_records,
                            left_count, right_records, right_count, lo,
                            mesh_x, mesh_y, mesh_z) !=
            static_cast<uint32_t>(output)) {
      return;
    }
    source_rank = lo;
    source_tag = 2;
  }

  const uint32_t* record = nullptr;
  Key64 key{0u, 0u};
  if (source_tag == 0) {
    int source_row = stay_indices[source_rank];
    if (source_row < 0) return;
    key = CompactStayKeyAt(pmid, stay_indices, source_rank, mesh_x, mesh_y,
                           mesh_z);
    out_pmid[VectorOffset(output, 0)] = pmid[VectorOffset(source_row, 0)];
    out_pmid[VectorOffset(output, 1)] = pmid[VectorOffset(source_row, 1)];
    out_pmid[VectorOffset(output, 2)] = pmid[VectorOffset(source_row, 2)];
    for (int component = 0; component < 3; ++component) {
      out_disp[VectorOffset(output, component)] =
          disp[VectorOffset(source_row, component)];
      out_vel[VectorOffset(output, component)] =
          vel[VectorOffset(source_row, component)];
    }
    // The transpose path indexes the compact stay stream.  The original
    // authoritative slot is carried by stay_indices and is used by the
    // forward copy above; keep provenance indices in the existing compact
    // stream convention used by pmpp_route_merge_aux.
    if constexpr (Metadata) {
      out_index[output] = static_cast<int32_t>(source_rank);
    }
  } else {
    constexpr int value_words = RecordTraits<Real>::kValueWords;
    constexpr int record_words = RecordTraits<Real>::kRecordWords;
    record = source_tag == 1
                 ? left_records + RecordOffset(source_rank, record_words)
                 : right_records + RecordOffset(source_rank, record_words);
    key = LoadKey(record);
    uint64_t raveled = static_cast<uint64_t>(key.lo) |
                       (static_cast<uint64_t>(key.hi) << 32);
    uint64_t yz_stride = static_cast<uint64_t>(mesh_y) * mesh_z;
    out_pmid[VectorOffset(output, 0)] =
        static_cast<Coord>(raveled / yz_stride);
    uint64_t yz = raveled % yz_stride;
    out_pmid[VectorOffset(output, 1)] =
        static_cast<Coord>(yz / static_cast<uint64_t>(mesh_z));
    out_pmid[VectorOffset(output, 2)] =
        static_cast<Coord>(yz % static_cast<uint64_t>(mesh_z));
    for (int component = 0; component < 3; ++component) {
      out_disp[VectorOffset(output, component)] = RecordTraits<Real>::Load(
          record + 2 + component * value_words);
      out_vel[VectorOffset(output, component)] = RecordTraits<Real>::Load(
          record + 2 + 3 * value_words + component * value_words);
    }
    if constexpr (Metadata) {
      out_index[output] = static_cast<int32_t>(source_rank);
    }
  }
  if constexpr (Metadata) {
    StoreKey(out_key + RecordOffset(output, 2), key);
    out_tag[output] = source_tag;
  }
  out_valid[output] = 1;
}

__global__ void WriteBidirCountKernel(const int32_t* stay_count,
                                      const int32_t* left_count,
                                      const int32_t* right_count,
                                      int32_t* output) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int64_t total = static_cast<int64_t>(max(*stay_count, 0)) +
                  static_cast<int64_t>(max(*left_count, 0)) +
                  static_cast<int64_t>(max(*right_count, 0));
  *output = total > static_cast<int64_t>(INT32_MAX)
                ? INT32_MAX
                : static_cast<int32_t>(total);
}

template <typename Real>
__global__ void TransposeSplitKernel(const Real* merged, const uint8_t* tags,
                                     const int32_t* indices, int n,
                                     int payload_width, int auth_size,
                                     int share_capacity, Real* stay,
                                     Real* incoming_left,
                                     Real* incoming_right) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  int tag = static_cast<int>(tags[row]);
  int source = indices[row];
  if (source < 0) return;
  Real* destination = nullptr;
  int destination_size = 0;
  if (tag == 0) {
    destination = stay;
    destination_size = auth_size;
  } else if (tag == 1) {
    destination = incoming_left;
    destination_size = share_capacity;
  } else if (tag == 2) {
    destination = incoming_right;
    destination_size = share_capacity;
  }
  if (destination == nullptr || source >= destination_size) return;
  for (int component = 0; component < payload_width; ++component) {
    size_t destination_offset = static_cast<size_t>(source) * payload_width +
                                static_cast<size_t>(component);
    size_t source_offset = static_cast<size_t>(row) * payload_width +
                           static_cast<size_t>(component);
    destination[destination_offset] = merged[source_offset];
  }
}

template <typename Real>
__global__ void TransposeScatterKernel(const Real* source, const int32_t* pos,
                                       const uint8_t* valid, int rows,
                                       int payload_width, int auth_size,
                                       Real* output) {
  size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = static_cast<size_t>(rows) * payload_width;
  if (linear >= total) return;
  int row = static_cast<int>(linear / payload_width);
  int component = static_cast<int>(linear % payload_width);
  int destination = pos[row];
  if (valid[row] != 0 && destination >= 0 && destination < auth_size) {
    output[static_cast<size_t>(destination) * payload_width + component] =
        source[linear];
  }
}

}  // namespace

ffi::Error RouteOffsetProbeImpl(
    cudaStream_t stream, int32_t component, int32_t record_words,
    ffi::Buffer<ffi::U32> row_lo, ffi::Buffer<ffi::U32> row_hi,
    ffi::ResultBuffer<ffi::U32> output) {
  if (component < 0 || component >= 3 || record_words <= 0 ||
      row_lo.element_count() != 1 || row_hi.element_count() != 1 ||
      output->element_count() != 4) {
    return ffi::Error::InvalidArgument("invalid routing offset probe input");
  }
  OffsetProbeKernel<<<1, 1, 0, stream>>>(
      row_lo.typed_data(), row_hi.typed_data(), component, record_words,
      output->typed_data());
  if (cudaGetLastError() != cudaSuccess) {
    return ffi::Error::Internal("routing offset probe launch failed");
  }
  return ffi::Error::Success();
}

template <ffi::DataType DType>
ffi::Error RoutePackTypedImpl(
    cudaStream_t stream, ffi::ScratchAllocator scratch, int32_t global_nmesh,
    int32_t mesh_x, int32_t mesh_y, int32_t mesh_z, int32_t direction,
    int32_t num_devices, int32_t capacity, ffi::Buffer<ffi::S32> pmid,
    ffi::Buffer<DType> disp, ffi::Buffer<DType> vel,
    ffi::Buffer<ffi::U8> valid, ffi::Buffer<DType> x_mod,
    ffi::Buffer<ffi::S32> owned_start_buffer,
    ffi::Buffer<ffi::S32> owned_end_buffer,
    ffi::Buffer<ffi::S32> slice_width_buffer,
    ffi::ResultBuffer<ffi::U32> records, ffi::ResultBuffer<ffi::S32> count,
    ffi::ResultBuffer<ffi::U8> classes) {
  using Real = ffi::NativeType<DType>;
  static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>);
  constexpr int record_words = RecordTraits<Real>::kRecordWords;
  int n = static_cast<int>(pmid.element_count() / 3);
  if (n <= 0 || capacity < 0 || direction == 0 || num_devices < 2) {
    return ffi::Error::InvalidArgument("invalid route-pack shape or topology");
  }
  int num_blocks = (n + kThreads - 1) / kThreads;
  auto block_counts_mem = scratch.Allocate(sizeof(uint32_t) * num_blocks,
                                           alignof(uint32_t));
  if (!block_counts_mem) {
    return ffi::Error::Internal("unable to allocate route-pack block-count scratch");
  }
  auto block_offsets_mem = scratch.Allocate(sizeof(uint32_t) * num_blocks,
                                            alignof(uint32_t));
  if (!block_offsets_mem) {
    return ffi::Error::Internal("unable to allocate route-pack block-offset scratch");
  }
  auto* block_counts = static_cast<uint32_t*>(*block_counts_mem);
  auto* block_offsets = static_cast<uint32_t*>(*block_offsets_mem);

  cudaError_t status = cudaMemsetAsync(
      records->typed_data(), 0, sizeof(uint32_t) * capacity * record_words,
      stream);
  if (status != cudaSuccess) return ffi::Error::Internal("route-pack record clear failed");
  dim3 blocks((n + kThreads - 1) / kThreads);
  ClassifyKernel<Real><<<blocks, kThreads, 0, stream>>>(
      x_mod.typed_data(), valid.typed_data(), classes->typed_data(),
      block_counts, n, global_nmesh,
      owned_start_buffer.typed_data(), owned_end_buffer.typed_data(),
      slice_width_buffer.typed_data(), direction, num_devices);
  if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("route-pack classify launch failed");

  size_t temp_bytes = 0;
  status = cub::DeviceScan::ExclusiveSum(
      nullptr, temp_bytes, block_counts, block_offsets, num_blocks, stream);
  if (status != cudaSuccess) return ffi::Error::Internal("CUB route-pack scan query failed");
  auto temp_mem = scratch.Allocate(temp_bytes, 1);
  if (!temp_mem) {
    return ffi::Error::Internal("unable to allocate CUB route-pack workspace (bytes=" +
                                std::to_string(temp_bytes) + ")");
  }
  status = cub::DeviceScan::ExclusiveSum(
      *temp_mem, temp_bytes, block_counts, block_offsets, num_blocks, stream);
  if (status != cudaSuccess) return ffi::Error::Internal("CUB route-pack scan failed");

  WriteRecordsKernel<Real><<<blocks, kThreads, 0, stream>>>(
      pmid.typed_data(), disp.typed_data(), vel.typed_data(),
      classes->typed_data(), block_offsets, records->typed_data(), n, mesh_x,
      mesh_y, mesh_z, capacity, direction);
  if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("route-pack write launch failed");
  WriteCountKernel<<<1, 1, 0, stream>>>(block_counts, block_offsets,
                                        count->typed_data(), num_blocks);
  if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("route-pack count launch failed");
  return ffi::Error::Success();
}

template <ffi::DataType DType, ffi::DataType CoordDType = ffi::S32>
ffi::Error RouteBidirPackTypedImpl(
    cudaStream_t stream, ffi::ScratchAllocator scratch, int32_t global_nmesh,
    int32_t mesh_x, int32_t mesh_y, int32_t mesh_z, int32_t slice_width,
    int32_t num_devices, int32_t capacity, int32_t stay_capacity,
    ffi::Buffer<CoordDType> pmid,
    ffi::Buffer<DType> disp, ffi::Buffer<DType> vel,
    ffi::Buffer<ffi::U8> valid, ffi::Buffer<DType> x_mod,
    ffi::Buffer<ffi::S32> owned_start,
    ffi::Buffer<ffi::S32> owned_end,
    ffi::ResultBuffer<ffi::U32> left_records,
    ffi::ResultBuffer<ffi::U32> right_records,
    ffi::ResultBuffer<ffi::S32> left_count,
    ffi::ResultBuffer<ffi::S32> right_count,
    ffi::ResultBuffer<ffi::U8> classes,
    ffi::ResultBuffer<ffi::S32> stay_indices,
    ffi::ResultBuffer<ffi::S32> stay_count) {
  using Real = ffi::NativeType<DType>;
  using Coord = ffi::NativeType<CoordDType>;
  static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>);
  static_assert(std::is_same_v<Coord, int16_t> || std::is_same_v<Coord, int32_t>);
  constexpr int record_words = RecordTraits<Real>::kRecordWords;
  int n = static_cast<int>(pmid.element_count() / 3);
  if (n <= 0 || global_nmesh <= 0 || capacity < 0 || stay_capacity < 0 ||
      num_devices < 2) {
    return ffi::Error::InvalidArgument("invalid bidirectional route-pack shape or topology");
  }
  int num_blocks = (n + kThreads - 1) / kThreads;
  auto left_counts_mem = scratch.Allocate(sizeof(uint32_t) * num_blocks,
                                          alignof(uint32_t));
  auto right_counts_mem = scratch.Allocate(sizeof(uint32_t) * num_blocks,
                                           alignof(uint32_t));
  auto stay_counts_mem = scratch.Allocate(sizeof(uint32_t) * num_blocks,
                                          alignof(uint32_t));
  auto left_offsets_mem = scratch.Allocate(sizeof(uint32_t) * num_blocks,
                                           alignof(uint32_t));
  auto right_offsets_mem = scratch.Allocate(sizeof(uint32_t) * num_blocks,
                                            alignof(uint32_t));
  auto stay_offsets_mem = scratch.Allocate(sizeof(uint32_t) * num_blocks,
                                           alignof(uint32_t));
  if (!left_counts_mem || !right_counts_mem || !stay_counts_mem ||
      !left_offsets_mem || !right_offsets_mem || !stay_offsets_mem) {
    return ffi::Error::Internal("unable to allocate bidirectional route metadata");
  }
  auto* left_counts = static_cast<uint32_t*>(*left_counts_mem);
  auto* right_counts = static_cast<uint32_t*>(*right_counts_mem);
  auto* stay_counts = static_cast<uint32_t*>(*stay_counts_mem);
  auto* left_offsets = static_cast<uint32_t*>(*left_offsets_mem);
  auto* right_offsets = static_cast<uint32_t*>(*right_offsets_mem);
  auto* stay_offsets = static_cast<uint32_t*>(*stay_offsets_mem);

  dim3 blocks(num_blocks);
  ClassifyBidirKernel<Real><<<blocks, kThreads, 0, stream>>>(
      x_mod.typed_data(), valid.typed_data(), classes->typed_data(),
      left_counts, right_counts, stay_counts, n, global_nmesh,
      owned_start.typed_data(), owned_end.typed_data(), slice_width,
      num_devices);
  if (cudaGetLastError() != cudaSuccess) {
    return ffi::Error::Internal("bidirectional route classification launch failed");
  }

  size_t temp_bytes = 0;
  cudaError_t status = cub::DeviceScan::ExclusiveSum(
      nullptr, temp_bytes, left_counts, left_offsets, num_blocks, stream);
  if (status != cudaSuccess) {
    return ffi::Error::Internal("CUB bidirectional route scan query failed");
  }
  auto temp_mem = scratch.Allocate(temp_bytes, 1);
  if (!temp_mem) {
    return ffi::Error::Internal("unable to allocate CUB bidirectional route workspace");
  }
  auto scan = [&](const uint32_t* counts, uint32_t* offsets) -> cudaError_t {
    return cub::DeviceScan::ExclusiveSum(
        *temp_mem, temp_bytes, counts, offsets, num_blocks, stream);
  };
  status = scan(left_counts, left_offsets);
  if (status != cudaSuccess) return ffi::Error::Internal("CUB left route scan failed");
  status = scan(right_counts, right_offsets);
  if (status != cudaSuccess) return ffi::Error::Internal("CUB right route scan failed");
  status = scan(stay_counts, stay_offsets);
  if (status != cudaSuccess) return ffi::Error::Internal("CUB stay route scan failed");

  if (capacity > 0) {
    status = cudaMemsetAsync(left_records->typed_data(), 0,
                             sizeof(uint32_t) * capacity * record_words, stream);
    if (status != cudaSuccess) return ffi::Error::Internal("left route record clear failed");
    status = cudaMemsetAsync(right_records->typed_data(), 0,
                             sizeof(uint32_t) * capacity * record_words, stream);
    if (status != cudaSuccess) return ffi::Error::Internal("right route record clear failed");
  }
  if (stay_capacity > 0) {
    status = cudaMemsetAsync(stay_indices->typed_data(), 0,
                             sizeof(int32_t) * stay_capacity, stream);
    if (status != cudaSuccess) return ffi::Error::Internal("stay index clear failed");
  }
  blocks = dim3(num_blocks);
  WriteBidirRecordsKernel<Real, Coord><<<blocks, kThreads, 0, stream>>>(
      pmid.typed_data(), disp.typed_data(), vel.typed_data(),
      classes->typed_data(), left_offsets, right_offsets, stay_offsets,
      left_records->typed_data(), right_records->typed_data(),
      stay_indices->typed_data(), n, mesh_x, mesh_y, mesh_z, capacity,
      stay_capacity);
  if (cudaGetLastError() != cudaSuccess) {
    return ffi::Error::Internal("bidirectional route write launch failed");
  }
  WriteCountKernel<<<1, 1, 0, stream>>>(left_counts, left_offsets,
                                        left_count->typed_data(), num_blocks);
  WriteCountKernel<<<1, 1, 0, stream>>>(right_counts, right_offsets,
                                        right_count->typed_data(), num_blocks);
  WriteCountKernel<<<1, 1, 0, stream>>>(stay_counts, stay_offsets,
                                        stay_count->typed_data(), num_blocks);
  if (cudaGetLastError() != cudaSuccess) {
    return ffi::Error::Internal("bidirectional route count launch failed");
  }
  return ffi::Error::Success();
}

template <ffi::DataType DType, ffi::DataType CoordDType = ffi::S32>
ffi::Error RouteMergeBidirTypedImpl(
    cudaStream_t stream, int32_t mesh_x, int32_t mesh_y, int32_t mesh_z,
    int32_t capacity, ffi::Buffer<CoordDType> pmid,
    ffi::Buffer<DType> disp, ffi::Buffer<DType> vel,
    ffi::Buffer<ffi::S32> stay_indices,
    ffi::Buffer<ffi::S32> stay_count,
    ffi::Buffer<ffi::U32> left_records, ffi::Buffer<ffi::S32> left_count,
    ffi::Buffer<ffi::U32> right_records, ffi::Buffer<ffi::S32> right_count,
    ffi::ResultBuffer<CoordDType> out_pmid,
    ffi::ResultBuffer<DType> out_disp,
    ffi::ResultBuffer<DType> out_vel,
    ffi::ResultBuffer<ffi::U8> out_valid,
    ffi::ResultBuffer<ffi::U8> out_tag,
    ffi::ResultBuffer<ffi::S32> out_index,
    ffi::ResultBuffer<ffi::U32> out_key,
    ffi::ResultBuffer<ffi::S32> out_count) {
  using Real = ffi::NativeType<DType>;
  using Coord = ffi::NativeType<CoordDType>;
  static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>);
  static_assert(std::is_same_v<Coord, int16_t> || std::is_same_v<Coord, int32_t>);
  int n = static_cast<int>(pmid.element_count() / 3);
  if (n <= 0 || capacity < 0 || mesh_x <= 0 || mesh_y <= 0 || mesh_z <= 0) {
    return ffi::Error::InvalidArgument("invalid bidirectional route-merge shape");
  }
  cudaError_t status = cudaSuccess;
  if (capacity > 0) {
    status = cudaMemsetAsync(out_pmid->typed_data(), 0,
                             sizeof(Coord) * capacity * 3, stream);
    if (status != cudaSuccess) return ffi::Error::Internal("bidir merge pmid clear failed");
    status = cudaMemsetAsync(out_disp->typed_data(), 0,
                             sizeof(Real) * capacity * 3, stream);
    if (status != cudaSuccess) return ffi::Error::Internal("bidir merge displacement clear failed");
    status = cudaMemsetAsync(out_vel->typed_data(), 0,
                             sizeof(Real) * capacity * 3, stream);
    if (status != cudaSuccess) return ffi::Error::Internal("bidir merge velocity clear failed");
    status = cudaMemsetAsync(out_valid->typed_data(), 0,
                             sizeof(uint8_t) * capacity, stream);
    if (status != cudaSuccess) return ffi::Error::Internal("bidir merge validity clear failed");
    status = cudaMemsetAsync(out_tag->typed_data(), 0,
                             sizeof(uint8_t) * capacity, stream);
    if (status != cudaSuccess) return ffi::Error::Internal("bidir merge tag clear failed");
    status = cudaMemsetAsync(out_index->typed_data(), 0xff,
                             sizeof(int32_t) * capacity, stream);
    if (status != cudaSuccess) return ffi::Error::Internal("bidir merge index clear failed");
    status = cudaMemsetAsync(out_key->typed_data(), 0,
                             sizeof(uint32_t) * capacity * 2, stream);
    if (status != cudaSuccess) return ffi::Error::Internal("bidir merge key clear failed");

    // Counts are device scalars.  The kernel clamps them to the fixed-
    // capacity payloads; the uncapped total is written separately below by a
    // tiny kernel. Avoiding a host read keeps the FFI handler asynchronous.
    dim3 blocks((capacity + kThreads - 1) / kThreads);
    MergePathBidirKernel<true, Real, Coord><<<blocks, kThreads, 0, stream>>>(
        pmid.typed_data(), disp.typed_data(), vel.typed_data(),
        stay_indices.typed_data(), left_records.typed_data(),
        right_records.typed_data(), stay_count.typed_data(), left_count.typed_data(),
        right_count.typed_data(), mesh_x, mesh_y, mesh_z, capacity,
        out_pmid->typed_data(), out_disp->typed_data(), out_vel->typed_data(),
        out_valid->typed_data(), out_tag->typed_data(), out_index->typed_data(),
        out_key->typed_data());
    if (cudaGetLastError() != cudaSuccess) {
      return ffi::Error::Internal("bidir merge launch failed");
    }
  }
  WriteBidirCountKernel<<<1, 1, 0, stream>>>(
      stay_count.typed_data(), left_count.typed_data(), right_count.typed_data(),
      out_count->typed_data());
  if (cudaGetLastError() != cudaSuccess) {
    return ffi::Error::Internal("bidir merge count launch failed");
  }
  return ffi::Error::Success();
}

ffi::Error RouteMergeBidirPrimalI16Impl(
    cudaStream_t stream, int32_t mesh_x, int32_t mesh_y, int32_t mesh_z,
    int32_t capacity, ffi::Buffer<ffi::S16> pmid,
    ffi::Buffer<ffi::F32> disp, ffi::Buffer<ffi::F32> vel,
    ffi::Buffer<ffi::S32> stay_indices,
    ffi::Buffer<ffi::S32> stay_count,
    ffi::Buffer<ffi::U32> left_records,
    ffi::Buffer<ffi::S32> left_count,
    ffi::Buffer<ffi::U32> right_records,
    ffi::Buffer<ffi::S32> right_count,
    ffi::ResultBuffer<ffi::S16> out_pmid,
    ffi::ResultBuffer<ffi::F32> out_disp,
    ffi::ResultBuffer<ffi::F32> out_vel,
    ffi::ResultBuffer<ffi::U8> out_valid,
    ffi::ResultBuffer<ffi::S32> out_count) {
  int n = static_cast<int>(pmid.element_count() / 3);
  if (n <= 0 || capacity < 0 || mesh_x <= 0 || mesh_y <= 0 || mesh_z <= 0) {
    return ffi::Error::InvalidArgument(
        "invalid primal bidirectional route-merge shape");
  }
  cudaError_t status = cudaSuccess;
  if (capacity > 0) {
    status = cudaMemsetAsync(out_pmid->typed_data(), 0,
                             sizeof(int16_t) * capacity * 3, stream);
    if (status != cudaSuccess) {
      return ffi::Error::Internal("primal bidir merge pmid clear failed");
    }
    status = cudaMemsetAsync(out_disp->typed_data(), 0,
                             sizeof(float) * capacity * 3, stream);
    if (status != cudaSuccess) {
      return ffi::Error::Internal(
          "primal bidir merge displacement clear failed");
    }
    status = cudaMemsetAsync(out_vel->typed_data(), 0,
                             sizeof(float) * capacity * 3, stream);
    if (status != cudaSuccess) {
      return ffi::Error::Internal("primal bidir merge velocity clear failed");
    }
    status = cudaMemsetAsync(out_valid->typed_data(), 0,
                             sizeof(uint8_t) * capacity, stream);
    if (status != cudaSuccess) {
      return ffi::Error::Internal("primal bidir merge validity clear failed");
    }

    dim3 blocks((capacity + kThreads - 1) / kThreads);
    MergePathBidirKernel<false, float, int16_t>
        <<<blocks, kThreads, 0, stream>>>(
            pmid.typed_data(), disp.typed_data(), vel.typed_data(),
            stay_indices.typed_data(), left_records.typed_data(),
            right_records.typed_data(), stay_count.typed_data(),
            left_count.typed_data(), right_count.typed_data(), mesh_x, mesh_y,
            mesh_z, capacity, out_pmid->typed_data(), out_disp->typed_data(),
            out_vel->typed_data(), out_valid->typed_data(), nullptr, nullptr,
            nullptr);
    if (cudaGetLastError() != cudaSuccess) {
      return ffi::Error::Internal("primal bidir merge launch failed");
    }
  }
  WriteBidirCountKernel<<<1, 1, 0, stream>>>(
      stay_count.typed_data(), left_count.typed_data(),
      right_count.typed_data(), out_count->typed_data());
  if (cudaGetLastError() != cudaSuccess) {
    return ffi::Error::Internal("primal bidir merge count launch failed");
  }
  return ffi::Error::Success();
}

ffi::Error RouteBidirDriftPackPrimalI16Impl(
    cudaStream_t stream, ffi::ScratchAllocator scratch,
    int32_t global_nmesh, int32_t mesh_x, int32_t mesh_y, int32_t mesh_z,
    int32_t slice_width, int32_t num_devices, int32_t capacity,
    ffi::Buffer<ffi::S16> pmid, ffi::Buffer<ffi::F32> disp,
    ffi::Buffer<ffi::F32> vel, ffi::Buffer<ffi::PRED> valid,
    ffi::Buffer<ffi::F32> drift_factor,
    ffi::Buffer<ffi::F32> disp_size,
    ffi::Buffer<ffi::S32> owned_start,
    ffi::Buffer<ffi::S32> owned_end,
    ffi::ResultBuffer<ffi::U32> left_records,
    ffi::ResultBuffer<ffi::U32> right_records,
    ffi::ResultBuffer<ffi::S32> left_count,
    ffi::ResultBuffer<ffi::S32> right_count,
    ffi::ResultBuffer<ffi::U32> stay_block_counts,
    ffi::ResultBuffer<ffi::S32> stay_count,
    ffi::ResultBuffer<ffi::S32> invalid_count) {
  constexpr int record_words = RecordTraits<float>::kRecordWords;
  int n = static_cast<int>(pmid.element_count() / 3);
  int num_blocks = (n + kThreads - 1) / kThreads;
  if (n <= 0 || global_nmesh <= 0 || capacity < 0 || num_devices < 2 ||
      mesh_x <= 0 || mesh_y <= 0 || mesh_z <= 0 || mesh_x > 32768 ||
      mesh_y > 32768 || mesh_z > 32768 ||
      drift_factor.element_count() != 1 || disp_size.element_count() != 1 ||
      stay_block_counts->element_count() != num_blocks) {
    return ffi::Error::InvalidArgument(
        "invalid fused-drift bidirectional route-pack shape or topology");
  }

  auto left_counts_mem = scratch.Allocate(sizeof(uint32_t) * num_blocks,
                                          alignof(uint32_t));
  auto right_counts_mem = scratch.Allocate(sizeof(uint32_t) * num_blocks,
                                           alignof(uint32_t));
  auto invalid_counts_mem = scratch.Allocate(sizeof(uint32_t) * num_blocks,
                                             alignof(uint32_t));
  auto left_offsets_mem = scratch.Allocate(sizeof(uint32_t) * num_blocks,
                                           alignof(uint32_t));
  auto right_offsets_mem = scratch.Allocate(sizeof(uint32_t) * num_blocks,
                                            alignof(uint32_t));
  auto stay_offsets_mem = scratch.Allocate(sizeof(uint32_t) * num_blocks,
                                           alignof(uint32_t));
  auto invalid_offsets_mem = scratch.Allocate(sizeof(uint32_t) * num_blocks,
                                              alignof(uint32_t));
  if (!left_counts_mem || !right_counts_mem || !invalid_counts_mem ||
      !left_offsets_mem || !right_offsets_mem || !stay_offsets_mem ||
      !invalid_offsets_mem) {
    return ffi::Error::Internal(
        "unable to allocate fused-drift route block metadata");
  }
  auto* left_counts = static_cast<uint32_t*>(*left_counts_mem);
  auto* right_counts = static_cast<uint32_t*>(*right_counts_mem);
  auto* invalid_counts = static_cast<uint32_t*>(*invalid_counts_mem);
  auto* left_offsets = static_cast<uint32_t*>(*left_offsets_mem);
  auto* right_offsets = static_cast<uint32_t*>(*right_offsets_mem);
  auto* stay_offsets = static_cast<uint32_t*>(*stay_offsets_mem);
  auto* invalid_offsets = static_cast<uint32_t*>(*invalid_offsets_mem);

  cudaError_t status = cudaSuccess;
  if (capacity > 0) {
    status = cudaMemsetAsync(left_records->typed_data(), 0,
                             sizeof(uint32_t) * capacity * record_words,
                             stream);
    if (status != cudaSuccess) {
      return ffi::Error::Internal("fused-drift left record clear failed");
    }
    status = cudaMemsetAsync(right_records->typed_data(), 0,
                             sizeof(uint32_t) * capacity * record_words,
                             stream);
    if (status != cudaSuccess) {
      return ffi::Error::Internal("fused-drift right record clear failed");
    }
  }
  dim3 blocks(num_blocks);
  ClassifyDriftBidirCountsKernel<<<blocks, kThreads, 0, stream>>>(
      pmid.typed_data(), disp.typed_data(), vel.typed_data(),
      valid.typed_data(), left_counts, right_counts,
      stay_block_counts->typed_data(), invalid_counts, n, global_nmesh,
      owned_start.typed_data(), owned_end.typed_data(), slice_width,
      num_devices, drift_factor.typed_data(), disp_size.typed_data());
  if (cudaGetLastError() != cudaSuccess) {
    return ffi::Error::Internal(
        "fused-drift route classification launch failed");
  }

  size_t temp_bytes = 0;
  status = cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, left_counts,
                                         left_offsets, num_blocks, stream);
  if (status != cudaSuccess) {
    return ffi::Error::Internal("CUB fused-drift route scan query failed");
  }
  auto temp_mem = scratch.Allocate(temp_bytes, 1);
  if (!temp_mem) {
    return ffi::Error::Internal(
        "unable to allocate CUB fused-drift route workspace");
  }
  auto scan = [&](const uint32_t* counts, uint32_t* offsets) -> cudaError_t {
    return cub::DeviceScan::ExclusiveSum(*temp_mem, temp_bytes, counts,
                                         offsets, num_blocks, stream);
  };
  status = scan(left_counts, left_offsets);
  if (status != cudaSuccess) {
    return ffi::Error::Internal("CUB fused-drift left scan failed");
  }
  status = scan(right_counts, right_offsets);
  if (status != cudaSuccess) {
    return ffi::Error::Internal("CUB fused-drift right scan failed");
  }
  status = scan(stay_block_counts->typed_data(), stay_offsets);
  if (status != cudaSuccess) {
    return ffi::Error::Internal("CUB fused-drift stay scan failed");
  }
  status = scan(invalid_counts, invalid_offsets);
  if (status != cudaSuccess) {
    return ffi::Error::Internal("CUB fused-drift invalid scan failed");
  }

  WriteDriftBidirRecordsKernelI16<<<blocks, kThreads, 0, stream>>>(
      pmid.typed_data(), disp.typed_data(), vel.typed_data(),
      valid.typed_data(), left_offsets, right_offsets,
      left_records->typed_data(), right_records->typed_data(), n,
      global_nmesh, owned_start.typed_data(), owned_end.typed_data(),
      slice_width, num_devices, drift_factor.typed_data(),
      disp_size.typed_data(), mesh_x, mesh_y, mesh_z, capacity);
  if (cudaGetLastError() != cudaSuccess) {
    return ffi::Error::Internal("fused-drift route write launch failed");
  }
  WriteCountKernel<<<1, 1, 0, stream>>>(left_counts, left_offsets,
                                        left_count->typed_data(), num_blocks);
  WriteCountKernel<<<1, 1, 0, stream>>>(right_counts, right_offsets,
                                        right_count->typed_data(), num_blocks);
  WriteCountKernel<<<1, 1, 0, stream>>>(
      stay_block_counts->typed_data(), stay_offsets,
      stay_count->typed_data(), num_blocks);
  WriteCountKernel<<<1, 1, 0, stream>>>(invalid_counts, invalid_offsets,
                                        invalid_count->typed_data(),
                                        num_blocks);
  if (cudaGetLastError() != cudaSuccess) {
    return ffi::Error::Internal("fused-drift route count launch failed");
  }
  return ffi::Error::Success();
}

ffi::Error RouteBidirDriftMergePrimalI16Impl(
    cudaStream_t stream, ffi::ScratchAllocator scratch,
    int32_t global_nmesh, int32_t mesh_x, int32_t mesh_y, int32_t mesh_z,
    int32_t slice_width, int32_t num_devices, int32_t record_capacity,
    int32_t capacity, ffi::Buffer<ffi::S16> pmid,
    ffi::Buffer<ffi::F32> disp, ffi::Buffer<ffi::F32> vel,
    ffi::Buffer<ffi::PRED> valid, ffi::Buffer<ffi::F32> drift_factor,
    ffi::Buffer<ffi::F32> disp_size,
    ffi::Buffer<ffi::S32> owned_start,
    ffi::Buffer<ffi::S32> owned_end,
    ffi::Buffer<ffi::U32> stay_block_counts,
    ffi::Buffer<ffi::S32> stay_count,
    ffi::Buffer<ffi::U32> left_records,
    ffi::Buffer<ffi::S32> left_count,
    ffi::Buffer<ffi::U32> right_records,
    ffi::Buffer<ffi::S32> right_count,
    ffi::ResultBuffer<ffi::S16> out_pmid,
    ffi::ResultBuffer<ffi::F32> out_disp,
    ffi::ResultBuffer<ffi::F32> out_vel,
    ffi::ResultBuffer<ffi::PRED> out_valid,
    ffi::ResultBuffer<ffi::S32> out_count) {
  int n = static_cast<int>(pmid.element_count() / 3);
  int num_blocks = (n + kThreads - 1) / kThreads;
  if (n <= 0 || capacity < 0 || record_capacity < 0 ||
      global_nmesh <= 0 || num_devices < 2 || mesh_x <= 0 || mesh_y <= 0 ||
      mesh_z <= 0 || mesh_x > 32768 || mesh_y > 32768 || mesh_z > 32768 ||
      drift_factor.element_count() != 1 || disp_size.element_count() != 1 ||
      stay_block_counts.element_count() != num_blocks) {
    return ffi::Error::InvalidArgument(
        "invalid fused-drift bidirectional route-merge shape or topology");
  }
  auto stay_offsets_mem = scratch.Allocate(sizeof(uint32_t) * num_blocks,
                                           alignof(uint32_t));
  if (!stay_offsets_mem) {
    return ffi::Error::Internal(
        "unable to allocate fused-drift stay block offsets");
  }
  auto* stay_offsets = static_cast<uint32_t*>(*stay_offsets_mem);
  size_t temp_bytes = 0;
  cudaError_t status = cub::DeviceScan::ExclusiveSum(
      nullptr, temp_bytes, stay_block_counts.typed_data(), stay_offsets,
      num_blocks, stream);
  if (status != cudaSuccess) {
    return ffi::Error::Internal(
        "CUB fused-drift merge scan query failed");
  }
  auto temp_mem = scratch.Allocate(temp_bytes, 1);
  if (!temp_mem) {
    return ffi::Error::Internal(
        "unable to allocate CUB fused-drift merge workspace");
  }
  status = cub::DeviceScan::ExclusiveSum(
      *temp_mem, temp_bytes, stay_block_counts.typed_data(), stay_offsets,
      num_blocks, stream);
  if (status != cudaSuccess) {
    return ffi::Error::Internal("CUB fused-drift merge scan failed");
  }

  if (capacity > 0) {
    status = cudaMemsetAsync(out_pmid->typed_data(), 0,
                             sizeof(int16_t) * capacity * 3, stream);
    if (status != cudaSuccess) {
      return ffi::Error::Internal("fused-drift merge pmid clear failed");
    }
    status = cudaMemsetAsync(out_disp->typed_data(), 0,
                             sizeof(float) * capacity * 3, stream);
    if (status != cudaSuccess) {
      return ffi::Error::Internal("fused-drift merge disp clear failed");
    }
    status = cudaMemsetAsync(out_vel->typed_data(), 0,
                             sizeof(float) * capacity * 3, stream);
    if (status != cudaSuccess) {
      return ffi::Error::Internal("fused-drift merge vel clear failed");
    }
    status = cudaMemsetAsync(out_valid->typed_data(), 0,
                             sizeof(bool) * capacity, stream);
    if (status != cudaSuccess) {
      return ffi::Error::Internal("fused-drift merge valid clear failed");
    }
  }

  dim3 blocks(num_blocks);
  ScatterDriftStayKernelI16<<<blocks, kThreads, 0, stream>>>(
      pmid.typed_data(), disp.typed_data(), vel.typed_data(),
      valid.typed_data(), stay_block_counts.typed_data(), stay_offsets,
      left_records.typed_data(), left_count.typed_data(),
      right_records.typed_data(), right_count.typed_data(), n, global_nmesh,
      owned_start.typed_data(), owned_end.typed_data(), slice_width,
      num_devices, drift_factor.typed_data(), disp_size.typed_data(), mesh_x,
      mesh_y, mesh_z, record_capacity, capacity, out_pmid->typed_data(),
      out_disp->typed_data(), out_vel->typed_data(), out_valid->typed_data());
  if (cudaGetLastError() != cudaSuccess) {
    return ffi::Error::Internal("fused-drift stay scatter launch failed");
  }
  if (record_capacity > 0) {
    blocks = dim3((record_capacity + kThreads - 1) / kThreads);
    ScatterDriftIncomingKernelI16<false><<<blocks, kThreads, 0, stream>>>(
        pmid.typed_data(), disp.typed_data(), vel.typed_data(),
        valid.typed_data(), stay_block_counts.typed_data(), stay_offsets,
        left_records.typed_data(), left_count.typed_data(),
        right_records.typed_data(), right_count.typed_data(), n,
        global_nmesh, owned_start.typed_data(), owned_end.typed_data(),
        slice_width, num_devices, drift_factor.typed_data(),
        disp_size.typed_data(), mesh_x, mesh_y, mesh_z, record_capacity,
        capacity, out_pmid->typed_data(), out_disp->typed_data(),
        out_vel->typed_data(), out_valid->typed_data());
    ScatterDriftIncomingKernelI16<true><<<blocks, kThreads, 0, stream>>>(
        pmid.typed_data(), disp.typed_data(), vel.typed_data(),
        valid.typed_data(), stay_block_counts.typed_data(), stay_offsets,
        left_records.typed_data(), left_count.typed_data(),
        right_records.typed_data(), right_count.typed_data(), n,
        global_nmesh, owned_start.typed_data(), owned_end.typed_data(),
        slice_width, num_devices, drift_factor.typed_data(),
        disp_size.typed_data(), mesh_x, mesh_y, mesh_z, record_capacity,
        capacity, out_pmid->typed_data(), out_disp->typed_data(),
        out_vel->typed_data(), out_valid->typed_data());
    if (cudaGetLastError() != cudaSuccess) {
      return ffi::Error::Internal(
          "fused-drift incoming scatter launch failed");
    }
  }
  WriteBidirCountKernel<<<1, 1, 0, stream>>>(
      stay_count.typed_data(), left_count.typed_data(),
      right_count.typed_data(), out_count->typed_data());
  if (cudaGetLastError() != cudaSuccess) {
    return ffi::Error::Internal("fused-drift merge count launch failed");
  }
  return ffi::Error::Success();
}

template <ffi::DataType DType>
ffi::Error RouteMergeTypedImpl(
    cudaStream_t stream, ffi::ScratchAllocator scratch, int32_t mesh_x,
    int32_t mesh_y, int32_t mesh_z, int32_t capacity,
    ffi::Buffer<ffi::S32> pmid, ffi::Buffer<DType> disp,
    ffi::Buffer<DType> vel, ffi::Buffer<ffi::U8> stay,
    ffi::Buffer<ffi::U32> incoming_records,
    ffi::Buffer<ffi::S32> incoming_count, ffi::ResultBuffer<ffi::S32> out_pmid,
    ffi::ResultBuffer<DType> out_disp, ffi::ResultBuffer<DType> out_vel,
    ffi::ResultBuffer<ffi::U8> out_valid) {
  return LaunchMerge<false, DType>(
      stream, std::move(scratch), pmid, disp, vel, stay, incoming_records,
      incoming_count, out_pmid, out_disp, out_vel, out_valid, nullptr, nullptr,
      mesh_x, mesh_y, mesh_z, capacity);
}

template <ffi::DataType DType>
ffi::Error RouteMergeAuxTypedImpl(
    cudaStream_t stream, ffi::ScratchAllocator scratch, int32_t mesh_x,
    int32_t mesh_y, int32_t mesh_z, int32_t capacity,
    ffi::Buffer<ffi::S32> pmid, ffi::Buffer<DType> disp,
    ffi::Buffer<DType> vel, ffi::Buffer<ffi::U8> stay,
    ffi::Buffer<ffi::U32> incoming_records,
    ffi::Buffer<ffi::S32> incoming_count, ffi::ResultBuffer<ffi::S32> out_pmid,
    ffi::ResultBuffer<DType> out_disp, ffi::ResultBuffer<DType> out_vel,
    ffi::ResultBuffer<ffi::U8> out_valid,
    ffi::ResultBuffer<ffi::U8> out_tag,
    ffi::ResultBuffer<ffi::S32> out_index) {
  return LaunchMerge<true, DType>(
      stream, std::move(scratch), pmid, disp, vel, stay, incoming_records,
      incoming_count, out_pmid, out_disp, out_vel, out_valid, &out_tag,
      &out_index, mesh_x, mesh_y, mesh_z, capacity);
}

template <ffi::DataType DType>
ffi::Error RouteTransposeSplitTypedImpl(
    cudaStream_t stream, int32_t auth_size, int32_t share_capacity,
    ffi::Buffer<DType> merged_cot, ffi::Buffer<ffi::U8> source_tag,
    ffi::Buffer<ffi::S32> source_idx,
    ffi::ResultBuffer<DType> stay_cot,
    ffi::ResultBuffer<DType> incoming_left_cot,
    ffi::ResultBuffer<DType> incoming_right_cot) {
  using Real = ffi::NativeType<DType>;
  static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>);
  int n = static_cast<int>(source_tag.element_count());
  if (n <= 0 || auth_size < 0 || share_capacity < 0 ||
      merged_cot.element_count() % n != 0) {
    return ffi::Error::InvalidArgument("invalid route-transpose split shape");
  }
  int payload_width = static_cast<int>(merged_cot.element_count() / n);
  cudaError_t status = cudaMemsetAsync(
      stay_cot->typed_data(), 0,
      sizeof(Real) * auth_size * payload_width, stream);
  if (status != cudaSuccess) return ffi::Error::Internal("route-transpose stay clear failed");
  status = cudaMemsetAsync(
      incoming_left_cot->typed_data(), 0,
      sizeof(Real) * share_capacity * payload_width, stream);
  if (status != cudaSuccess) return ffi::Error::Internal("route-transpose left clear failed");
  status = cudaMemsetAsync(
      incoming_right_cot->typed_data(), 0,
      sizeof(Real) * share_capacity * payload_width, stream);
  if (status != cudaSuccess) return ffi::Error::Internal("route-transpose right clear failed");
  dim3 blocks((n + kThreads - 1) / kThreads);
  TransposeSplitKernel<Real><<<blocks, kThreads, 0, stream>>>(
      merged_cot.typed_data(), source_tag.typed_data(), source_idx.typed_data(),
      n, payload_width, auth_size, share_capacity, stay_cot->typed_data(),
      incoming_left_cot->typed_data(), incoming_right_cot->typed_data());
  if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("route-transpose split launch failed");
  return ffi::Error::Success();
}

template <ffi::DataType DType>
ffi::Error RouteTransposeScatterTypedImpl(
    cudaStream_t stream, int32_t auth_size, int32_t share_capacity,
    ffi::Buffer<DType> stay_cot, ffi::Buffer<DType> send_left_cot,
    ffi::Buffer<DType> send_right_cot, ffi::Buffer<ffi::S32> stay_pos,
    ffi::Buffer<ffi::U8> stay_valid, ffi::Buffer<ffi::S32> send_left_pos,
    ffi::Buffer<ffi::U8> send_left_valid,
    ffi::Buffer<ffi::S32> send_right_pos,
    ffi::Buffer<ffi::U8> send_right_valid,
    ffi::ResultBuffer<DType> output) {
  using Real = ffi::NativeType<DType>;
  static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>);
  if (auth_size < 0 || share_capacity < 0 ||
      stay_cot.element_count() % std::max(auth_size, 1) != 0) {
    return ffi::Error::InvalidArgument("invalid route-transpose scatter shape");
  }
  int payload_width = auth_size == 0
                          ? 0
                          : static_cast<int>(stay_cot.element_count() / auth_size);
  cudaError_t status = cudaMemsetAsync(
      output->typed_data(), 0,
      sizeof(Real) * auth_size * payload_width, stream);
  if (status != cudaSuccess) return ffi::Error::Internal("route-transpose output clear failed");
  if (payload_width == 0) return ffi::Error::Success();
  auto launch = [&](ffi::Buffer<DType> source, ffi::Buffer<ffi::S32> pos,
                    ffi::Buffer<ffi::U8> valid, int rows) -> cudaError_t {
    size_t total = static_cast<size_t>(rows) * payload_width;
    dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads));
    TransposeScatterKernel<Real><<<blocks, kThreads, 0, stream>>>(
        source.typed_data(), pos.typed_data(), valid.typed_data(), rows,
        payload_width, auth_size, output->typed_data());
    return cudaGetLastError();
  };
  status = launch(stay_cot, stay_pos, stay_valid, auth_size);
  if (status != cudaSuccess) return ffi::Error::Internal("route-transpose stay scatter launch failed");
  status = launch(send_left_cot, send_left_pos, send_left_valid, share_capacity);
  if (status != cudaSuccess) return ffi::Error::Internal("route-transpose left scatter launch failed");
  status = launch(send_right_cot, send_right_pos, send_right_valid, share_capacity);
  if (status != cudaSuccess) return ffi::Error::Internal("route-transpose right scatter launch failed");
  return ffi::Error::Success();
}

constexpr auto RoutePackImpl = &RoutePackTypedImpl<ffi::F32>;
constexpr auto RouteOffsetProbe = &RouteOffsetProbeImpl;
constexpr auto RoutePackF64Impl = &RoutePackTypedImpl<ffi::F64>;
constexpr auto RouteBidirPackImpl = &RouteBidirPackTypedImpl<ffi::F32>;
constexpr auto RouteBidirPackF64Impl = &RouteBidirPackTypedImpl<ffi::F64>;
constexpr auto RouteBidirPackI16Impl =
    &RouteBidirPackTypedImpl<ffi::F32, ffi::S16>;
constexpr auto RouteBidirPackF64I16Impl =
    &RouteBidirPackTypedImpl<ffi::F64, ffi::S16>;
constexpr auto RouteMergeBidirImpl = &RouteMergeBidirTypedImpl<ffi::F32>;
constexpr auto RouteMergeBidirF64Impl = &RouteMergeBidirTypedImpl<ffi::F64>;
constexpr auto RouteMergeBidirI16Impl =
    &RouteMergeBidirTypedImpl<ffi::F32, ffi::S16>;
constexpr auto RouteMergeBidirF64I16Impl =
    &RouteMergeBidirTypedImpl<ffi::F64, ffi::S16>;
constexpr auto RouteMergeBidirPrimalI16 = &RouteMergeBidirPrimalI16Impl;
constexpr auto RouteBidirDriftPackPrimalI16 =
    &RouteBidirDriftPackPrimalI16Impl;
constexpr auto RouteBidirDriftMergePrimalI16 =
    &RouteBidirDriftMergePrimalI16Impl;
constexpr auto RouteMergeImpl = &RouteMergeTypedImpl<ffi::F32>;
constexpr auto RouteMergeF64Impl = &RouteMergeTypedImpl<ffi::F64>;
constexpr auto RouteMergeAuxImpl = &RouteMergeAuxTypedImpl<ffi::F32>;
constexpr auto RouteMergeAuxF64Impl = &RouteMergeAuxTypedImpl<ffi::F64>;
constexpr auto RouteTransposeSplitImpl =
    &RouteTransposeSplitTypedImpl<ffi::F32>;
constexpr auto RouteTransposeSplitF64Impl =
    &RouteTransposeSplitTypedImpl<ffi::F64>;
constexpr auto RouteTransposeScatterImpl =
    &RouteTransposeScatterTypedImpl<ffi::F32>;
constexpr auto RouteTransposeScatterF64Impl =
    &RouteTransposeScatterTypedImpl<ffi::F64>;

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_offset_probe, RouteOffsetProbe,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int32_t>("component")
        .Attr<int32_t>("record_words")
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::U32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_pack, RoutePackImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int32_t>("global_nmesh")
        .Attr<int32_t>("mesh_x")
        .Attr<int32_t>("mesh_y")
        .Attr<int32_t>("mesh_z")
        .Attr<int32_t>("direction")
        .Attr<int32_t>("num_devices")
        .Attr<int32_t>("capacity")
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U8>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_bidir_pack, RouteBidirPackImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int32_t>("global_nmesh")
        .Attr<int32_t>("mesh_x")
        .Attr<int32_t>("mesh_y")
        .Attr<int32_t>("mesh_z")
        .Attr<int32_t>("slice_width")
        .Attr<int32_t>("num_devices")
        .Attr<int32_t>("capacity")
        .Attr<int32_t>("stay_capacity")
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_merge, RouteMergeImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int32_t>("mesh_x")
        .Attr<int32_t>("mesh_y")
        .Attr<int32_t>("mesh_z")
        .Attr<int32_t>("capacity")
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::U8>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_merge_aux, RouteMergeAuxImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int32_t>("mesh_x")
        .Attr<int32_t>("mesh_y")
        .Attr<int32_t>("mesh_z")
        .Attr<int32_t>("capacity")
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_merge_bidir, RouteMergeBidirImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int32_t>("mesh_x")
        .Attr<int32_t>("mesh_y")
        .Attr<int32_t>("mesh_z")
        .Attr<int32_t>("capacity")
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_bidir_pack_i16, RouteBidirPackI16Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int32_t>("global_nmesh")
        .Attr<int32_t>("mesh_x")
        .Attr<int32_t>("mesh_y")
        .Attr<int32_t>("mesh_z")
        .Attr<int32_t>("slice_width")
        .Attr<int32_t>("num_devices")
        .Attr<int32_t>("capacity")
        .Attr<int32_t>("stay_capacity")
        .Arg<ffi::Buffer<ffi::S16>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_merge_bidir_i16, RouteMergeBidirI16Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int32_t>("mesh_x")
        .Attr<int32_t>("mesh_y")
        .Attr<int32_t>("mesh_z")
        .Attr<int32_t>("capacity")
        .Arg<ffi::Buffer<ffi::S16>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S16>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_merge_bidir_primal_i16, RouteMergeBidirPrimalI16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int32_t>("mesh_x")
        .Attr<int32_t>("mesh_y")
        .Attr<int32_t>("mesh_z")
        .Attr<int32_t>("capacity")
        .Arg<ffi::Buffer<ffi::S16>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S16>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_bidir_drift_pack_primal_i16,
    RouteBidirDriftPackPrimalI16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int32_t>("global_nmesh")
        .Attr<int32_t>("mesh_x")
        .Attr<int32_t>("mesh_y")
        .Attr<int32_t>("mesh_z")
        .Attr<int32_t>("slice_width")
        .Attr<int32_t>("num_devices")
        .Attr<int32_t>("capacity")
        .Arg<ffi::Buffer<ffi::S16>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::PRED>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_bidir_drift_merge_primal_i16,
    RouteBidirDriftMergePrimalI16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int32_t>("global_nmesh")
        .Attr<int32_t>("mesh_x")
        .Attr<int32_t>("mesh_y")
        .Attr<int32_t>("mesh_z")
        .Attr<int32_t>("slice_width")
        .Attr<int32_t>("num_devices")
        .Attr<int32_t>("record_capacity")
        .Attr<int32_t>("capacity")
        .Arg<ffi::Buffer<ffi::S16>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::PRED>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S16>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::PRED>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_transpose_split, RouteTransposeSplitImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int32_t>("auth_size")
        .Attr<int32_t>("share_capacity")
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_transpose_scatter, RouteTransposeScatterImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int32_t>("auth_size")
        .Attr<int32_t>("share_capacity")
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_pack_f64, RoutePackF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int32_t>("global_nmesh")
        .Attr<int32_t>("mesh_x")
        .Attr<int32_t>("mesh_y")
        .Attr<int32_t>("mesh_z")
        .Attr<int32_t>("direction")
        .Attr<int32_t>("num_devices")
        .Attr<int32_t>("capacity")
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U8>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_bidir_pack_f64, RouteBidirPackF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int32_t>("global_nmesh")
        .Attr<int32_t>("mesh_x")
        .Attr<int32_t>("mesh_y")
        .Attr<int32_t>("mesh_z")
        .Attr<int32_t>("slice_width")
        .Attr<int32_t>("num_devices")
        .Attr<int32_t>("capacity")
        .Attr<int32_t>("stay_capacity")
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_merge_f64, RouteMergeF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int32_t>("mesh_x")
        .Attr<int32_t>("mesh_y")
        .Attr<int32_t>("mesh_z")
        .Attr<int32_t>("capacity")
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::U8>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_merge_aux_f64, RouteMergeAuxF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int32_t>("mesh_x")
        .Attr<int32_t>("mesh_y")
        .Attr<int32_t>("mesh_z")
        .Attr<int32_t>("capacity")
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_merge_bidir_f64, RouteMergeBidirF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int32_t>("mesh_x")
        .Attr<int32_t>("mesh_y")
        .Attr<int32_t>("mesh_z")
        .Attr<int32_t>("capacity")
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_bidir_pack_f64_i16, RouteBidirPackF64I16Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int32_t>("global_nmesh")
        .Attr<int32_t>("mesh_x")
        .Attr<int32_t>("mesh_y")
        .Attr<int32_t>("mesh_z")
        .Attr<int32_t>("slice_width")
        .Attr<int32_t>("num_devices")
        .Attr<int32_t>("capacity")
        .Attr<int32_t>("stay_capacity")
        .Arg<ffi::Buffer<ffi::S16>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_merge_bidir_f64_i16, RouteMergeBidirF64I16Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int32_t>("mesh_x")
        .Attr<int32_t>("mesh_y")
        .Attr<int32_t>("mesh_z")
        .Attr<int32_t>("capacity")
        .Arg<ffi::Buffer<ffi::S16>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S16>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_transpose_split_f64, RouteTransposeSplitF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int32_t>("auth_size")
        .Attr<int32_t>("share_capacity")
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pmpp_route_transpose_scatter_f64, RouteTransposeScatterF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int32_t>("auth_size")
        .Attr<int32_t>("share_capacity")
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::F64>>());
