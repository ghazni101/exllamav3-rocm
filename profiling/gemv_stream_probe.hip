// Minimal K=4 GEMV address-pattern reproduction; no extraction or dp4a.
// Logical (k,n)=(5120,17408): k/16 packed rows, n*8 bytes/row,
// 44,564,480 bytes/copy. Eight independent copies exceed the 96 MiB cache.
// This measures aligned 16-byte loads, not production GEMV throughput.
// hipcc --offload-arch=gfx1100 -O3 gemv_stream_probe.hip -o gemv_stream_probe
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(expr) do { const hipError_t err = (expr); if (err != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #expr, hipGetErrorString(err)); std::exit(1); \
} } while (0)

constexpr int K = 5120, N = 17408;
constexpr int Rows = K / 16, RowWords = N * 2;
constexpr int Threads = 256, Bands = 8, Copies = 8, Reps = 10;
constexpr int Spans = RowWords * sizeof(uint32_t) / (32 * sizeof(uint4));
constexpr int Blocks = (Spans + 7) / 8;
constexpr size_t Words = size_t(Rows) * RowWords;
constexpr int SinkWords = Blocks * Bands * Threads;
static_assert(RowWords * sizeof(uint32_t) % (32 * sizeof(uint4)) == 0);
static_assert(Words * sizeof(uint32_t) == size_t(K) * N / 2);

__global__ void stream16(const uint32_t* __restrict__ input,
                         uint32_t* __restrict__ sink)
{
    const int span = blockIdx.x * 8 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    uint32_t sum = 0;
    if (span < Spans)
    {
        const int begin = Rows * blockIdx.y / Bands;
        const int end = Rows * (blockIdx.y + 1) / Bands;
        for (int row = begin; row < end; ++row)
        {
            const uint4 value = *reinterpret_cast<const uint4*>(
                input + size_t(row) * RowWords + span * 128 + lane * 4);
            sum += value.x + value.y + value.z + value.w;
        }
    }
    // Unique, unconditional sink per thread; no races or conditional DCE guard.
    sink[(blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x] = sum;
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<uint32_t> host(Words * Copies);
    uint32_t expected[Copies] = {};
    uint32_t state = 0x31415926;
    for (int copy = 0; copy < Copies; ++copy)
        for (size_t i = 0; i < Words; ++i)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            host[size_t(copy) * Words + i] = state;
            expected[copy] += state;
        }

    uint32_t *input = nullptr, *sink = nullptr;
    CK(hipMalloc(&input, host.size() * sizeof(uint32_t)));
    CK(hipMalloc(&sink, size_t(SinkWords) * sizeof(uint32_t)));
    CK(hipMemcpy(input, host.data(), host.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    std::printf("logical k=%d n=%d; packed rows=%d stride=%zu B; copy=%zu B; pool=%zu B\n",
                K, N, Rows, RowWords * sizeof(uint32_t), Words * sizeof(uint32_t),
                host.size() * sizeof(uint32_t));
    std::vector<uint32_t> result(SinkWords);
    for (int copy = 0; copy < Copies; ++copy)
    {
        stream16<<<dim3(Blocks, Bands), Threads>>>(input + size_t(copy) * Words, sink);
        CK(hipGetLastError());
        CK(hipMemcpy(result.data(), sink, result.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
        uint32_t actual = 0;
        for (uint32_t value : result) actual += value;
        if (actual != expected[copy])
        {
            std::fprintf(stderr, "checksum FAIL copy=%d actual=%08x expected=%08x\n",
                         copy, actual, expected[copy]);
            return 1;
        }
    }
    std::printf("checksum PASS: all %d independent copies\n", Copies);

    hipEvent_t start, end;
    CK(hipEventCreate(&start)); CK(hipEventCreate(&end));
    CK(hipEventRecord(start));
    for (int rep = 0; rep < Reps; ++rep)
        for (int copy = 0; copy < Copies; ++copy)
            stream16<<<dim3(Blocks, Bands), Threads>>>(input + size_t(copy) * Words, sink);
    CK(hipGetLastError());
    CK(hipEventRecord(end)); CK(hipEventSynchronize(end));
    float ms = 0;
    CK(hipEventElapsedTime(&ms, start, end));
    const double bytes = double(Words) * sizeof(uint32_t) * Copies * Reps;
    std::printf("16B/lane grid=(%d,%d) threads=%d: %.4f ms/copy, %.1f GB/s useful input\n",
                Blocks, Bands, Threads, ms / (Copies * Reps), bytes / (ms * 1e6));
    CK(hipEventDestroy(start)); CK(hipEventDestroy(end));
    CK(hipFree(sink)); CK(hipFree(input));
}
