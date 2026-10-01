// Repack A/B probe for the K=6 narrow unit (lm_head shape).
//
// MODE 0 (legacy): warp per adjacent block pair; per k-row each lane reads 4
//   scattered 4B words per block (ext4w x2 for ext8w), addresses inside a
//   192-word block-pair region whose stride to the next row is ~3 MB.
// MODE 1 (repack): B'[kb][pair][lane][8] — the 8 words a lane needs for both
//   blocks are contiguous; one row = 32B/lane = 1KB contiguous per warp.
//
// Same extraction math + hash-mul + dp4a chain both arms; per-warp checksum of
// hashed windows is compared for bit-exactness.
//
//   hipcc --offload-arch=gfx1100 -O3 -mcumode -o repack_probe repack_probe.hip
//   ./repack_probe
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

#define THREADS 256
#define K_BITS 6
#define PW 8                    // words per lane per row per block-pair (4 A + 4 B)
#define NROWS 40                // rows_per slice (rows_total=320, ksplit=8)
#define NBP_PER_WARP_PAIR 0     // unused

// lm_head: k=5120 -> 320 krows, n=248320 -> 970 nb256 -> 7760 pairs
#define ROWS_TOTAL 40   // probe slice only
#define NPAIRS (970 * 8)
#define WORDS_PER_ROW_PAIR (32 * PW)            // 32 lanes * 8 words
#define ROW_STRIDE_U32 ((size_t)(970 * 256) * K_BITS / 2)  // n*K/2 = 372480 u32 (probe slice)

__device__ __forceinline__ int dp4a_us(uint32_t a, uint32_t b, int c)
{
    return __builtin_amdgcn_sudot4(true, a, false, b, c, false);
}

__device__ __forceinline__ uint32_t fshift(uint32_t b, uint32_t a, int s)
{
    uint64_t m = ((uint64_t) a << 32) | (uint64_t) b;
    return (uint32_t) (m >> s);
}

template <int bits>
__device__ __forceinline__ int wrap_idx(int i)
{
    constexpr int words = bits * 256 / 32;
    return i >= words ? i - words : i;
}

// ext4w-equivalent indices for K=6, computed once per lane
struct src6 { int i0a, i2a, s2a, i0b, i2b, s2b; };
__device__ __forceinline__ src6 ext8w_addr6(int t0)
{
    src6 s;
    int b0 = (t0 + 257) * K_BITS - 16;
    int b2 = b0 + 3 * K_BITS + 16;
    s.i0a = b0 / 32;
    s.i2a = (b2 - 1) / 32;
    s.s2a = (s.i2a + 1) * 32 - b2;
    b0 = (t0 + 261) * K_BITS - 16;
    b2 = b0 + 3 * K_BITS + 16;
    s.i0b = b0 / 32;
    s.i2b = (b2 - 1) / 32;
    s.s2b = (s.i2b + 1) * 32 - b2;
    s.i0a = wrap_idx<K_BITS>(s.i0a); s.i2a = wrap_idx<K_BITS>(s.i2a);
    s.i0b = wrap_idx<K_BITS>(s.i0b); s.i2b = wrap_idx<K_BITS>(s.i2b);
    return s;
}

// hash+dp4a for one block's ext4w pair -> 8 windows
__device__ __forceinline__ void ext4w_w(uint32_t a, uint32_t b, int s2,
    uint32_t& w0, uint32_t& w1, uint32_t& w2, uint32_t& w3)
{
    w3 = fshift(b, a, s2) & 0xffff;
    w2 = fshift(b, a, s2 + K_BITS) & 0xffff;
    w1 = fshift(b, a, s2 + K_BITS * 2) & 0xffff;
    w0 = fshift(b, a, s2 + K_BITS * 3) & 0xffff;
}

__device__ __forceinline__ uint32_t hash8(uint32_t a0, uint32_t b0, int s2a,
                                          uint32_t a1, uint32_t b1, int s2b,
                                          uint32_t as01x, uint32_t as01y,
                                          uint32_t as89x, uint32_t as89y,
                                          int acc0, int acc1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    ext4w_w(a0, b0, s2a, w0, w1, w2, w3);
    ext4w_w(a1, b1, s2b, w4, w5, w6, w7);
    w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
    w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
    acc0 = dp4a_us(w0, as01x, acc0);
    acc0 = dp4a_us(w1, as01y, acc0);
    acc0 = dp4a_us(w2, as89x, acc0);
    acc0 = dp4a_us(w3, as89y, acc0);
    acc1 = dp4a_us(w4, as01x, acc1);
    acc1 = dp4a_us(w5, as01y, acc1);
    acc1 = dp4a_us(w6, as89x, acc1);
    acc1 = dp4a_us(w7, as89y, acc1);
    return (uint32_t) acc0 ^ ((uint32_t) acc1 * 2654435761u);
}

// MODE 0: scattered loads from legacy layout B[kb][nbp][96 words]
__global__ __launch_bounds__(THREADS)
void k_scattered(const uint32_t* __restrict__ B, const uint32_t* __restrict__ sh_fake,
                 int units_per_block, uint32_t* __restrict__ out)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    src6 s = ext8w_addr6(lane << 3);
    int nbp0 = (blockIdx.x * 8 + warp) + gridDim.x * 8 * units_per_block;  // unused tail
    uint32_t chk = 0;
    for (int u = 0; u < units_per_block; ++u)
    {
        int nbp = blockIdx.x * 8 + warp + u * (gridDim.x * 8);
        if (nbp >= NPAIRS) break;
        const uint32_t* bp = B + (size_t) nbp * (K_BITS * 16);
        for (int kb = 0; kb < NROWS; ++kb)
        {
            const uint32_t* blockA = bp + (size_t) kb * ROW_STRIDE_U32;
            const uint32_t* blockB = blockA + 8 * K_BITS;
            uint32_t wa[4] = {blockA[s.i0a], blockA[s.i2a], blockA[s.i0b], blockA[s.i2b]};
            uint32_t wb[4] = {blockB[s.i0a], blockB[s.i2a], blockB[s.i0b], blockB[s.i2b]};
            chk += hash8(wa[0], wa[1], s.s2a, wa[2], wa[3], s.s2b,
                         0x01010101u * (lane + 1), 0x02020202u, 0x03030303u, 0x04040404u,
                         0, 0);
            chk += hash8(wb[0], wb[1], s.s2a, wb[2], wb[3], s.s2b,
                         0x05050505u, 0x06060606u, 0x07070707u, 0x08080808u,
                         0, 0);
        }
    }
    if (chk == 0xdeadbeef) out[blockIdx.x * THREADS + threadIdx.x] = chk;
    else out[blockIdx.x * THREADS + threadIdx.x] = chk;
}

// MODE 1: repacked layout B2[kb][pair][lane][8]: contiguous per lane.
__global__ __launch_bounds__(THREADS)
void k_repacked(const uint32_t* __restrict__ B2, const uint32_t* __restrict__ sh_fake,
                int units_per_block, uint32_t* __restrict__ out)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    src6 s = ext8w_addr6(lane << 3);   // same math; words now contiguous
    uint32_t chk = 0;
    for (int u = 0; u < units_per_block; ++u)
    {
        int nbp = blockIdx.x * 8 + warp + u * (gridDim.x * 8);
        if (nbp >= NPAIRS) break;
        const uint32_t* lp = B2 + (size_t) nbp * WORDS_PER_ROW_PAIR + lane * PW;
        for (int kb = 0; kb < NROWS; ++kb)
        {
            const uint4* p = (const uint4*) (lp + (size_t) kb * (NPAIRS * WORDS_PER_ROW_PAIR));
            uint4 q0 = p[0];
            uint4 q1 = p[1];
            // layout: [a0 a1 a2 a3][b0 b1 b2 b3]
            chk += hash8(q0.x, q0.y, s.s2a, q0.z, q0.w, s.s2b,
                         0x01010101u * (lane + 1), 0x02020202u, 0x03030303u, 0x04040404u,
                         0, 0);
            chk += hash8(q1.x, q1.y, s.s2a, q1.z, q1.w, s.s2b,
                         0x05050505u, 0x06060606u, 0x07070707u, 0x08080808u,
                         0, 0);
        }
    }
    out[blockIdx.x * THREADS + threadIdx.x] = chk;
}

int main()
{
    int dev = 0; hipSetDevice(dev);
    hipDeviceProp_t prop; hipGetDeviceProperties(&prop, 0);
    // legacy buffer covers the rows the kernel touches: rows 0..ROWS_TOTAL-1 plus
    // the widest pair offset within row 0.
    size_t words = (size_t)(ROWS_TOTAL - 1) * ROW_STRIDE_U32 + (size_t) NPAIRS * K_BITS * 16;
    // repacked buffer: B2[kb][pair][lane][8]
    size_t words2 = (size_t) ROWS_TOTAL * NPAIRS * WORDS_PER_ROW_PAIR;
    printf("device: %s CUs=%d clock=%.2f GHz | B %.0f MB  B2 %.0f MB\n",
           prop.name, prop.multiProcessorCount, prop.clockRate * 1e-6,
           words * 4.0 / 1e6, words2 * 4.0 / 1e6);

    uint32_t *d_B=nullptr, *d_B2=nullptr, *d_out0=nullptr, *d_out1=nullptr, *d_fake=nullptr;
    if (hipMalloc(&d_B, words * 4) || hipMalloc(&d_B2, words2 * 4)
        || hipMalloc(&d_out0, 384 * THREADS * 4) || hipMalloc(&d_out1, 384 * THREADS * 4)
        || hipMalloc(&d_fake, 1024)) { printf("hipMalloc failed (VRAM)\n"); return 3; }

    // fill legacy layout
    std::vector<uint32_t> h(words);
    for (size_t i = 0; i < words; ++i) h[i] = (uint32_t) (i * 2654435761u);
    hipMemcpy(d_B, h.data(), words * 4, hipMemcpyHostToDevice);

    // build repacked copy on host (only the (kb, pair) range both modes touch)
    // B2[kb][pair][lane][j]: j 0..3 = blockA words i0a,i2a,i0b,i2b ; 4..7 = blockB
    {
        std::vector<uint32_t> h2(words2);
        // compute per-lane src indices on host (mirror of ext8w_addr6)
        int iL[32][4];
        for (int lane = 0; lane < 32; ++lane)
        {
            int t0 = lane << 3;
            int b0 = (t0 + 257) * K_BITS - 16, b2 = b0 + 3 * K_BITS + 16;
            int i0a = b0 / 32, i2a = (b2 - 1) / 32;
            b0 = (t0 + 261) * K_BITS - 16; b2 = b0 + 3 * K_BITS + 16;
            int i0b = b0 / 32, i2b = (b2 - 1) / 32;
            auto wr = [](int i) { constexpr int W = K_BITS * 256 / 32; return i >= W ? i - W : i; };
            iL[lane][0] = wr(i0a); iL[lane][1] = wr(i2a);
            iL[lane][2] = wr(i0b); iL[lane][3] = wr(i2b);
        }
        // pairs visited by grid 384 x 8 warps, all u
        // repack covers kb in [0,NROWS) only (one slice); other rows inert
        for (int pair = 0; pair < NPAIRS && pair < 384 * 8 * 3; ++pair)
        {
            for (int kb = 0; kb < NROWS; ++kb)
            {
                const uint32_t* blockA = h.data() + (size_t) pair * (K_BITS * 16) + (size_t) kb * ROW_STRIDE_U32;
                const uint32_t* blockB = blockA + 8 * K_BITS;
                uint32_t* dst = h2.data() + (size_t) kb * (NPAIRS * WORDS_PER_ROW_PAIR)
                              + (size_t) pair * WORDS_PER_ROW_PAIR;
                for (int lane = 0; lane < 32; ++lane)
                {
                    for (int j = 0; j < 4; ++j)
                    {
                        dst[lane * PW + j] = blockA[iL[lane][j]];
                        dst[lane * PW + 4 + j] = blockB[iL[lane][j]];
                    }
                }
            }
        }
        hipMemcpy(d_B2, h2.data(), words2 * 4, hipMemcpyHostToDevice);
    }

    int grid = 384;
    int units_per_block = (NPAIRS + grid * 8 - 1) / (grid * 8);

    auto bench = [&](auto kern, const uint32_t* buf, uint32_t* outb, const char* name)
    {
        for (int i = 0; i < 3; ++i) kern<<<grid, THREADS>>>(buf, d_fake, units_per_block, outb);
        hipDeviceSynchronize();
        hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
        hipEventRecord(e0);
        int reps = 10;
        for (int i = 0; i < reps; ++i) kern<<<grid, THREADS>>>(buf, d_fake, units_per_block, outb);
        hipEventRecord(e1); hipEventSynchronize(e1);
        float ms; hipEventElapsedTime(&ms, e0, e1);
        double bytes = (double) grid * 8 * NROWS * WORDS_PER_ROW_PAIR * 4 * units_per_block;
        bytes = (double) NROWS * NPAIRS * WORDS_PER_ROW_PAIR * 4;
        printf("%-10s t=%8.3f ms  %7.1f GB/s (useful %7.1f)\n", name, ms / reps,
               bytes / (ms / reps) / 1e6, bytes / (ms / reps) / 1e6);
        hipEventDestroy(e0); hipEventDestroy(e1);
    };

    bench(k_scattered, d_B, d_out0, "scattered");
    bench(k_repacked, d_B2, d_out1, "repacked");
    bench(k_scattered, d_B, d_out0, "scattered2");
    bench(k_repacked, d_B2, d_out1, "repacked2");

    // checksum compare
    std::vector<uint32_t> o0(384 * THREADS), o1(384 * THREADS);
    hipMemcpy(o0.data(), d_out0, o0.size() * 4, hipMemcpyDeviceToHost);
    hipMemcpy(o1.data(), d_out1, o1.size() * 4, hipMemcpyDeviceToHost);
    long long bad = 0;
    for (size_t i = 0; i < o0.size(); ++i) if (o0[i] != o1[i]) ++bad;
    printf("checksum mismatches: %lld\n", bad);
    return bad ? 2 : 0;
}
