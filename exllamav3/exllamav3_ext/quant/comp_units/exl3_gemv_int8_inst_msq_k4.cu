#include "exl3_gemv_int8_instances.cuh"
#include "../exl3_gemv_int8_kernel.cuh"

void* exl3_gemv_int8_msq_sel_k4(bool c_fp32, bool residual, int load_k)
{
    #define SEL_(L_) \
        if (c_fp32)  return residual ? (void*) exl3_gemv_int8_msq_kernel<4, true, true, L_> \
                                     : (void*) exl3_gemv_int8_msq_kernel<4, true, false, L_>; \
        else         return residual ? (void*) exl3_gemv_int8_msq_kernel<4, false, true, L_> \
                                     : (void*) exl3_gemv_int8_msq_kernel<4, false, false, L_>;
    if (load_k == 2) { SEL_(2) }
    if (load_k == 1) { SEL_(1) }
    SEL_(0)
    #undef SEL_
}
