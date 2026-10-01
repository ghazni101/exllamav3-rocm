#include "exl3_gemv_int8_instances.cuh"
#include "../exl3_gemv_int8_kernel.cuh"

void* exl3_gemv_int8_sq_sel_k4(int M, bool c_fp32, bool residual, int load_k)
{
    #define SELM_(M_, L_) \
        if (c_fp32)  return residual ? (void*) exl3_gemv_int8_sq_kernel<4, M_, true, true, L_> \
                                     : (void*) exl3_gemv_int8_sq_kernel<4, M_, true, false, L_>; \
        else         return residual ? (void*) exl3_gemv_int8_sq_kernel<4, M_, false, true, L_> \
                                     : (void*) exl3_gemv_int8_sq_kernel<4, M_, false, false, L_>;
    if (load_k == 2)
    {
        switch (M)
        {
            case 1: { SELM_(1, 2) }
            case 2: { SELM_(2, 2) }
            case 4: { SELM_(4, 2) }
        }
    }
    else if (load_k == 1)
    {
        switch (M)
        {
            case 1: { SELM_(1, 1) }
            case 2: { SELM_(2, 1) }
            case 4: { SELM_(4, 1) }
        }
    }
    else
    {
        switch (M)
        {
            case 1: { SELM_(1, 0) }
            case 2: { SELM_(2, 0) }
            case 4: { SELM_(4, 0) }
        }
    }
    #undef SELM_
    return nullptr;
}
