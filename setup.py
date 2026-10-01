import os

from setuptools import setup

try:
    from torch import version as torch_version
    from torch.utils import cpp_extension
except ImportError:
    torch_version = None
    cpp_extension = None

torch = torch_version is not None

extension_name = "exllamav3_ext"
precompile = "EXLLAMA_NOCOMPILE" not in os.environ
verbose = "EXLLAMA_VERBOSE" in os.environ
ext_debug = "EXLLAMA_EXT_DEBUG" in os.environ

if precompile and not torch:
    print("Cannot precompile unless torch is installed.")
    print("To explicitly JIT install run EXLLAMA_NOCOMPILE= pip install <xyz>")

windows = os.name == "nt"

extra_cflags = []
is_hip = bool(torch_version is not None and torch_version.hip)
if is_hip:
    # hipcc is clang-based; nvcc-only flags (-Xcudafe, --use_fast_math, -lineinfo) are not
    # accepted. -ffast-math is the closest equivalent of --use_fast_math. gfx10/gfx11
    # targets execute in wave32 by default, matching the 32-lane assumptions in the kernels.
    extra_cuda_cflags = ["-O3", "-ffast-math", "-DHIPBLAS_USE_HIP_HALF"]
    if os.environ.get("EXL3_WMMA") == "1":
        # Opt-in hardware WMMA for the paired m16n16k16 MMA on gfx11. Off by
        # default: the path has a known NaN bug on M>=3 GEMM shapes.
        extra_cuda_cflags += ["-DEXL3_WMMA"]
    if os.environ.get("EXL3_CUMODE") == "1":
        # gfx11 default is WGP (2 CUs). CU mode schedules each 256-thread GEMV
        # block on one CU, doubling the number of independent workgroup slots
        # (48 WGPs → 96 CUs). Measured: CU+MULT=2 = 38.36 vs WGP 35.67.
        extra_cuda_cflags += ["-mcumode", "-DEXL3_CUMODE"]
        # gfx1100 4096/256: CU + MULT=2 is 38.36 vs WGP 35.67. Pair with
        # default MULT=2 under -DEXL3_CUMODE (exl3_gemv_int8.cu).
        # hipcc also reads HIPCC_FLAGS; ninja does not echo argv so the
        # Dockerfile cannot grep the compiler command line.
        os.environ["HIPCC_FLAGS"] = (os.environ.get("HIPCC_FLAGS", "") + " -mcumode").strip()
else:
    extra_cuda_cflags = [
        "-lineinfo", "-O3", "--use_fast_math",
        "-Xcudafe", "--diag_suppress=177",
        "-Xcudafe", "--diag_suppress=20012",
    ]

if windows:
    # NOMINMAX: windows.h otherwise defines min/max function-like macros that break every
    # std::min/std::max call site parsed after it (WIN32_LEAN_AND_MEAN does not suppress them).
    # Defined globally so it holds regardless of include order in any TU.
    # No -std flags here: torch's cpp_extension appends its own (unconditionally on the Windows
    # nvcc path), and a second -std argument is a fatal nvcc error, not an override.
    extra_cflags += ["/Ox", "/Zc:preprocessor", "/DWIN32_LEAN_AND_MEAN", "/DNOMINMAX"]
    extra_cuda_cflags += ["-DWIN32_LEAN_AND_MEAN", "-DNOMINMAX", "-Xcompiler=/Zc:preprocessor"]
    if ext_debug:
        extra_cflags += ["/Zi"]
        extra_cuda_cflags += []
else:
    extra_cflags += ["-Ofast"]
    extra_cuda_cflags += []
    if ext_debug:
        extra_cflags += ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]
        extra_cuda_cflags += []

if not is_hip and (cuda_host_cxx := os.environ.get("CUDAHOSTCXX")):
    extra_cuda_cflags += ["-ccbin", cuda_host_cxx]

extra_compile_args = {
    "cxx": extra_cflags,
    "nvcc": extra_cuda_cflags,
}
if is_hip:
    extra_compile_args["hipcc"] = extra_cuda_cflags
    extra_compile_args["hip"] = extra_cuda_cflags
# pip's pyproject backend swallows setup.py stdout, so the Dockerfile cannot
# grep the print. Stamp a file in the same RUN layer instead.
_flag_line = "EXL3_HIP_CFLAGS: %s is_hip=%s EXL3_CUMODE=%s\n" % (
    extra_cuda_cflags, is_hip, os.environ.get("EXL3_CUMODE"))
try:
    with open("/tmp/exl3_hip_cflags.txt", "w") as _hf:
        _hf.write(_flag_line)
except OSError:
    pass
print(_flag_line, end="", flush=True)

library_dir = "exllamav3"
sources_dir = os.path.join(library_dir, extension_name)
sources = [
    os.path.relpath(os.path.join(root, file), start=os.path.dirname(__file__))
    for root, _, files in os.walk(sources_dir)
    for file in files
    if file.endswith(('.c', '.cpp', '.cu'))
    # Skip hipify outputs: they are regenerated in-place by torch's BuildExtension
    # on every ROCm rebuild, and compiling them alongside their non-hipified
    # counterparts produces duplicate-symbol link errors.
    and '_hip.' not in file and not file.startswith('hip_')
]

setup_kwargs = {}
if precompile and cpp_extension is not None:
    setup_kwargs = {
        "ext_modules": [
            cpp_extension.CUDAExtension(
                extension_name,
                sources,
                extra_compile_args=extra_compile_args,
                include_dirs=[sources_dir],
                libraries=(
                    ["hipblas"] if is_hip else
                    ["cublas"] if windows else
                    []
                ),
            )
        ],
        "cmdclass": {"build_ext": cpp_extension.BuildExtension},
    }

setup(
    verbose=verbose,
    **setup_kwargs,
)
