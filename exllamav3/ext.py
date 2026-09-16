from __future__ import annotations
import importlib.machinery
import importlib.util
import torch
from torch.utils.cpp_extension import load
import os
import sys
from .util.arch_list import maybe_set_arch_list_env

extension_name = "exllamav3_ext"
verbose = False  # Print wall of text when compiling
ext_debug = False  # Compile with debug options

# Determine if we're on Windows

windows = (os.name == "nt")

# Determine if extension is already installed or needs to be built

def is_precompiled_extension_available():
    spec = importlib.util.find_spec(extension_name)
    if not spec or not spec.origin or not spec.loader:
        return False
    return any(
        spec.origin.endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )

if is_precompiled_extension_available():
    import exllamav3_ext
else:

    # Kludge to get compilation working on Windows

    if windows:

        def find_msvc():

            # Possible locations for MSVC, in order of preference

            program_files_x64 = os.environ.get("ProgramW6432", os.environ.get("ProgramFiles", r"C:\Program Files"))
            program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")

            msvc_dirs = \
            [
                a + "\\Microsoft Visual Studio\\" + b + "\\" + c + "\\VC\\Tools\\MSVC\\"
                for b in ["2022", "2019", "2017"]
                for a in [program_files_x64, program_files_x86]
                for c in ["BuildTools", "Community", "Professional", "Enterprise", "Preview"]
            ]

            for msvc_dir in msvc_dirs:
                if not os.path.exists(msvc_dir): continue

                # Prefer the latest version

                versions = sorted(os.listdir(msvc_dir), reverse = True)
                for version in versions:

                    compiler_dir = msvc_dir + version + "\\bin\\Hostx64\\x64"
                    if os.path.exists(compiler_dir) and os.path.exists(compiler_dir + "\\cl.exe"):
                        return compiler_dir

            # No path found

            return None

        import subprocess

        # Check if cl.exe is already in the path

        try:

            subprocess.check_output(["where", "/Q", "cl"])

        # If not, try to find an installation of Visual Studio and append the compiler dir to the path

        except subprocess.CalledProcessError as e:

            cl_path = find_msvc()
            if cl_path:
                if verbose:
                    print(" -- Injected compiler path:", cl_path)
                os.environ["path"] += ";" + cl_path
            else:
                print(" !! Unable to find cl.exe; compilation will probably fail", file = sys.stderr)

    # compiler flags

    extra_cflags = []
    is_hip = torch.version.hip is not None
    if is_hip:
        # hipcc is clang-based; nvcc-only flags (-Xcudafe, --use_fast_math, -lineinfo,
        # --ptxas-options) are not accepted. -ffast-math is the closest equivalent of
        # --use_fast_math. gfx10/gfx11 targets execute in wave32 by default, matching the
        # 32-lane assumptions throughout the kernels.
        extra_cuda_cflags = ["-O3", "-ffast-math", "-DHIPBLAS_USE_HIP_HALF"]
        if os.environ.get("EXL3_WMMA") == "1":
            # Opt-in hardware WMMA for the paired m16n16k16 MMA on gfx11. Off by
            # default: the path has a known NaN bug on M>=3 GEMM shapes.
            extra_cuda_cflags += ["-DEXL3_WMMA"]
    else:
        extra_cuda_cflags = [
            "-lineinfo", "-O3", "--use_fast_math",
            "-Xcudafe", "--diag_suppress=177",
            "-Xcudafe", "--diag_suppress=20012",
        ]

    if windows:
        # TODO: preprocessor and lean_and_mean flags are needed for Windows cu132 build, verify that they don't break
        #       older cu128 builds
        # NOMINMAX: windows.h otherwise defines min/max function-like macros that break every
        # std::min/std::max call site parsed after it (WIN32_LEAN_AND_MEAN does not suppress them).
        # Defined globally so it holds regardless of include order in any TU (mirrors setup.py).
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

    if not windows and not is_hip and (cuda_host_cxx := os.environ.get("CUDAHOSTCXX")):
        extra_cuda_cflags += ["-ccbin", cuda_host_cxx]

    if verbose:
        extra_cuda_cflags += ["-v" if is_hip else "--ptxas-options=-v"]

    # linker flags

    extra_ldflags = []

    if windows:
        extra_ldflags += ["hipblas.lib" if is_hip else "cublas.lib"]
        if sys.base_prefix != sys.prefix:
            extra_ldflags += [f"/LIBPATH:{os.path.join(sys.base_prefix, 'libs')}"]
    elif is_hip:
        # The extension calls hipBLAS and hipBLASLt directly (hgemm.cu, graph.cu); link both
        extra_ldflags += ["-lhipblas", "-lhipblaslt"]

    # sources

    library_dir = os.path.dirname(os.path.abspath(__file__))
    sources_dir = os.path.join(library_dir, extension_name)
    sources = [
        os.path.abspath(os.path.join(root, file))
        for root, _, files in os.walk(sources_dir)
        for file in files
        if file.endswith(('.c', '.cpp', '.cu'))
        # Skip hipify outputs left over from a previous build: torch's hipify marks
        # them already-translated (hipified_path = None), which crashes the ninja
        # writer, and they must not be compiled as sources anyway.
        and '_hip.' not in file and not file.startswith('hip_')
    ]

    extra_include_paths = [sources_dir]
    if is_hip:
        # The pip ROCm SDK ships runtime headers only; torch's c10 headers pull in
        # thrust/complex.h, which lives in a full ROCm install. Add it when present.
        for rocm_root in (
            os.environ.get("ROCM_PATH"),
            os.environ.get("ROCM_HOME"),
            "/opt/rocm",
        ):
            if not rocm_root:
                continue
            inc = os.path.join(rocm_root, "include")
            if os.path.exists(os.path.join(inc, "thrust", "complex.h")):
                extra_include_paths.append(inc)
                break

    maybe_set_arch_list_env()
    exllamav3_ext = load(
        name = extension_name,
        sources = sources,
        extra_include_paths = extra_include_paths,
        verbose = verbose,
        extra_ldflags = extra_ldflags,
        extra_cuda_cflags = extra_cuda_cflags,
        extra_cflags = extra_cflags
    )
