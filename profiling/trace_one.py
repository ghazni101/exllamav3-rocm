import os, sys
import torch
from exllamav3 import Config, Model
from exllamav3.ext import exllamav3_ext as ext

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/Qwen3.8-27B-SC_4.00bpw_H5_V6")
M = int(os.environ.get("TRACE_M", "8"))
REPS = int(os.environ.get("TRACE_REPS", "5"))

torch.manual_seed(0)
config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)

# first K=4 q_proj of the target shape
lin = None
for m_ in model.modules:
    a = getattr(m_, "attn", None)
    if a is not None and getattr(a, "q_proj", None) is not None:
        lin = a.q_proj
        break
lin.load(device = "cuda:0")
inner = lin.inner
k, n = inner.in_features, inner.out_features
x = torch.randn((M, k), dtype = torch.half, device = "cuda:0") * 0.05
xh = torch.empty_like(x)
c = torch.empty((M, n), dtype = torch.half, device = "cuda:0")
for _ in range(REPS):
    ext.exl3_gemm(x, inner.trellis, c, inner.suh, xh, inner.svh, -1, False, True, 0)
torch.cuda.synchronize()
print(f"[trace_one] done m={M} msq={os.environ.get('EXL3_INT8_MSQ', 'unset')}")
