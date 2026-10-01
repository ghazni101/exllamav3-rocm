#!/usr/bin/env bash
# Live correctness on the 41.30 incumbent image (goal-isa-had).
# 1) GEMV numerical: msq vs coop vs per-matrix int8
# 2) Natural-prompt greedy smoke (64 tok)
# 3) 4096/256 greedy re-run vs final-post-revert.json (pre-ISA golden)
set -euo pipefail
ROOT=/home/ghazni/github/exllamav3-rocm
OUT=$ROOT/profiling/out_guided
IMG=exllamav3-rocm:goal-isa-had

docker run --rm --name exl3-had-r40-correctness \
  --device /dev/kfd --device /dev/dri \
  --group-add 44 --group-add 993 --ipc host --shm-size 4g \
  --ulimit nofile=65536:65536 \
  -e PYTHONUNBUFFERED=1 \
  -e EXL3_MODEL=/models/qwen38-27b \
  -e EXL3_SQ_GRID_MULT=2 \
  -e EXL3_SQ_ROWS_PER=40 \
  -e EXL3_SQ_LAUNCH_LOG=4 \
  -e TG_OUTPUT=/out_guided/had-r40-recheck.json \
  -e TG_REFERENCE=/out_guided/final-post-revert.json \
  -e TG_REQUIRE_PARITY=1 \
  -v /home/ghazni/models/exl3/Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw:/models/qwen38-27b:ro \
  -v "$ROOT/profiling":/prof:ro \
  -v "$OUT":/out_guided \
  --entrypoint bash "$IMG" -c '
set -euo pipefail
echo "=== 1. GEMV numerical: test_msq_ab.py ==="
python3 /opt/exllamav3/test_msq_ab.py
echo "=== 2. Natural-prompt greedy smoke (64 tok) ==="
python3 - << "PY"
import os, json, torch
from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.cache import CacheLayer_fp16
from exllamav3.generator.sampler import GreedySampler
model_dir = os.environ["EXL3_MODEL"]
config = Config.from_directory(model_dir)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=2048, layer_type=CacheLayer_fp16)
model.load()
tok = Tokenizer.from_config(config)
gen = Generator(model=model, cache=cache, tokenizer=tok)
assert gen.num_draft_tokens == 0
prompt = "The capital of France is"
ids = tok.encode(prompt, add_bos=True)
job = Job(input_ids=ids, max_new_tokens=64, stop_conditions=[], sampler=GreedySampler(), seed=0)
gen.enqueue(job)
while True:
    events = gen.iterate()
    torch.cuda.synchronize()
    for e in events:
        if e.get("eos"):
            seq = job.sequences[0].sequence_ids.torch().flatten()
            text = tok.decode(seq.unsqueeze(0), decode_special_tokens=False)
            new_ids = seq.tolist()[-int(e["new_tokens"]):]
            print("NATURAL_PROMPT:", prompt)
            print("NATURAL_OUT:", text)
            print("NATURAL_NEW_IDS:", new_ids)
            print("NATURAL_NTOK", e["new_tokens"])
            Path = __import__("pathlib").Path
            Path("/out_guided/had-r40-natural.json").write_text(json.dumps({
                "prompt": prompt, "text": text if isinstance(text, str) else str(text),
                "new_ids": new_ids, "n": int(e["new_tokens"]),
            }, indent=2) + "\n")
            raise SystemExit(0)
PY
echo "=== 3. 4096/256 greedy vs final-post-revert golden ==="
python3 /prof/guided_tg.py
echo "=== ALL CHECKS DONE ==="
'
