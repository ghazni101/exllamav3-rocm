# exllamav3 ROCm dev workflow — immutable base image + bind-mounted repo.
#
# The image contains torch + deps + toolchain only; the working tree is mounted
# read-only, so Python edits are live after a container restart and C++ edits
# just recompile the JIT extension (see Dockerfile.rocm10 header). No image
# rebuild per experiment.
#
#   make image           build the base image
#   make shell           interactive bash inside the container
#   make serve           serve_rocm.py (model loaded, port published)
#   make profile MODE=kernel|api|pmc   rocprofv3 trace -> .profiling/
#   make ext             force-recompile the HIP extension (JIT cache)
#   make sweep           remove profiling outputs + pruned dead images/volumes

IMAGE    ?= local/exl3-rocm:gfx1101-base
ARCH     ?= gfx1101
MODEL_DIR?= /home/ghazni/models/exl3/turboderp/Qwen3.8-27B-EXL3-SC_1.40bpw_H3_V3
MODEL_TAG?= qwen38-27b
CACHE_VOL?= exl3_exl3-cache
PORT     ?= 9001
PROF_OUT ?= $(CURDIR)/.profiling
MODES    := kernel api pmc

# GPU + mount plumbing shared by every run. Repo rw at /opt/exllamav3 — torch's
# ROCm JIT hipifies the extension sources IN PLACE (writes */*.hip and *_hip.*
# files into exllamav3/exllamav3_ext/, already gitignored); a :ro mount breaks
# the compile. Model:ro at /models/<tag>; cache volume at /root/.cache (holds
# the JIT-compiled .so + HF cache).
COMMON := --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
          -e EXL3_MODEL=/models/$(MODEL_TAG) -e EXL3_CACHE_TOKENS=32768 -e MAX_JOBS=8 \
          -v $(CURDIR):/opt/exllamav3 \
          -v $(MODEL_DIR):/models/$(MODEL_TAG):ro \
          --mount source=$(CACHE_VOL),target=/root/.cache

.PHONY: image shell serve profile ext sweep

image:
	docker build -q -f Dockerfile.rocm10 --build-arg PYTORCH_ROCM_ARCH=$(ARCH) -t $(IMAGE) .
	@echo "built $(IMAGE)"

shell:
	docker run --rm -it $(COMMON) $(IMAGE) bash

serve:
	docker run --rm -it -p $(PORT):9001 $(COMMON) $(IMAGE) python3 /opt/exllamav3/serve_rocm.py

profile: MODE ?= kernel
profile:
	@[ "$(MODE)" = "kernel" -o "$(MODE)" = "api" -o "$(MODE)" = "pmc" ] || \
		(echo "usage: make profile MODE=kernel|api|pmc" >&2; exit 2)
	$(eval PROF_RUN := $(PROF_OUT)/$(MODE)-$(shell date +%Y%m%d-%H%M%S))
	mkdir -p $(PROF_RUN)
	docker run --rm -it -e OUT=/profiling -v $(PROF_RUN):/profiling \
		$(COMMON) $(IMAGE) ./profile_rocm.sh $(MODE)
	@echo "profile output: $(PROF_RUN)"

# Rebuild the JIT extension explicitly (e.g. after touching C++ sources without
# restarting, or to warm the cache before a timed run).
ext:
	docker run --rm -it -e EXLLAMA_VERBOSE=1 $(COMMON) $(IMAGE) \
		python3 -c "import exllamav3; print('exllamav3_ext OK')"

sweep:
	rm -rf $(PROF_OUT)
	docker container prune -f
	docker image prune -f
	docker builder prune -f --keep-storage 20GB
	@echo "profiling outputs removed; docker pruned"