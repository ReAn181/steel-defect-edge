#!/usr/bin/env bash
# Build the ExecuTorch Llama runner (`llama_main`) natively on a Jetson Orin Nano Developer Kit 8GB
# (aarch64, JetPack 6.2.x = Ubuntu 22.04, system Python 3.10, gcc 11, 128 GB SD card).
# - Needs internet ONCE (git, pip, apt). The finished binary then runs fully offline.
# - Takes several hours on an Orin Nano with 8 GB RAM and an SD card (the build is deliberately limited to a few
#   parallel jobs so it does not run out of memory). Run it inside `tmux` so closing the terminal does not kill it.
# - ET_BRANCH must match the executorch version in llama_deploy/export_info.json (notebook 01): yours is 1.6.0.
# - Written from the ExecuTorch docs; it could not be tested on an Orin Nano before delivery. If a step fails, read the
#   error, fix that step, and re-run: the script is safe to re-run.
set -euo pipefail

ET_BRANCH="${ET_BRANCH:-release/1.6}"
ET_TAG="${ET_TAG:-v1.6.0}"                           # used if the branch name does not exist
WORK="${WORK:-$HOME/et_build}"
REPO_DIR="${REPO_DIR:-$HOME/steel_defect_edge}"      # where this repo lives (the binary is copied to $REPO_DIR/bin)
JOBS="${JOBS:-3}"                                    # parallel compile jobs; 8 GB RAM is tight, raise only if the build is stable
SWAP_GB="${SWAP_GB:-8}"
mkdir -p "$WORK" "$REPO_DIR/bin"
cd "$WORK"
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" MAX_JOBS="$JOBS"

echo "== 0) disk and swap =="
df -h / | tail -1
if ! swapon --show | grep -q /swapfile; then
  # Jetson images already have small zram swap; the build needs more. A swap file on the SD card is slow and wears
  # the card a little, so it is removed at the end of this script (see last lines).
  sudo fallocate -l "${SWAP_GB}G" /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
fi

echo "== 1) system packages =="
sudo apt-get update
sudo apt-get install -y build-essential git curl wget pkg-config libssl-dev python3-pip python3-dev tmux

echo "== 2) Python 3.11 environment (JetPack's Python 3.10 is too old for ExecuTorch 1.6) =="
if [ ! -d "$HOME/miniforge3" ]; then
  curl -L -o miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
  bash miniforge.sh -b -p "$HOME/miniforge3"
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda create -y -n et311 python=3.11 || true
conda activate et311
pip install --upgrade pip cmake ninja     # Ubuntu 22.04's apt cmake (3.22) is older than ExecuTorch needs

echo "== 3) ExecuTorch source ($ET_BRANCH, fallback tag $ET_TAG) =="
if [ ! -d executorch ]; then
  git clone --branch "$ET_BRANCH" https://github.com/pytorch/executorch.git \
    || git clone --branch "$ET_TAG" https://github.com/pytorch/executorch.git
fi
cd executorch
git submodule sync && git submodule update --init --recursive
./install_executorch.sh

echo "== 4) build the Llama runner (CPU / XNNPACK) =="
make llama-cpu
# If `make llama-cpu` does not exist on your branch, use the workflow presets instead:
#   cmake --workflow --preset llm-release
#   cd examples/models/llama && cmake --workflow --preset llama-cpu

BIN=cmake-out/examples/models/llama/llama_main
ls -lh "$BIN"
cp "$BIN" "$REPO_DIR/bin/llama_main"
find cmake-out -name 'libllama_runner*.so*' -exec cp {} "$REPO_DIR/bin/" \; || true

echo "== 5) remove the temporary swap file (saves SD card space and wear) =="
sudo swapoff /swapfile 2>/dev/null && sudo rm -f /swapfile || true

echo "Done. Runner copied to $REPO_DIR/bin/llama_main"
echo "Next: copy llama_deploy/ (the .pte and tokenizer.model) into $REPO_DIR, then run notebook 02's checker."
