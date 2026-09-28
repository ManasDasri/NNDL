#!/usr/bin/env bash
# Provision a GPU VM, train every model on it, retrieve the results, delete it.
#
# The VM is created and destroyed inside one script on purpose. A GPU left
# running overnight costs more than the entire training run, and a forgotten
# one costs more than the project.
#
#   ./deploy/azure/run_training.sh [size] [region]
set -euo pipefail

SIZE="${1:-Standard_NC24ads_A100_v4}"
REGION="${2:-eastus}"
GROUP="${GROUP:-nndl-training}"
VM="${VM:-nndl-gpu}"
# Ubuntu HPC image: NVIDIA driver and CUDA are preinstalled, which saves the
# 10+ minutes a driver install would otherwise add to a 10 minute job.
IMAGE="microsoft-dsvm:ubuntu-hpc:2204:latest"
REPO="https://github.com/ManasDasri/NNDL.git"
LOCAL_OUT="${LOCAL_OUT:-outputs}"

cleanup() {
  echo
  echo "==> deleting resource group $GROUP (billing stops here)"
  az group delete --name "$GROUP" --yes --no-wait 2>/dev/null || true
}
# Delete on success, failure or Ctrl-C alike.
trap cleanup EXIT

echo "==> $SIZE in $REGION"
az group create --name "$GROUP" --location "$REGION" --output none

az vm create \
  --resource-group "$GROUP" --name "$VM" \
  --image "$IMAGE" --size "$SIZE" \
  --admin-username azureuser --generate-ssh-keys \
  --os-disk-size-gb 128 --public-ip-sku Standard \
  --output none

IP=$(az vm show -d --resource-group "$GROUP" --name "$VM" --query publicIps -o tsv)
echo "==> vm ready at $IP"

ssh-keygen -R "$IP" >/dev/null 2>&1 || true
until ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "azureuser@$IP" true 2>/dev/null; do
  sleep 5
done

ssh -o StrictHostKeyChecking=no "azureuser@$IP" bash -s <<'REMOTE'
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv git >/dev/null
git clone --depth 1 https://github.com/ManasDasri/NNDL.git ~/NNDL
cd ~/NNDL
python3 -m venv .venv
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -e .

./.venv/bin/legal-risk-prepare

for pair in "legal-bert:legal_bert" "bert:bert"; do
  model="${pair%%:*}"; out="${pair##*:}"
  echo "=== training $model ==="
  ./.venv/bin/legal-risk-train-transformer \
      --model "$model" --epochs 3 --batch_size 32 --grad_accumulation 1 \
      --output_dir "outputs/$out"
  ./.venv/bin/legal-risk-evaluate --run_dir "outputs/$out" --split test --pooling max
done

echo "=== training the CNN baseline for the comparison floor ==="
./.venv/bin/legal-risk-train-cnn --epochs 12 --output_dir outputs/cnn
./.venv/bin/legal-risk-evaluate --run_dir outputs/cnn --split test --pooling max

./.venv/bin/legal-risk-compare outputs/cnn outputs/bert outputs/legal_bert
echo "=== REMOTE COMPLETE ==="
REMOTE

echo "==> retrieving results"
mkdir -p "$LOCAL_OUT"
scp -o StrictHostKeyChecking=no -r "azureuser@$IP:~/NNDL/outputs/*" "$LOCAL_OUT/"
scp -o StrictHostKeyChecking=no "azureuser@$IP:~/NNDL/docs/results.md" docs/results.md 2>/dev/null || true

echo "==> done; results in $LOCAL_OUT/"
