# Training on Azure

One script creates a GPU VM, trains every model, pulls the results back, and
deletes the VM.

```bash
az login
./deploy/azure/run_training.sh                       # A100, eastus
./deploy/azure/run_training.sh Standard_NC4as_T4_v3  # cheaper T4
```

## Check quota first

This is the gate that decides whether Azure is usable at all. Free, student and
credit-only subscriptions frequently ship with GPU quota set to **zero**:

```bash
az vm list-usage --location eastus -o table | grep -iE 'NC|ND|NV'
```

A `Limit` of 0 for the family you want means the VM cannot be created. Request
an increase under Subscription → Usage + quotas, and expect one to two business
days. Until then, Colab is free and immediate.

## Choosing a size

With mixed precision enabled, fine-tuning both BERT variants is roughly ten
minutes of GPU work. Provisioning and setup take longer than the training, so
the fastest card is not always worth it.

| Size | GPU | ~$/hour | Notes |
|---|---|---:|---|
| `Standard_NC4as_T4_v3` | T4 16GB | ~0.53 | Cheapest. Fine at batch 16. |
| `Standard_NC6s_v3` | V100 16GB | ~3.06 | Middle ground. |
| `Standard_NC24ads_A100_v4` | A100 80GB | ~3.67 | Fastest. Default here. Allows batch 32. |

Prices vary by region; treat them as order-of-magnitude. The whole run costs
about a dollar on any of them, because it lasts minutes.

## The VM deletes itself

The script traps `EXIT`, so the resource group is deleted on success, on
failure, and on Ctrl-C alike.

> This is the part to keep if you rewrite the script. A GPU left running
> overnight costs more than the training run; one left for a week costs more
> than the project.

Verify nothing survives:

```bash
az group list -o table
az vm list -d -o table
```

## Why the Ubuntu HPC image

`microsoft-dsvm:ubuntu-hpc:2204` ships with the NVIDIA driver and CUDA already
installed. Installing them by hand adds ten minutes and a reboot to a job that
only runs for ten.

## What the run produces

Trained models for BERT, Legal-BERT and the CNN baseline, each evaluated at
window and document level, plus `docs/results.md` generated from the runs
themselves. All of it is copied back into `outputs/` before the VM dies.

Push the models to the Hub afterwards with `legal-risk-export-hf`
([chapter 11](../../handbook/11-deployment.md)).
