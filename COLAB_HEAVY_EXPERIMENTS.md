# Colab/A100 Heavy Experiments

This file is for experiments that should not be run on the local 16GB GPU.

## 1. QASPER Long Retrieval

Runnable script:

```bash
bash scripts/colab_qasper_long_retrieval.sh
```

Recommended Colab command:

```bash
!git clone <your-lsso-repo-url> /content/LSSO
%cd /content/LSSO
!SEEDS="1 2 3" DOC_LENS="1024" MODELS="mha lsso16 lsso32" bash scripts/colab_qasper_long_retrieval.sh
```

Long-context extension:

```bash
!SEEDS="1 2 3" DOC_LENS="1024 2048" MODELS="mha lsso16 lsso32" BATCH_SIZE=8 GRAD_ACCUM=4 bash scripts/colab_qasper_long_retrieval.sh
```

Outputs default to:

```text
/content/drive/MyDrive/lsso_qasper_long_runs
```

if Google Drive is mounted, otherwise:

```text
/content/lsso_qasper_long_runs
```

The script writes `.done` markers, so rerunning it skips completed jobs.

## 2. Not Yet In This Colab File

These are still paper-plan items, but this repo does not yet have clean official training entrypoints for them:

```text
LRA Text/Retrieval/ListOps
MS MARCO -> BEIR transfer
ImageNet-100 classification
official BiMamba retrieval baseline if mamba-ssm install fails
```

Do not fake these with ad hoc implementations. Add dedicated scripts before using them for paper tables.
