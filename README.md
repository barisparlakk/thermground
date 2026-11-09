# thermground

A small vision-language model that answers operator questions on thermal-only UAV images for night search-and-rescue, grounded in visual evidence rather than language priors.

## Setup

### Mac (Apple silicon)

```bash
brew install uv
git clone <repo-url> thermground && cd thermground
uv sync
uv run pytest
```

### Colab (CUDA)

Training and transformers inference run on Colab with the `train` extra. Open
`notebooks/colab_train.ipynb` on a GPU runtime and run it top to bottom: it clones the
repository, installs `.[train,viz]`, downloads HIT-UAV, rebuilds the QA set and runs
training and evaluation. See `docs/training.md`.

Datasets live under `data/`, which is not tracked by git.
