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

```python
!pip install -q uv
!git clone <repo-url> thermground
%cd thermground
!uv pip install --system -e .
```

Place datasets under `data/` (for example by mounting Google Drive and symlinking it); the directory is not tracked by git.
