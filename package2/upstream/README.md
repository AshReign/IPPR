# Temporal post-processing for rip current segmentation (42177 A2, Group 12)

## Setup
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt       # then install torch for your machine
copy configs\paths.example.yaml configs\paths.yaml   # edit your paths
pytest
```

## Data
RipVIS v1.8.4 (CC BY-NC 4.0). Never commit dataset files; only manifests in `data/manifests/`.