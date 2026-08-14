# AMFBO

AMFBO is a reorganized, adaptive multimodal failure diagnosis framework. It keeps the original diagnosis logic while restructuring the project around the `amfbo` package, making the codebase easier to navigate and extend.

## Project Structure

```
AMFBO/
├── amfbo/
│   ├── config/                 # experiment configuration
│   ├── core/                   # training, augmentation, losses, models
│   │   ├── losses/
│   │   └── models/
│   │       ├── backbone/
│   │       └── fusion/
│   ├── extraction/             # raw telemetry extraction and preprocessing
│   │   ├── drain/
│   │   └── utils/
│   ├── pipeline/               # event embedding and dataset building
│   │   └── events/
│   └── utils/                  # shared helpers
├── data/                       # dataset, unchanged
├── dataloader/                 # cached dataloaders, unchanged
├── logs/                       # training logs, unchanged
├── result/                     # training results, unchanged
├── main.py                     # entry point
└── README.md
```

## Getting Started

Requirements:

- python=3.8.12
- pytorch=2.1.1
- fasttext=0.9.2
- dgl=2.1.0

Run:

```bash
python main.py
```

The main configuration is in `amfbo/config/experiment_config.py`. The `data`, `dataloader`, `logs`, and `result` directories and their contents are intentionally kept untouched.
