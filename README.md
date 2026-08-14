# AMFBO

AMF-BO: Adaptive Multimodal Fusion with Balanced Optimization for Microservice Failure Diagnosis

The framework leverages three types of monitoring data—metrics, traces, and logs—to accomplish two core diagnostic tasks:
- Root Cause Localization
- Failure Type Identification
## Project Structure

```
AMFBO/
├── amfbo/
│   ├── config/                 # experiment configuration
│   ├── core/                   # training, losses, models, MDC/CAMF/MBO related implementation
│   │   ├── losses/
│   │   └── models/
│   │       ├── backbone/
│   │       └── fusion/
│   ├── extraction/             # raw telemetry extraction and preprocessing
│   │   ├── drain/
│   │   └── utils/
│   ├── pipeline/               # event embedding and dataset building
│   │   └── events/
│   └── utils/                  # general tools
├── data/                       # raw data
├── dataloader/                 # training/testing dataloader
├── logs/                       # training logs
├── result/                     # training results
├── main.py                     # project entrance
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

## frame structure