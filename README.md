# AMF-BO

### AMF-BO: Adaptive Multimodal Fusion with Balanced Optimization for Microservice Failure Diagnosis

*AMF-BO* is designed to achieve accurate failure diagnosis in microservice-based systems. It leverages three types of monitoring data—metrics, traces, and logs—to accomplish two core diagnostic tasks:
- Root Cause Localization
- Failure Type Identification
This repository offers the core implementation of *AMF-BO*.

![](./amfbo/structure.jpg)

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

## Dataset

We evaluate AMF-BO on three public benchmark datasets covering logs, metrics, and traces.
- **D1 (GAIA)** – Collected from a QR‑code login system. Available at [GAIA](https://github.com/CloudWise-OpenSource/GAIA-DataSet).
- **D2 (AIOps‑22)** – From the 2022 CCF AIOps Challenge, based on HipsterShop. Download from the [challenge website](https://competition.aiops-challenge.com).
- **D3 (SockShop)** – Deployed on Kubernetes with ChaosMesh injections. The benchmark is from [microservices‑demo](https://github.com/microservices-demo/microservices-demo); our processed version is included in the `data/` directory of this repository.


## Getting Started

<B>Requirements</B>
- python=3.8.12
- pytorch=2.1.1
- fasttext=0.9.2
- dgl=2.1.0

<B>Run</B>
You can run the below commands:
```python
python main.py
```
The meanings of some parameters in `amfbo/config` are as follows:
- `dataset`: The dataset that you want to use, e.g., gaia, aiops22, sockshop.
- `alert_embedding_dim`: Dimensionality of the FastText event embeddings, passed to both event encoding and the model encoder. (Default: 128)
- `graph_layers`: Number of GraphSAGE layers in the graph encoder. Controls how far failure signals propagate through the service dependency graph. (Default: 2)
- `lr`: Learning rate for the Adam optimizer. (Default: 0.001)
- `weight_decay`: L2 regularization strength for Adam. (Default: 0.0001)
- `patience`: The number of consecutive epochs allowed without improvement before training stops early. (Default: 10)
- `alpha`: A balancing factor that controls the weights of Information Uniqueness and Fusion Sensitivity in weighted fusion.
