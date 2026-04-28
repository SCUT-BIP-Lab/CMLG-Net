# CMLG-Net: Cross-Modality Local-Global Network for Robust Hand Gesture Authentication

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TIFS-blue)](https://ieeexplore.ieee.org/document/xxxxxxxx)
[![Dataset](https://img.shields.io/badge/Dataset-SCUT--DHGA--br-green)](https://github.com/SCUT-BIP-Lab)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Official PyTorch implementation of **CMLG-Net** (Cross-modality Local-Global Network), a two-stream CNN framework for robust and accurate dynamic hand gesture authentication under challenging real-world conditions.

> **Yufeng Zhang, Wenxiong Kang, and Wenwei Song**  
> *IEEE Transactions on Information Forensics and Security (TIFS), 2024*

---

## Overview

Robust fine-grained behavioral features are critical for dynamic hand gesture authentication. However, behavioral characteristics are abstract and complex, making them harder to capture than physiological traits. Moreover, varying illumination and backgrounds in practical applications pose additional challenges to conventional RGB-based methods.  

**CMLG-Net** addresses these issues with two complementary modules:

- **Temporal Scale Pyramid (TSP)**: Captures fine-grained local motion cues at multiple temporal scales using parallel convolutions with different kernel sizes.
- **Cross-Modality Temporal Non-Local (CMTNL)**: Aggregates global temporal features and cross-modality (RGB-D) information via an attention mechanism.

Together, these modules produce a comprehensive and robust behavioral representation that combines multi-scale (short- and long-term) and multimodal (RGB-D) information.

![CMLG-Net Architecture](https://raw.githubusercontent.com/SCUT-BIP-Lab/CMLG-Net/master/img/CMLGNet.png)

*Overall architecture of CMLG-Net. It contains independent RGB and depth branches with a shared design. Each branch uses a pretrained ResNet18 backbone, inserts a TSP module at Conv1 with a residual connection, and concludes with the CMTNL module to summarize global temporal and multimodal features. Finally, enhanced features from both branches are concatenated for the final identity representation.*

---

## Key Features

- ✅ **State-of-the-art accuracy** – achieves 0.497% EER on SCUT-DHGA and 4.848% on SCUT-DHGA-br under challenging protocols
- ✅ **Robust to illumination & background changes** – explicitly designed for real-world scenes
- ✅ **Multi-scale temporal modeling** – TSP module captures local motion at various scales, and CMTNL module captures long-term motion patterns
- ✅ **Multimodal learning** – leverages both RGB and depth modalities with cross-modality attention

---

## Method Highlights

### 1. Temporal Scale Pyramid (TSP) Module
- Multiple parallel convolution branches with different temporal kernel sizes
- Captures fine-grained local motion cues across short, medium, and long temporal scales
- Inserted at Conv1 block with a residual connection

### 2. Cross-Modality Temporal Non-Local (CMTNL) Module
- Aggregates global temporal features and cross-modality (RGB-D) information

- Composed of two sub-modules: UME (Unimodal Enhancement) and CMI (Cross-Modality Integration)

### 3. Adaptive multimodal Fusion
 - Fuse the RGB and depth modalities using dynamic weights based on their quality
- Three independent loss functions supervise unimodal (RGB, depth) and multimodal identity features
- Gradient from CMTNL is blocked during back-propagation, allowing each branch to focus on modality-specific features without interference

---

## Dataset: SCUT-DHGA-br

SCUT-DHGA-br is a challenging derived dataset designed to evaluate robustness under practical conditions. It is built by applying **background replacement** and **lighting adjustment** to the testing set of the original SCUT-DHGA dataset. This dataset can be downloaded from [SCUT-DHGA-br](https://github.com/SCUT-BIP-Lab/SCUT-DHGA-br).

| Feature | Description |
|---------|-------------|
| Total backgrounds | 3,627 (airports, classrooms, malls, museums, subways, etc.) |
| Brightness factor | Random between 0.5 and 1 |
| Purpose | Simulate real-world illumination and background variations |

![SCUT-DHGA-br Examples](https://raw.githubusercontent.com/SCUT-BIP-Lab/CMLG-Net/master/img/SCUT-DHGA-br.png)

---

## Results

### Accuracy on SCUT-DHGA (EER %)

CMLG-Net is compared with 27 state-of-the-art methods, including 2D CNNs, 3D CNNs, symbiotic CNNs, and two-stream CNNs. It achieves the best EER under both UMG and RMG protocols.

![SCUT-DHGA Results](https://raw.githubusercontent.com/SCUT-BIP-Lab/CMLG-Net/master/img/SCUT-DHGA%20experiments.png)

### Accuracy and Efficiency on SCUT-DHGA-br

CMLG-Net significantly outperforms other approaches and meets real-time requirements on GPU devices.

![SCUT-DHGA Results](https://raw.githubusercontent.com/SCUT-BIP-Lab/CMLG-Net/master/img/SCUT-DHGA-br%20experiments.png)

---

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch ≥ 2.4.0

### Installation

```bash
git clone https://github.com/SCUT-BIP-Lab/CMLG-Net.git
cd CMLG-Net
pip install -r requirements.txt
```

### Data Preparation
1. Download the SCUT-DHGA dataset (original) and the SCUT-DHGA-br dataset (derived).

2. Organize the data as follows:
```text
data/
├── SCUT-DHGA/
│   ├── color_hand/     # RGB iamges
│   ├── depth_hand/     # Depth images
└── SCUT-DHGA-br/
    ├── color_hand/
    └── depth_hand/
```

### Training
```bash
# Train CMLG-Net on SCUT-DHGA under UMG protocol
python ./train.py --conf_file ./conf/CMLGNet/UMG/UMG1_SD_CMLGNet.conf --mode train
```

### Evaluation
```bash
# Evaluate CMLG-Net on SCUT-DHGA under UMG protocol
python ./train.py --conf_file ./conf/SSAF/UMG/UMG1_SD_CMLGNet.conf --mode eval
```
### Citation
If you find this work useful, please cite:

```bibtex
@ARTICLE{zhang2024cmlg,
  author={Zhang, Yufeng and Kang, Wenxiong and Song, Wenwei},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Robust and Accurate Hand Gesture Authentication With Cross-Modality Local-Global Behavior Analysis}, 
  year={2024},
  volume={19},
  number={},
  pages={8630-8643},
  keywords={Authentication;Videos;Feature extraction;Physiology;Robustness;Lighting;Spatiotemporal phenomena;Biometrics;hand gesture authentication;multimodal fusion;spatiotemporal analysis;behavioral characteristic representation},
  doi={10.1109/TIFS.2024.3451367}}
}
```

### Contact
**Biometrics and Intelligence Perception Lab**  
College of Automation Science and Engineering  
South China University of Technology, Guangzhou, China  

- **Yufeng Zhang**: auyfzhang@mail.scut.edu.cn  
- **Wenxiong Kang**: auwxkang@scut.edu.cn  

### License
MIT License. See [LICENSE](https://github.com/SCUT-BIP-Lab/CMLG-Net/blob/main/LICENSE) for details.
