# Video Frame Prediction with Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Predicting future frames in video sequences using **LSTM**, **PredRNN**, and **Transformers** models built from scratch in PyTorch.

## 📖 Overview
This project explores deep learning architectures to predict multiple consecutive frames in video sequences. The models (LSTM, PredRNN, Transformers) are implemented from scratch using PyTorch and tested on synthetic or real-world video datasets.

## 🧠 Models Implemented
1. **LSTM**: Sequential modeling for temporal dependencies.
2. **PredRNN**: Combines ConvLSTM with gradient highway units for spatiotemporal dynamics.
3. **Transformers**: Leverages self-attention mechanisms for long-range dependencies.

## ✨ Features
- Models built from scratch using PyTorch.
- Support for multi-step frame prediction.
- Modular code structure for easy experimentation.
- Comparative analysis of model performance.

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Video-Frame-Prediction-DL.git
   cd Video-Frame-Prediction-DL
2. Install dependencies:

        pip install numpy pandas torch matplotlib opencv-python
## 🚀 Usage
Training

      python train.py --model lstm --epochs 50 --batch_size 32
      python train.py --model predrnn --epochs 50 --batch_size 32
      python train.py --model transformer --epochs 50 --batch_size 32
## 📄 Report
For a detailed explanation of the methodology, experiments, and results, see the Project Report.




## 📁 ## Link to Models File  (Couldn't upload due to space issues)

### Download Models
[![Google Drive](https://img.shields.io/badge/Google%20Drive-Dataset-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/uc?export=download&id=1ray4Bjd9RqveiK_5sE5OgG9--GWU9XHQ)

**Direct Links**  
- [View File](https://drive.google.com/file/d/1ray4Bjd9RqveiK_5sE5OgG9--GWU9XHQ/view?usp=drive_link)
- [Download Direct](https://drive.google.com/uc?export=download&id=1ray4Bjd9RqveiK_5sE5OgG9--GWU9XHQ)

⚠️ **Note:** Ensure you have access permissions before downloading.
