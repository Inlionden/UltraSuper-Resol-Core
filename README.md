# Image Super-Resolution with SRGAN in PyTorch 🚀

This repository contains a PyTorch implementation of the **Super-Resolution Generative Adversarial Network (SRGAN)**. The goal of this project is to upscale low-resolution (LR) images to high-resolution (HR) counterparts that are both sharp and photorealistic.

This implementation is designed to be a clear and educational reference for students and developers interested in computer vision and generative models.

## 📝 Project Overview

Traditional super-resolution methods (like bicubic interpolation) often produce blurry and overly smooth images. SRGAN tackles this by training a deep neural network in an adversarial setting.

The key components are:
* A **Generator** network that learns to create convincing high-resolution images.
* A **Discriminator** network that learns to distinguish between real HR images and the fake super-resolved (SR) images from the generator.
* A **Perceptual Loss** function (using a pre-trained VGG network) that prioritizes photo-realistic details over simple pixel-wise accuracy.

This combination pushes the Generator to "hallucinate" fine textures and details, resulting in images that are perceptually superior to those produced by models trained on MSE loss alone.

---

## ✨ Features

* **SRGAN Architecture**: A complete implementation of the SRGAN generator and discriminator.
* **Perceptual Loss**: Utilizes a VGG19 feature extractor for content loss, ensuring high perceptual quality.
* **Generator Pre-training**: Includes a crucial pre-training step for the generator with pixel-wise L1 loss to stabilize the main adversarial training.
* **On-the-Fly Preprocessing**: LR images are generated from HR images during training, providing robust data augmentation.
* **Evaluation**: Calculates **Peak Signal-to-Noise Ratio (PSNR)** and **Structural Similarity Index (SSIM)** for quantitative assessment.
* **Modular Code**: The notebook is structured into 9 logical sections, from data loading to evaluation and deployment concepts.

---

## 🛠️ Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

You will need Python 3.8+ and the following libraries:

- PyTorch
- Torchvision
- TorchMetrics
- Pillow
- Matplotlib
- NumPy
- Kaggle (for downloading the dataset)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install the required packages:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## 📂 Project Structure

```text
super_resolution/
├── notebooks/              # Jupyter notebooks
│   └── super-resolution.ipynb
├── src/                    # Source code modules
│   ├── __init__.py
│   └── utils.py
├── .gitignore              # Git ignore rules
├── README.md               # Project documentation
├── DATASET.md              # Dataset metadata
├── LICENSE                 # MIT License
├── CONTRIBUTING.md         # Contribution guidelines
└── requirements.txt        # Python dependencies
```

### Dataset

This project uses the **DIV2K dataset**. See [DATASET.md](DATASET.md) for details.

1.  **Download**: You can download it from the [Kaggle DIV2K Dataset page](https://www.kaggle.com/datasets/eesungkim/div2k-dataset). You will need a `kaggle.json` API token.
2.  **Setup**: Unzip the dataset and place it in a directory structure that the notebook can access. Update the `DATA_DIR_HR` paths in the notebook to point to your local data directory.

---

## 🚀 Usage

1.  **Configure Paths**: Open the [notebooks/super-resolution.ipynb](notebooks/super-resolution.ipynb) notebook and ensure the paths in **Cell 2** point to your downloaded DIV2K dataset.
2.  **Run the Notebook**: Execute the cells of the notebook sequentially.
    * The notebook will first pre-train the generator.
    * Then, it will start the main adversarial training loop for the specified number of epochs.
3.  **Monitor Training**: The training progress, including generator and discriminator losses, will be printed periodically.
4.  **Get Results**: After training is complete, the final cells will:
    * Run evaluation and visualization on a sample from the validation set.
    * Save the trained generator's weights to a file named `srgan_generator.pth`.

---

## 📊 Results

The model's performance is evaluated using PSNR and SSIM. After a full training run, the expected output should show a significant visual improvement over the bicubic upscale, with sharper details and more realistic textures.

*(You can add your final PSNR/SSIM scores and a final output image here after running the project).*

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙏 Acknowledgments

* This project is an implementation based on the original paper: [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) by Christian Ledig, et al.
* The [DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) used for training and evaluation.

# auto-commit
