🧠 MNIST GAN – Handwritten Digit Generation

This project implements a Generative Adversarial Network (GAN) using PyTorch to generate realistic handwritten digits from the MNIST dataset.
It includes CNN-based architecture improvements, visualizations, and GIF-based progress tracking for a clear look at how the model learns over time.


🚀 Features

🧩 Modular Codebase — separate data loader (data_utils.py) and GAN training script (mnist_gan.py).

🧠 GAN Implementation — trained on MNIST for handwritten digit generation.

🎞️ GIF Visualization — automatically generates a training progress animation.

💾 Model Saving — generator and discriminator checkpoints stored after training.

🖼️ Sample Preview — view real and generated digits directly.


⚙️ Installation

git clone https://github.com/<your-username>/mnist-gan.git

cd mnist-gan

pip install -r requirements.txt


🏋️ Training the GAN

python mnist_gan.py --epochs 50 --batch_size 128


This will:

Download and load the MNIST dataset

Train the GAN

Save generated images after each epoch

Create an animated GIF showing progress



👩‍💻 Author

Sanskriti - AI Student passionate about deep learning and generative models.
