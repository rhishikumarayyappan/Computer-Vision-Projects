# Image Captioning with a CNN & Transformer
Author: Rhishi Kumar Ayyappan

### Project Overview

**Business Challenge:**
Beyond simple classification, modern AI must understand the *relationship* between visual data (images) and language (text). This project tackles this multi-modal challenge by building and training a model that can generate a human-like, descriptive caption for any given image.

**Key Achievements & Metrics**
* **Multi-Modal Architecture:** Successfully integrated a pre-trained **CNN Encoder (EfficientNetB0)** for image understanding and a **Transformer Decoder** for text generation.
* **Built From Scratch:** The core Transformer Decoder layer (including multi-head attention and positional embeddings) was built from scratch using Keras.
* **Rapid Prototyping:** Demonstrated a fast, feasible training loop on a free-tier GPU by pre-computing image features. The model shows clear signs of learning (decreasing loss, increasing accuracy) in just 10 epochs.
* **Generative Inference:** Implemented a "greedy search" inference loop to generate new, unseen captions word-by-word.

---

### Methods Used

* **Data:** Flickr8k dataset (8,000+ images, each with 5 unique captions).
* **Data Pipeline:**
    1.  Downloaded and parsed the raw image and text files.
    2.  Used the official `train/test` split files to ensure data integrity.
    3.  Used `tf.keras.TextVectorization` to build a vocabulary and tokenize all captions.
* **Encoder (Image Model):** A pre-trained, **frozen EfficientNetB0** was used as a powerful feature extractor.
* **Decoder (Language Model):** A custom-built **Transformer Decoder** was trained to map the image features to a sequence of text tokens.
* **Workflow:**
    1.  **Pre-computing:** All 8,000+ image features were pre-computed *once* to create a fast, efficient training pipeline.
    2.  **Training:** The decoder was trained on the pre-computed features. Training on a small subset (10 epochs, 50 steps/epoch) proves the pipeline's viability.
    3.  **Inference:** A greedy-search loop generates captions by feeding the model its own previous word prediction until an `<end>` token is produced.

---

### Business Impact

* **Demonstrates Advanced AI Skill:** This project proves proficiency in multi-modal architectures (CV + NLP) and Transformers, which are the foundation of modern Generative AI.
* **Foundation for SOTA Applications:** This architecture is the basis for more advanced tasks like Visual Question Answering (VQA), image-based search, and accessibility tools for the visually impaired.
* **Efficient Prototyping:** The pre-computation workflow demonstrates a key MLOps technique for rapid experimentation and fast training cycles, even with large image datasets.

---

### Visuals

#### Model Performance
The model was trained for 10 epochs (at 50 steps/epoch). The loss curves clearly show the model is learning and not overfitting, successfully validating the architecture and pipeline.

![Training and Validation Graphs](assets/loss_and_accuracy.png)

#### Inference Results
The model generates contextually relevant (though not perfect) captions for unseen test images, proving it learned the relationship between image features and language.

![Model Predictions](assets/predictions.jpg)

---

### How to Run

1.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Launch notebook:**
    * Open `Image_Captioning_with_Transformers_.ipynb` in Google Colab (recommended) or Jupyter.
    * Run all cells. The notebook will:
        1.  Download and unzip the 1.1GB Flickr8k dataset.
        2.  Pre-compute all image features (takes ~2-3 minutes).
        3.  Train the decoder model (takes ~2-3 minutes).
        4.  Display the results.

---

### Tech Stack

* Python
* TensorFlow 2.x / Keras
* NumPy
* Seaborn
* Matplotlib
* OpenCV
* TQDM
