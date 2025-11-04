# Image Captioning with a Custom Transformer

**Author:** Rhishi Kumar Ayyappan

---

## Project Overview

**Business Challenge:**
Manually captioning large volumes of images is costly and time-consuming. This prevents businesses from making their content searchable, accessible (for visually impaired users), and engaging. This is a critical bottleneck for e-commerce, digital asset management, and web accessibility compliance.

This project implements a complete, end-to-end image captioning model that automatically generates descriptive, human-like captions for any given image.

---

## Key Achievements & Metrics

-   **Custom Transformer Decoder:** Successfully built a complete Transformer Decoder **from scratch** using Keras, demonstrating a deep understanding of modern attention-based architectures.
-   **Encoder-Decoder Pipeline:** Integrated the custom decoder with a **pre-trained, frozen EfficientNetB0** encoder, effectively leveraging transfer learning for powerful image feature extraction.
-   **Successful Training:** The model was trained successfully, as evidenced by the smooth convergence of training/validation loss and accuracy curves.
-   **High-Quality Predictions:** The final model generates contextually accurate and grammatically sound captions for images from the test set, as shown in the visual examples.

---

## Methods Used

-   **Data:** Flickr8k dataset (a standard benchmark for this task).
-   **Image Encoder:** A pre-trained **EfficientNetB0** model (frozen) to extract rich, semantic feature vectors from images.
-   **Text Decoder:** A **Transformer Decoder built from scratch**, including:
    * Token & Positional Embeddings
    * Masked Multi-Head Attention Layers
    * Encoder-Decoder Cross-Attention
-   **Text Processing:** A Keras `TextVectorization` layer to create a robust vocabulary and handle tokenization, padding, and `start`/`end` tokens.
-   **Training:**
    * **Optimizer:** Adam
    * **Loss:** Sparse Categorical Crossentropy
    * **Pipeline:** An efficient `tf.data` pipeline for batching and prefetching.

---

## Business Impact

-   **Accessibility (WCAG):** This model can be deployed to **automatically generate alt-text** for all images on a website, ensuring compliance with accessibility standards and making content available to visually impaired users.
-   **E-commerce & SEO:** Automates the creation of product descriptions and image metadata. This **improves SEO ranking** and makes product catalogs **more searchable** for customers.
-   **Digital Asset Management (DAM):** Enables businesses to catalog and search vast, unstructured image libraries using natural language queries, saving thousands of man-hours in manual tagging.

---

## Visuals

-   **Training & Validation Curves:** Plots demonstrating the model's learning progress and convergence over time.
-   **Sample Prediction:** An example of a generated caption from the trained model on a test image.

![Training & Validation Curves](images/loss_and_accuracy.png)
![Sample Prediction](images/predictions.png)

---

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/rhishikumarayyappan/Computer-Vision-Projects.git](https://github.com/rhishikumarayyappan/Computer-Vision-Projects.git)
    cd Computer-Vision-Projects/Image_Captioning_with_Transformers
    ```
    *(Note: Adjust folder name if needed)*

2.  **Install requirements:**
    ```bash
    pip install tensorflow seaborn matplotlib
    ```
    *(You can also create a `requirements.txt` file with these packages)*

3.  **Launch the notebook:**
    ```bash
    jupyter notebook Image_Captioning_with_Transformers_.ipynb
    ```

4.  **Run all cells** (Runtime -> Run all). The notebook is designed to automatically:
    * Download and unzip the Flickr8k dataset.
    * Build the encoder and decoder.
    * Train the model.
    * Display the results and sample predictions.

---

## Tech Stack

-   Python
-   TensorFlow
-   Keras (Functional & Subclassing API)
-   EfficientNet (Transfer Learning)
-   Seaborn & Matplotlib (for visualizations)
-   Google Colab (for GPU-accelerated training)

---

**For the full implementation—including the from-scratch Transformer build—see the included Jupyter Notebook!**
