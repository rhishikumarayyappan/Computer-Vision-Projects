# U-Net for Image Segmentation
Author: Rhishi Kumar Ayyappan

### Project Overview

**Business Challenge:**
While object detection places a box *around* an object, many advanced AI tasks require a precise, pixel-level understanding of an object's exact shape. This project demonstrates the ability to build and train a U-Net, the foundational architecture for semantic segmentation, which is critical for fields like medical imaging (tumor/organ segmentation) and autonomous systems (road/lane segmentation).

**Key Achievements & Metrics**
* **Technical Depth:** Built a complete U-Net architecture from scratch, including all encoder (downsampling) and decoder (upsampling) blocks with skip-connections.
* **High Performance:** Achieved ~89% validation accuracy after 20 epochs.
* **Stable Training:** Loss curves (see below) show stable, non-overfitting training, proving the model generalizes well.
* **Reproducible Pipeline:** Used the `tensorflow-datasets` library to create a 100% reliable and reproducible data-loading and preprocessing pipeline.

---

### Methods Used

* **Data:** `oxford_iiit_pet:4.0.0` (from TensorFlow Datasets).
* **Model:** U-Net, built from scratch using the Keras Functional API.
* **Workflow:**
    1.  Loaded and preprocessed data (resizing, normalization).
    2.  Built the U-Net model by defining reusable `downsample_block` and `upsample_block` functions.
    3.  Trained the model for 20 epochs on a Colab T4 GPU.
* **Evaluation:** Plotted training vs. validation accuracy and loss. Performed qualitative validation by visually comparing the model's "Predicted Mask" against the "True Mask".

---

### Business Impact

* **Demonstrates Architectural Competence:** This project proves a fundamental understanding of deep learning architectures beyond just using a pre-built library. It shows the ability to implement a complex, published architecture (U-Net) from scratch.
* **Unlocks High-Precision Tasks:** This workflow is the foundation for any task requiring pixel-level precision. This skill is directly applicable to high-value industries like:
    * **Medical AI:** Segmenting tumors, organs, or cells from MRI/CT scans.
    * **Geospatial Analysis:** Classifying land use from satellite imagery.
    * **Autonomous Vehicles:** Identifying the exact shape of roads, lanes, and pedestrians.

---

### Visuals

#### Model Performance
The model training was stable, with validation accuracy and loss tracking closely with training, indicating a well-built model that generalizes effectively.

![Training and Validation Graphs](assets/loss_and_accuracy.png)

#### Inference Results
The model successfully learned to predict the pixel-level masks for various pets, closely matching the ground truth.

![Model Predictions](assets/results.png)

---

### How to Run

1.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Launch notebook:**
    * Open `U_Net_for_Image_Segmentation.ipynb` in Google Colab (recommended) or Jupyter.
    * Run all cells. The notebook will automatically download the `oxford_iiit_pet` dataset from TensorFlow Datasets and begin training.

---

### Tech Stack

* Python
* TensorFlow 2.x / Keras
* TensorFlow Datasets (TFDS)
* Matplotlib
* Seaborn
* NumPy
