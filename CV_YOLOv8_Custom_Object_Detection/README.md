# YOLOv8 Object Detection & Deployment Demo
Author: Rhishi Kumar Ayyappan

### Project Overview

**Business Challenge:**
Real-time, accurate object detection is a fundamental component for countless AI applications, from autonomous systems and retail analytics to safety monitoring. This project demonstrates the complete end-to-end workflow for training, evaluating, and deploying a state-of-the-art YOLOv8 model.

**Key Achievements & Metrics**
* **mAP50:** 0.846
* **mAP50-95:** 0.677
* **Model:** YOLOv8-Nano (a lightweight, fast model)
* **Deployment:** Interactive Gradio web app for live inference.

---

### Methods Used

* **Data:** `COCO128` (a standard benchmark subset of the MS COCO dataset).
* **Model:** YOLOv8-Nano (`yolov8n.pt`) using Transfer Learning.
* **Workflow:** Fine-tuning for 50 epochs, evaluation using mAP, and deployment with Gradio.
* **Evaluation:** Mean Average Precision (mAP), Precision/Recall curves, Loss Curves, and Confusion Matrix.

---

### Business Impact

* **Production-Ready Prototype:** The project is not just a notebook; it includes a standalone `app.py` script that serves the model in a user-friendly web interface, proving its utility beyond a static analysis.
* **Versatile & Scalable Workflow:** This end-to-end pipeline serves as a robust template that can be rapidly adapted for any custom business problem (e.g., detecting safety-helmet violations, monitoring retail shelf inventory, or finding manufacturing defects).
* **Quantitative Benchmarking:** The model achieved a **mAP50 of 0.846**, providing a strong, measurable baseline. This proves the workflow is effective before applying it to more complex, custom data.

---

### Visuals

#### Live Demo Application
An interactive Gradio web app allows any user to upload an image and receive real-time predictions.

![Gradio Demo Screenshot](assets/gradio_demo.png)

#### Model Performance & Inference
The model was trained for 50 epochs and validated on a test split.

| Training & Validation Loss | Confusion Matrix |
| :---: | :---: |
| ![Training Results](assets/training_results.jpg) | ![Confusion Matrix](assets/confusion_matrix.jpg) |

**Inference Examples:**
| `zidane.jpg` | `bus.jpg` |
| :---: | :---: |
| ![Zidane Prediction](assets/sample2.jpg) | ![Bus Prediction](assets/sample1.jpg) |

---

### How to Run

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/rhishikumarayyappan/CV.git](https://github.com/rhishikumarayyappan/CV.git)
    cd CV/CV_YOLOv8_Custom_Object_Detection
    ```
2.  **Set up a virtual environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Live Demo:**
    ```bash
    python app.py
    ```
    A public URL will be generated. Open it in your browser to upload your own images.

5.  **Explore the Training:**
    To see how the model was trained, open and run the `YOLOv8_Custom_Object_Detection_with_Gradio.ipynb` notebook in Google Colab.

---

### Tech Stack

* Python
* Ultralytics (YOLOv8)
* Gradio
* OpenCV
* Matplotlib
* NumPy
