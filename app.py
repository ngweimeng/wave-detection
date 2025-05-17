import streamlit as st
from PIL import Image
import os
import glob
import yaml

# Set page config
st.set_page_config(page_title="Surf Pocket Detector", page_icon="🏄")

# Sidebar
st.sidebar.title("📖 Project Links")
st.sidebar.markdown("[View on GitHub](https://github.com/ngweimeng/wave-detection)")
st.sidebar.markdown("💬 Questions or suggestions? Feel free to reach out on my GitHub!")

# Title and Introduction
st.title("Wave Detection for Surfing with Computer Vision")
st.header("Introduction")
st.markdown(
    """
    Every surfer knows the ocean’s mood swings: one day you’re catching the perfect line, the next you’re flattened by a 6‑ft breaker and tumbled like laundry in a washing machine. I’ve spent the last six weeks chasing waves in Bali and Sri Lanka—wipeouts and all—because nothing beats the thrill of carving your own pocket.

    While I dream of landing an aerial 360 or finding that perfect barrel, I realized it all comes down to mastering the fundamentals: positioning yourself in the “pocket” of the wave, where speed and control meet. That spark led me here—to build a computer vision model that can track the pocket frame by frame, helping surfers visualize their best line before they even paddle out.

    This project demonstrates how to break surf footage into individual frames, and use a fine‑tuned YOLOv8‑nano model to predict the pocket in real time. The idea comes from something surfers often do on the beach called “mind surfing”—watching waves from the beach and imagining how to ride them—and now we’re teaching a machine to do the same.
    """
)

# Show wave anatomy diagram
diagram_path = "assets/pocket_eg.jpg"
if os.path.exists(diagram_path):
    st.image(
        diagram_path, caption="Wave Anatomy: The Pocket Zone", use_container_width=True
    )
else:
    st.warning(f"Diagram not found at {diagram_path}.")

st.markdown(
    """
    In the diagram above, the **"pocket"** refers to the steep, powerful zone just ahead of where the wave breaks. This is where surfers aim to stay for optimal performance — and the region that this model is trained to detect.
    """
)

# Data Collection and Preprocessing
st.markdown("---")
st.header("Data Collection and Preprocessing")
st.markdown(
    """
    I collected video clips from Surfline—a popular surf forecasting platform—focusing on the Canggu, Bali beach break. Not only is Canggu my favourite beach in Bali, its mix of left- and right-breaking waves makes it perfect for training a model to recognize wave pockets in both directions.
    """
)

# Show Canggu Beach Image
canggu_img = "assets/canggu_beach.png"
if os.path.exists(canggu_img):
    st.image(
        canggu_img,
        caption="Canggu Beach: left- and right-breaking waves, highlighted in red",
        use_container_width=True,
    )
else:
    st.warning(f"Canggu beach image not found at {canggu_img}.")

st.markdown(
    """
    Using FFmpeg, I converted each MP4 surf clip into a sequence of 1280 × 720 frames at 1 fps. Those frames were then batch-uploaded to Roboflow for annotation. 
    
    Below is the Python snippet from `upload_to_roboflow.ipynb` that loops over the downloaded MP4 files and invokes FFmpeg to extract frames:
    """
)
frame_extract_snippet = """
import os, glob, subprocess

HOME = os.getcwd()
base_dir = f"{HOME}/content/video_to_upload"
output_base = f"{HOME}/content/extracted_frames"
mp4_files = glob.glob(f"{base_dir}/*.mp4")

for filepath in sorted(mp4_files):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    out_dir = os.path.join(output_base, filename)
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "frame_%04d.png")
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", filepath,
        "-vf", "fps=1", "-q:v", "1", "-pix_fmt", "rgb24",
        pattern
    ], check=True)
"""
st.code(frame_extract_snippet)


# Data Labeling
st.markdown("---")
st.header("Data Labeling")
st.markdown(
    """
    First, I labelled every “pocket” in each frame using Roboflow’s annotation tool. In total, I annotated 2,077 frames, marking approximately 3000 pockets. Once all 2077 frames were labeled, I then created a 70/20/10 train/valid/test split of the data. This is a common practice in machine learning to ensure that the model can generalize well to unseen data. The split was done randomly, ensuring that each set had a good representation of the different types of pockets.
    """
)

with st.expander("What’s a train/validation/test split?"):
    st.markdown(
        """
        **Training set** (70%): The model “sees” these examples and adjusts its internal parameters to learn the task.  
        
        **Validation set** (20%): Used during training to check how well the model is doing on unseen data and to fine-tune things like learning rate or architecture.  
        
        **Test set** (10%): Held completely separate until the end—this final slice tells you the true performance on brand-new data.
        """
    )

st.markdown(
    """
    Before export, I ran **Auto‑Orient** on every image. This step removes hidden orientation tags (EXIF data) so that all images display correctly—whether they were taken in portrait or landscape—avoiding accidental mix‑ups during training.

    I then **downsampled** each image to 640×640 pixels. Smaller images load faster and still contain enough detail for pocket detection.

    Finally, I exported everything in **YOLOv8 TXT format**. Each split folder (train/valid/test) has:
    1. An `images/` folder with the `.jpg` files  
    2. A `labels/` folder with matching `.txt` files  
    """
)

# Show YOLOv8 TXT export diagram
diagram_path = "assets/file_directory.png"
if os.path.exists(diagram_path):
    st.image(
        diagram_path,
        caption="Contents of the YOLOv8 PyTorch TXT export",
        use_container_width=True,
    )
else:
    st.warning(f"Diagram not found at {diagram_path}.")

st.markdown(
    """
    Within the `.labels/` folder, each `.txt` file contains one line per bounding box, formatted as:

    ```
    class_id x_center y_center width height
    ```

    Because “pocket” is our only class, the `class_id` is always `0`. All five numbers are normalized between 0 and 1 by dividing the x-center and width by the image’s width, and the y-center and height by its height—ensuring consistent annotations at any resolution.
    
    The `data.yaml` file ties it all together: it points to the train/valid/test folders, sets **nc: 1** for one class, and names that class **pocket**. With our data organized and ready, we can jump into training the model.
    """
)

# Model Development
st.markdown("---")
st.header("Model Development")
st.markdown(
    """    
    I used the YOLOv8-nano model as my pre-trained model, which was trained on the COCO dataset (80 object classes includings dogs, cars and traffic lights). Despite its advanced architecture, many of its layers mirror those in basic convolutional networks.

    To adapt the YOLOv8-nano model for my project, I utilized the pre-trained weights from the COCO dataset, add the pocket class to the model, and then adjust the weights as needed to be able to detect the pocket of a wave in an image. Thanks to Ultralytics’ clear documentation, the custom training script came together quickly.

    Refer to `train_model_with_custom_dataset.ipynb` on the entire script used to train my model.
    """
)

st.markdown(
    """
    When training, I set the task to **detection** and pointed the model argument to `yolov8n.pt`. YOLOv8-nano is the smallest of the five YOLOv8 variants, with 225 layers and roughly 3 million parameters. We trained for 40 epochs, using early stopping with a patience of 10 epochs—so training halts if accuracy doesn’t improve for 10 consecutive rounds. Inputs were resized to 512×512 and processed in batches of 16.

    After fine-tuning, I first checked the performance on our 20% validation split, then ran a final evaluation on the 10% test split using a confidence threshold of 0.3. The resulting weights (`best.pt`) were exported and deployed via the Roboflow API, so I could conduct inference tasks on other video frames.
    """
)

# Performance Analysis
st.markdown("---")
st.header("Performance Analysis")
st.markdown(
    """
    I began by visually comparing the model’s predictions on the validation set with the manual pocket annotations. This side-by-side review below *(left image shows the original manual labels, right image shows the model’s predictions)* illustrated that the model’s bounding boxes closely matched the ground truth with high confidence.    
    """
)

# Relative paths to your images
labels_path = "assets/val_batch0_labels.jpg"
preds_path = "assets/val_batch0_pred.jpg"

# Display ground truth vs. predictions side by side
col1, col2 = st.columns(2)

if os.path.exists(labels_path):
    col1.image(
        labels_path,
        caption="Manual labels: True pocket locations",
        use_container_width=True,
    )
else:
    col1.warning(f"Labels image not found at {labels_path}.")

if os.path.exists(preds_path):
    col2.image(
        preds_path,
        caption="Model predictions: Detected pockets",
        use_container_width=True,
    )
else:
    col2.warning(f"Predictions image not found at {preds_path}.")

# Performance Analysis – Selectable Metrics
st.markdown(
    """
    Next, I’ll dive into the model’s performance metrics. Use the dropdown below to explore: (1) confusion matrix, (2) F1-confidence curve, (3) precision-confidence curve, (4) precision-recall curve (mAP@0.5), and (5) recall-confidence curve.    
    """
)
# Dropdown for metric selection
metric = st.selectbox(
    "Choose a metric to display:",
    [
        "Confusion Matrix",
        "F1-Confidence Curve",
        "Precision-Confidence Curve",
        "Precision-Recall Curve (mAP@0.5)",
        "Recall-Confidence Curve",
    ],
)

# Paths, explanations, and relevance to wave-pocket detection
assets = {
    "Confusion Matrix": (
        "assets/confusion_matrix.png",
        """
        **What it shows:**  
        A 2×2 table of true/false positives and true/false negatives on the validation set.

        **Why it matters:**  
        By seeing that we correctly detect 516 pockets (TP) versus 64 misses (FN) and only 13 false alarms (FP), we understand where the model errs—helping us decide whether to collect more edge-case images or adjust thresholds.
        """,
    ),
    "F1-Confidence Curve": (
        "assets/F1_curve.png",
        """
        **What it shows:**  
        F1 score (harmonic mean of precision and recall) plotted against confidence thresholds.

        **Why it matters:**  
        The peak at 0.93 around threshold 0.38 tells us the sweet spot where pocket detections are both accurate and comprehensive—crucial for live surf-cam inference to avoid missed pockets or spurious boxes.
        """,
    ),
    "Precision-Confidence Curve": (
        "assets/P_curve.png",
        """
        **What it shows:**  
        Precision (correct detections / total detections) across confidence thresholds.

        **Why it matters:**  
        Precision reaching 1.00 by 0.86 confidence means high-confidence predictions are nearly all true pockets. This guides us on safe confidence cutoffs to minimize false alerts during real-time monitoring.
        """,
    ),
    "Precision-Recall Curve (mAP@0.5)": (
        "assets/PR_curve.png",
        """
        **What it shows:**  
        Trade-off between precision and recall, summarized by the area under the curve (mAP@0.5).

        **Why it matters:**  
        An mAP of 0.959 indicates outstanding overall detection quality—meaning the model reliably spots pockets even as we vary detection thresholds, which is key for robust performance in changing wave conditions.
        """,
    ),
    "Recall-Confidence Curve": (
        "assets/R_curve.png",
        """
        **What it shows:**  
        Recall (true positive rate) over confidence thresholds.

        **Why it matters:**  
        Keeping recall above 0.89 below 0.80 confidence means most pockets are caught early. We can tune our threshold to ensure few pockets are missed, which is critical when tracking wave patterns.
        """,
    ),
}

# Display the selected metric with image and tailored explanation
img_path, explanation = assets[metric]
col1, col2 = st.columns([4, 2])
col1.image(img_path, caption=metric, use_container_width=True)
col2.markdown(explanation)

# Data Sample Section
st.markdown("---")
st.header("Real-Time Video Inference Demo")

# First: inference on Canggu footage (trained location)
st.subheader("Inference on Canggu Surf Footage")
st.video(
    "assets/Inference_video.mp4",
    format="video/mp4",
    start_time=0,
    loop=True,
    autoplay=True,
    muted=True,
)

st.markdown(
    """
    To demonstrate real-time performance, I applied the trained model to a never-before-seen Canggu surf video (the same break as our training data). In the clip above, the model consistently and accurately detects both left- and right-hand pockets in each frame.
    
    I was deeply impressed by the model’s reliable first-pass detections, which underscore its ability to generalize to entirely new surf footage.
    """
)

# Second: inference on a different break (pipeline footage)
st.subheader("Inference on Pipeline Surf Footage")
st.video(
    "assets/pipeline_inference.mp4",
    format="video/mp4",
    start_time=0,
    loop=True,
    autoplay=True,
    muted=True,
)
st.markdown(
    """
    Next, I tested the model on a completely new surf location—the world-famous Pipeline. Despite never having seen Pipeline footage, the detector correctly identified most left-hand pockets and even captured many of the subtler right-hand pockets. A few of the gentler right pockets were missed, but overall the model performed well, demonstrating strong generalization across different breaks.
    """
)

# Third: inference on WSL competition footage (Bells Beach)
st.subheader("Inference on World Surf League (WSL) Competition Footage")
st.video(
    "assets/kanoa_igarashi_surf_video.mp4",
    format="video/mp4",
    start_time=0,
    loop=True,
    autoplay=True,
    muted=True,
)
st.markdown(
    """
    To stress-test the model, I applied it to WSL competition footage of Kanoa Igarashi (also my surfing idol 😂) at Bells Beach (Rip Curl Pro 2025). Despite heavy whitewash and much larger waves, it accurately pinpointed the pocket zones—a pleasantly surprising result I hadn’t expected.
    """
)
# Future Applications
st.markdown("---")
st.header("Potential Improvements")
st.markdown(
    """
    One way to boost accuracy is to switch to a larger YOLOv8 variant—such as small or medium—to leverage richer feature extraction. Training on full-resolution frames, rather than downsampled 512×512 crops, could also help capture finer wave details. Expanding the dataset beyond 2,000 images with a wider variety of wave types, lighting, and weather conditions will improve robustness. Increasing the number of training epochs—while monitoring mAP for diminishing returns—can ensure the model fully converges. Finally, performing systematic hyperparameter tuning (for learning rate, augmentation mix, and other settings) will help identify the optimal configuration for peak performance.
    """
)

st.header("Future Applications")
st.markdown(
    """
    Imagine using this pocket detector as an **interactive surf coach**, where surfers upload their own footage and instantly see bounding boxes highlighting the optimal position for speed and control. It could power **smart surf cameras** that track metrics like average ride duration, waves caught versus missed, and display the pocket zone in real time at any break. For coaches and athletes, a **session analytics dashboard** could aggregate pocket detections over an entire surf session to score performance and track improvement over time.

    Together, these possibilities show how pocket detection can evolve into a comprehensive suite of surf-training and beach-monitoring tools. Thank you for exploring this project—feel free to open an issue or reach out on GitHub with ideas and feedback!
    """
)
