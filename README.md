# Wave Detection for Surfing with Computer Vision

This project uses a fine-tuned YOLOv8-nano model to identify the “pocket” zone of a wave in real-time surf footage. The goal is to help surfers and coaches visualize optimal positioning on the wave.

## Demo Clips
Below are four inference videos produced by the model. Each demonstrates pocket detection on different surf footage:

### Canggu Surf Footage  
[![Canggu Surf GIF](assets/canggu_inference.gif)](assets/Inference_video.mp4)

### Pipeline Break Inference  
[![Pipeline Break GIF](assets/pipeline_inference.gif)](assets/pipeline_inference.mp4)

### World Surf League Competition Inference
[![WSL Competition GIF](assets/kanoa_inference.gif)](assets/kanoa_igarashi_surf_video.mp4)

### Personal Surf Session (Bonus)  
[![Bonus Surf Session GIF](assets/inference_bonus.gif)](assets/inference_bonus.mp4)

**Interactive write-up:** [read the full case study →](https://claude.ai/code/artifact/07da97af-3c76-47f2-9909-9946b5f74794) — a self-contained page with the demo clips and metrics inline (no cold-start).

_Optional:_ a live Streamlit version is also hosted at [wave-pocket-detection.streamlit.app](https://wave-pocket-detection.streamlit.app/) (may take a moment to wake).

## Run Locally

```bash
git clone https://github.com/ngweimeng/wave-detection.git
cd wave-detection
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Project Structure

```
wave-detection/
├── app.py                              # Streamlit write-up + results viewer
├── upload_to_roboflow.ipynb            # Frame extraction + Roboflow upload
├── train_model_with_custom_dataset.ipynb  # YOLOv8-nano fine-tuning
├── assets/                             # Diagrams, metric plots, demo clips
└── requirements.txt
```



