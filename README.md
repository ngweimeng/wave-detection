# Wave Detection for Surfers

## Introduction
This project uses machine learning to find the “pocket” of a wave — the best spot on a breaking wave where a surfer can go the fastest and stay in control. It works by analyzing surf footage, frame by frame, to predict where this pocket is. The idea comes from something surfers often do on the beach called “mind surfing,” where they watch the waves and imagine how they would ride them, choosing the best line without actually getting in the water.

![Parts of a Wave](https://www.saltwater-dreaming.com/learn-to-surf/images/parts-of-a-wave.jpg)  
*Source: [Saltwater Dreaming](https://www.saltwater-dreaming.com/learn-to-surf/parts-of-a-wave.htm)*

In the diagram above, the "pocket" refers to the steep, powerful zone just ahead of where the wave breaks. This is where surfers aim to stay for optimal performance — and the region that this model is trained to detect.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                         <- Source code for this project
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    │    
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    ├── plots.py                <- Code to create visualizations 
    │
    └── services                <- Service classes to connect with external platforms, tools, or APIs
        └── __init__.py 
```

--------