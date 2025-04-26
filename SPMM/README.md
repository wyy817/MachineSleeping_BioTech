# Spatio-Based Pathology MultiModal (SPMM)

This project implements a multimodal medical foundation model based on spatial transcriptomics, specifically designed for tumor/cancer slice prediction tasks. The model integrates spatial transcriptomic data, pathological images, and clinical text information to provide accurate tumor classification, staging, and prognosis prediction.

## Characteristics

- **Multimodal Integration**: Fuses spatial transcriptomics, pathological images, and clinical text information
- **Spatial Encoder**: Utilizes graph neural networks to capture cell-molecule spatial relationships
- **Text Encoder**: Processes clinical text using biomedical language models
- **Visual Encoder**: Handles WSI images with pre-trained medical image models
- **Modality Alignment**: Aligns cross-modal features through attention mechanisms
- **Optimization Mechanism**: Incorporates self-supervised learning, contrastive learning, and reinforcement learning
- **Prediction Module**: Provides tumor classification and survival prediction
- **Visualization Features**: Rich visualization tools to help interpret model decisions
- **Domain Generalization**: Enhances domain generalization using MMSEG adapters

## Innovations

1. **Cell-Molecule Spatial Relationships Modeling**
   - Construct a cell communication network to capture microenvironmental information
   - Integrate the spatial correlation between cell types and molecular expression

2. **Multimodal Knowledge Fusion**
   - Achieve deep integration of spatial transcriptomics, pathological images, and clinical text
   - Align cross-modal features through multi-head attention mechanisms

3. **Uncertainty Quantification**
   - Estimate prediction confidencee using Bayesian neural networks
   - Enhance model robustness through ensemble learning

## Installation

### environment requirements

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- Transformers
- scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Plotly

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/spatial-multimodal-medical-model.git
cd spatial-multimodal-medical-model

# Create a virtual environment
conda create -n spatial_mm python=3.8
conda activate spatial_mm

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric
pip install torch-geometric

# Install additional dependencies
pip install transformers scikit-learn matplotlib seaborn plotly
```

## User Guide

### Data Preparation

Place the spatial transcriptomics data, WSI images, and clinical text in the following directory structure:

```
data/
  └── sample_id/
      ├── spatial_data.h5ad  # Spatial Transcriptomics Data (AnnData format)
      ├── image.png          # Whole Slide Images (WSI)
      ├── clinical.txt       # Clinical Text Data
      └── labels.csv         # Classification Labels
```

Alternatively, use the 10x Visium data format:

```
data/
  └── sample_id/
      ├── filtered_feature_bc_matrix/  # Gene Expression Matrix
      ├── spatial/                     # Spatial Coordinates and Images
      └── labels.csv                   # Classification Labels
```

### Model Training

```bash
python main.py --mode train --data_dir ./data
```

### Prediction

```bash
python main.py --mode predict --data_dir ./data --model_path ./saved_models/best_model.pth
```

### Results Visualization

```bash
python main.py --mode visualize --data_dir ./data --model_path ./saved_models/best_model.pth
```

The generated visualizations will be saved in the ./visualizations directory, and you can view the comprehensive dashboard by opening ./visualizations/dashboard/index.html in a browser.

## Project Structure

```
.
├── config.py                 # Configuratuin File
├── main.py                   # Main Program Entry
├── models/                   # Model Definition
│   ├── spatial_encoder.py    # Spatial Encoder
│   ├── text_encoder.py       # Text Encoder
│   ├── vision_encoder.py     # Visual Encoder
│   ├── modality_alignment.py # Modality Alignment Module
│   ├── optimization.py       # Optimization Mechanism Module
│   ├── prediction.py         # Prediction Module
│   ├── mmseg_adapter.py      # MMSEG Adapter
│   └── model.py              # Overall Model
├── data/                     # Data Processing
│   ├── data_processor.py     # Data Preprocessing
│   ├── dataset.py            # Dataset Definition
│   └── data_loader.py        # Data Loader
├── utils/                    # Utility Functions
│   ├── evaluation.py         # Evaluation Functions
│   ├── loss_functions.py     # Loss Functions
│   └── metrics.py            # Evaluation Metrics
└── visualization/            # Visualization Module
    ├── attention.py          # Attention Visualization
    ├── spatial.py            # Spatial Visualization
    ├── multimodal.py         # Multimodal Visualization
    ├── tumor_microenvironment.py  # Tumor Microenvironment Visualization
    ├── interactive.py        # Interactive Visualization
    └── dashboard.py          # Dashboard Creator
```

## Model Structure

### Spatial Encoder

The spatial encoder utilizes Graph Attention Networks (GAT) to process spatial transcriptome data, capturing cell-molecule spatial relationships. It includes:

- Multi-layer graph attention layers to handle spatial dependencies between cells
- Point cloud processing components to extract spatial coordinate information
- Feature fusion layers to integrate graph features and spatial features

### Text Encoder

The text encoder is based on biomedical language models (such as PubMedBERT) and processes clinical text data, including:

- Pre-trained biomedical language models
- Additional layers for specific tasks (e.g., clustering, cell type inference, deconvolution)
- Output projection layer to align multimodal feature spaces

### Visual Encoder

The visual encoder processes WSI images and employs:

- A ViT-based backbone network to extract global visual features
- Tumor microenvironment identification modules to capture specific pathological features
- Feature fusion layers to integrate various visual features

### Modality Alignment

The modality alignment module achieves feature alignment across different modalities through multi-layer attention mechanisms:

- Text-to-visual cross-attention
- Visual-to-text cross-attention
- Spatial-to-text/visual cross-attention
- Multimodal fusion layers

### Prediction Module

The prediction module performs tumor classification and survival prediction based on aligned multimodal features:

- Weighted feature layers based on attention mechanisms
- Tumor classifier to predict tumor types and grades
- Survival predictor to estimate patient prognosis

## Model Evaluation

Testing across multiple cancer types demonstrates that this model has significant advantages compared to unimodal approaches:

- Improves classification accuracy
- Enhances the C-index for prognosis prediction
- Exhibits greater robustness to sparse data and data noise

## Visualization Features

This model offers a wide array of visualization tools to aid in understanding and interpreting prediction results:

- Visualization of spatial distribution of gene expression
- Visualization of cell types and domain distributions
- Tumor microenvironment interaction network visualization
- Feature importance heatmaps
- Interactive exploration tools (e.g., gene expression toggling, domain exploration)
- Comprehensive analysis dashboards

## Application Scenarios

- **Precision Tumor Diagnosis**: Integrating multimodal data to improve diagnostic accuracy
- **Cancer Typing and Grading**: Conducting refined typing based on molecular and morphological features
- **Therapeutic Response Prediction**: Predicting patient responses to various treatment plans
- **Tumor Microenvironment Analysis**: Unveiling interaction patterns between tumors and the immune system
- **Biomarker Discovery**: Identifying molecular markers with diagnostic and prognostic value

## References

1. Chen, Zhang, Tang, et al. (2024). ErwaNet: Edge-relational Window-attentional GNN for predicting gene expression from standard tissue images.
2. Steyaert et al. (2023). Multimodal fusion approaches for integrating complementary data types in cancer research.
3. Bottosso et al. (2024). Precision medicine for predicting drug sensitivity in breast cancer based on molecular understanding.
4. Lobato-Delgado, Priego-Torres, and Sanchez-Morillo (2022). Combining molecular, imaging, and clinical data analysis for cancer prognosis.

## Contribution Guidelines

We welcome issue submissions and pull requests. For major changes, please open an issue first to discuss what you want to modify.

## License

This project is licensed under the MIT License – for more details, please see the LICENSE file.

## Acknowledgments

- We thank all researchers who have contributed to the development of spatial transcriptomics technology.
- We appreciate the tools and libraries provided by the open-source community.
- Special thanks to the partners who supported this project with data contributions.

## Contact Information

For any questions or suggestions, please contact: yuyaowang817@gmail.com