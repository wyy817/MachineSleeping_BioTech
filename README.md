# Pathorchestra

Pathorchestra is an integrated biomedical platform that addresses healthcare challenges by combining AI-assisted diagnosis, tumor/cancer prediction, and medical dataset annotation. The platform aims to alleviate uneven distribution of medical resources, assist healthcare professionals, and contribute to the development of medical AI models.

## Overview

Pathorchestra consists of three specialized modules working together to address different aspects of healthcare challenges:

1. **MediFusion** - AI health consultation system for individual users
2. **SPMM (Spatio-Based Pathology MultiModal)** - Advanced tumor/cancer prediction model for medical professionals
3. **BioAnnotate** - Professional annotation platform for biomedical datasets

## Core Modules

### 1. MediFusion

MediFusion is an AI-based health problem diagnosis platform designed for individual users. Users can input their symptoms to receive preliminary diagnostic suggestions and recommendations for further actions.

**Key Features:**
- Symptom-based health consultation
- AI-powered preliminary diagnosis
- User-friendly web interface
- Personalized health recommendations

### 2. SPMM (Spatio-Based Pathology MultiModal)

SPMM is a state-of-the-art multimodal medical foundation model specifically designed for tumor/cancer slice prediction tasks. It integrates spatial transcriptomic data, pathological images, and clinical text to provide accurate tumor classification, staging, and prognosis prediction.

**Key Features:**
- Multimodal integration of spatial transcriptomics, images, and clinical text
- Spatial encoding using graph neural networks
- Advanced tumor classification and prognosis prediction
- Rich visualization tools for interpretability
- Domain generalization with MMSEG adapters

### 3. BioAnnotate

BioAnnotate is a professional annotation platform for Whole Slide Images (WSI) and spatial transcriptomics data in biomedical research, providing tools for researchers to annotate, classify, and analyze biological data.

**Key Features:**
- WSI annotation tools
- Spatial transcriptomics annotation
- Multi-modal data integration
- Project management and collaboration
- Export options for further analysis

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/wyy817/MachineSleeping_BioTech.git
   cd pathorchestra
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python run.py
   ```

## Requirements

- Python 3.8+
- Flask 3.1.0+
- PyTorch 1.9+
- Streamlit 1.22.0+
- Pandas, NumPy, Matplotlib, Plotly
- Additional requirements are specified in `requirements.txt`

## Usage

1. Open your browser and navigate to: http://127.0.0.1:5000/ (default)
2. From the main dashboard, select one of the three modules:
   - **MediFusion** - For health consultation
   - **SPMM** - For tumor/cancer prediction (professional use)
   - **BioAnnotate** - For biomedical data annotation

3. Each module will guide you through its specific workflows:
   - In MediFusion, input your symptoms to receive health consultation
   - In SPMM, upload multimodal data (spatial transcriptomics, images, text) for tumor analysis
   - In BioAnnotate, manage annotation projects for WSI and spatial transcriptomics data

## Project Structure

```
pathorchestra/
├── app/                    # Main application
│   ├── __init__.py         # App initialization
│   ├── routes/             # Route definitions
│   ├── models.py           # Data models
│   ├── forms.py            # Form definitions
│   └── utils.py            # Utility functions
├── config.py               # Configuration settings
├── medifusion/             # MediFusion module
│   └── llm.py              # AI model for health consultation
├── spmm/                   # SPMM module
│   ├── models/             # Model definitions
│   ├── data/               # Data processing
│   └── visualization/      # Visualization tools
├── bioannotate/            # BioAnnotate module
│   ├── wsi_annotation.py   # WSI annotation tools
│   └── spatial_annotation.py # Spatial annotation tools
├── static/                 # Static assets (CSS, JS, images)
├── templates/              # HTML templates
├── run.py                  # Application entry point
└── requirements.txt        # Dependencies
```

## UI Design

The Pathorchestra interface provides a unified experience for accessing all three modules:

1. **Main Dashboard**: The landing page features three prominent module cards with descriptions and entry points.
2. **Navigation Bar**: Consistent navigation across all modules with a clear way to return to the main dashboard.
3. **User Authentication**: A single login system for accessing all three modules with role-based permissions.
4. **Responsive Design**: Works well on desktop and mobile devices.

## Module Integration

The three modules work together seamlessly:

1. **Data Flow**: Annotations from BioAnnotate can be used to train SPMM models.
2. **Shared Components**: Common UI elements and user management across modules.
3. **Unified Authentication**: Single sign-on for all modules with role-based access control.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or suggestions, please open an issue on GitHub or contact us at yuyaowang817@gmail.com.
