# BioAnnotate

BioAnnotate is a professional annotation platform for Whole Slide Images (WSI) and spatial transcriptomics data in biomedical research. It provides an intuitive interface for researchers to annotate, classify, and analyze biological data.

## Features

- **WSI Annotation**: Mark regions of interest, classify cells/tissues, and measure features on whole slide images
- **Spatial Transcriptomics Annotation**: Annotate and classify spots in spatial transcriptomics data
- **Multi-modal Integration**: View and annotate matched WSI and spatial transcriptomics data together
- **Project Management**: Organize annotations by projects and manage workspaces
- **Export Options**: Export annotations in various formats for further analysis
- **User Authentication**: Secure user accounts and data management

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/bioannotate.git
   cd bioannotate
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
   streamlit run app.py
   ```

## Requirements

- Python 3.8+
- Streamlit
- OpenCV
- NumPy
- Pandas
- Matplotlib/Plotly
- anndata (for spatial transcriptomics)
- openslide-python (for WSI)

See `requirements.txt` for a complete list of dependencies.

## Usage

1. Start the application using `streamlit run app.py`
2. Log in or create a new account
3. Create a new project or select an existing one
4. Upload WSI or spatial transcriptomics data
5. Use the annotation tools to mark regions of interest
6. Save your annotations and export as needed

## Demo Data

The repository includes sample data for demonstration purposes:
- Sample WSI images
- Sample spatial transcriptomics data (in AnnData format)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For any questions or suggestions, please open an issue on GitHub or contact us at example@example.com.