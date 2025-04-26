import streamlit as st
import webbrowser

# Set page configuration
st.set_page_config(page_title="Pathorchestra", page_icon=":hospital:", layout="wide")

# Define the main title and subtitle
st.title("Pathorchestra")
st.subheader("Integrated Biomedical Platform for AI-assisted Healthcare")

# Define the interface for the MediFusion module
def medifusion():
    st.header("MediFusion - AI Health Consultation")
    st.write("Enter your symptoms to receive preliminary diagnostic suggestions and recommendations for further actions.")
    
    # User input for symptoms
    symptoms = st.text_area("Enter your symptoms here", height=150)
    
    # Simulate AI diagnosis results
    if st.button("Get Health Consultation"):
        if symptoms:
            st.write("### AI Diagnosis Result:")
            st.write(f"Based on your symptoms: {symptoms}, the AI suggests the following:")
            st.write("- Possible conditions: Symptom A, Symptom B, Symptom C")
            st.write("- Recommended actions: Consult a doctor, take over-the-counter medication, monitor symptoms")
        else:
            st.warning("Please enter your symptoms first.")

# Define the link to the SPMM module
def spmm():
    st.header("SPMM - Tumor/Cancer Prediction")
    st.write("Advanced tumor/cancer prediction model for medical professionals. Click the link below to access the SPMM app.")
    if st.button("Go to SPMM"):
        webbrowser.open("https://spmm-by-ms-ewhfbt4fp5w9rtwfy8rozo.streamlit.app/")

# Define the link to the BioAnnotate module
def bioannotate():
    st.header("BioAnnotate - Biomedical Dataset Annotation")
    st.write("Professional annotation platform for Whole Slide Images (WSI) and spatial transcriptomics data. Click the link below to access the BioAnnotate app.")
    if st.button("Go to BioAnnotate"):
        webbrowser.open("https://machinesleepingbiotech-zpysuvh7r2uuq3twksac79.streamlit.app/")

# Main page layout
st.sidebar.title("Select a Module")
module = st.sidebar.selectbox("Choose a module", ["MediFusion", "SPMM", "BioAnnotate"])

# Display the corresponding module interface based on user selection
if module == "MediFusion":
    medifusion()
elif module == "SPMM":
    spmm()
elif module == "BioAnnotate":
    bioannotate()
