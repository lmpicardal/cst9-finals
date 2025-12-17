import streamlit as st

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Patient Data Entry Form")

# --- Data Options ---
LOCATION_OPTIONS = ["Temporal", "Parietal", "Frontal", "Occipital"]
HISTOLOGY_OPTIONS = ["Astrocytoma", "Glioblastoma", "Meningioma", "Medulloblastoma"]
STAGE_OPTIONS = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
SYMPTOM_OPTIONS = ["Vision Issue", "Headache", "Seizure", "Nausea"]
GENDER_OPTIONS = ["Female" , "Male"]


# --- Form Title ---
st.title("BRAIN TUMOR DIAGNOSIS")

st.markdown("---")

st.header("Patient Information")

# Use columns for Age and Gender
col1_info, col2_info = st.columns(2)
with col1_info:
    st.number_input("Enter Age (Years)", min_value=1, max_value=120, value=1, key="age")
with col2_info:
    st.selectbox("Select Gender", options=GENDER_OPTIONS, key="gender")
    
st.markdown("---")

## --- Treatment and History 

st.header("Treatment and History")
col_rh, col_cs = st.columns(2)

# Note: The Yes/No radio button is set to align horizontally by default in Streamlit.
with col_rh:
    st.radio("Radiation Therapy?", options=["Yes", "No"], key="radiation")
    st.radio("Surgery Performed?", options=["Yes", "No"], key="surgery")
with col_cs:
    st.radio("Chemotherapy?", options=["Yes", "No"], key="chemotherapy")
    st.radio("Family History?", options=["Yes", "No"], key="family_history")
    
st.markdown("---")

## --- Symptom Selection Logic (Crucial for Non-Duplication) ---
st.header("Symptoms")

col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    # Symptom 1 selection
    sym1 = st.selectbox("Select Symptom 1", options=["- Select -"] + SYMPTOM_OPTIONS, key="symptom1")

# Calculate options available for Symptom 2
sym2_options = SYMPTOM_OPTIONS.copy()
if sym1 != "- Select -" and sym1 in sym2_options:
    sym2_options.remove(sym1)

with col_s2:
    # Symptom 2 selection
    sym2 = st.selectbox("Select Symptom 2", options=["- Select -"] + sym2_options, key="symptom2")

# Calculate options available for Symptom 3
sym3_options = SYMPTOM_OPTIONS.copy()
if sym1 != "- Select -" and sym1 in sym3_options:
    sym3_options.remove(sym1)
if sym2 != "- Select -" and sym2 in sym3_options:
    sym3_options.remove(sym2)

with col_s3:
    # Symptom 3 selection
    sym3 = st.selectbox("Select Symptom 3", options=["- Select -"] + sym3_options, key="symptom3")

st.markdown("---")
    
st.header("Tumor Description")

# Use columns for Tumor Size and Growth Rate
col1, col2 = st.columns(2)

with col1:
    st.number_input("Enter Tumor Size (cm)", min_value=0.1, max_value=30.0, value=.5, step=0.1, key="size")
with col2:
    st.number_input("Enter Tumor Growth Rate", min_value=0.0, value=0.5, step=0.1, key="growth_rate", format="%.2f")

st.markdown("---")

# Use another set of columns for Location, Histology, and Stage
col4, col5, col6 = st.columns(3)

with col4:
    st.selectbox("Select Tumor Location", options=LOCATION_OPTIONS, key="location")
with col5:
    st.selectbox("Select Histology Type", options=HISTOLOGY_OPTIONS, key="histology")
with col6:
    st.selectbox("Select Disease Stage", options=STAGE_OPTIONS, key="stage")

st.markdown("---")


# --- Submission Button and Output ---
if st.button("Submit Patient Data"):

    data = {
        "Age": st.session_state.age,
        "gender": st.session_state.gender,
        "Tumor Size": st.session_state.size,
        "Growth Rate": st.session_state.growth_rate,
        "Location": st.session_state.location,
        "Histology": st.session_state.histology,
        "Stage": st.session_state.stage,
        "Symptom 1": sym1,
        "Symptom 2": sym2,
        "Symptom 3": sym3,
        "Radiation": st.session_state.radiation,
        "Surgery": st.session_state.surgery,
        "Chemotherapy": st.session_state.chemotherapy,
        "Family History": st.session_state.family_history,
    }
    
    # Display collected data (in a real app, you would save this to a database)
    st.success("Data Submitted Successfully!")
    st.subheader("Calculating prognosis:")
    st.json(data)