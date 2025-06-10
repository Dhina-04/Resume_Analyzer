import streamlit as st
import pandas as pd
import PyPDF2
import google.generativeai as genai
import os
import io
from typing import Dict, List
import json
import uuid
from pathlib import Path

# Configure Gemini API
os.environ["GEMINI_API_KEY"] = st.secrets.get("GEMINI_API_KEY", "AIzaSyD098b2y21Iv-PLXGohUvWt-UNYqwlLshM")
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-2.0-flash"

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# Function to process resume with Gemini API
def analyze_resume(resume_text: str, job_description: str, job_role: str) -> Dict:
    prompt = f"""
    Analyze the following resume and job description. Extract key skills, summarize work experience, and calculate a match score (0-100) for the job role: {job_role}.
    Job Description: {job_description}
    Resume: {resume_text}
    
    Return a JSON object with:
    - skills: List of extracted skills
    - experience_summary: Summary of work experience
    - match_score: Integer score (0-100) indicating alignment with job role
    - missing_skills: List of skills from job description not found in resume
    """
    
    try:
        model = genai.GenerativeModel(MODEL)
        response = model.generate_content(prompt)
        result = json.loads(response.text.strip("```json\n").strip("```"))
        return result
    except Exception as e:
        st.error(f"Error analyzing resume: {e}")
        return None

# Function to process bulk upload CSV
def process_bulk_upload(csv_file, job_description, job_role):
    df = pd.read_csv(csv_file)
    results = []
    
    for _, row in df.iterrows():
        resume_text = row.get("resume_text", "")
        candidate_id = str(uuid.uuid4())
        if resume_text:
            analysis = analyze_resume(resume_text, job_description, job_role)
            if analysis:
                analysis["candidate_id"] = candidate_id
                analysis["name"] = row.get("name", "Unknown")
                analysis["email"] = row.get("email", "N/A")
                results.append(analysis)
    return results

# Streamlit App
st.title("Resume Analyzer with Gemini 2.0 Flash")

# Sidebar for Job Role and Description
st.sidebar.header("Job Details")
job_role = st.sidebar.selectbox("Select Job Role", ["Data Scientist", "Software Engineer", "Data Analyst", "Machine Learning Engineer", "Data Engineer"])
job_description = st.sidebar.text_area("Enter Job Description", "Enter the job description here...")

# Tabs for Single and Bulk Upload
tab1, tab2, tab3 = st.tabs(["Single Resume Upload", "Bulk Resume Upload", "Analytics Dashboard"])

# Single Resume Upload
with tab1:
    st.header("Upload a Single Resume")
    uploaded_file = st.file_uploader("Choose a PDF or Text file", type=["pdf", "txt"])
    
    if uploaded_file and job_description:
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = uploaded_file.read().decode("utf-8")
        
        if resume_text:
            with st.spinner("Analyzing resume..."):
                result = analyze_resume(resume_text, job_description, job_role)
                if result:
                    st.subheader("Analysis Results")
                    st.write(f"**Skills**: {', '.join(result['skills'])}")
                    st.write(f"**Work Experience Summary**: {result['experience_summary']}")
                    st.write(f"**Match Score**: {result['match_score']}/100")
                    st.write(f"**Missing Skills**: {', '.join(result['missing_skills'])}")
                    
                    # Option to view resume
                    if uploaded_file.type == "application/pdf":
                        st.download_button(
                            label="View/Download Resume",
                            data=uploaded_file,
                            file_name=uploaded_file.name,
                            mime="application/pdf"
                        )
                    else:
                        st.text_area("Resume Content", resume_text, height=300)

# Bulk Resume Upload
with tab2:
    st.header("Bulk Resume Upload")
    st.markdown("Upload a CSV file with columns: 'name', 'email', 'resume_text'. [Download sample CSV](#)")
    
    # Sample CSV download
    sample_data = pd.DataFrame({
        "name": ["John Doe", "Jane Smith"],
        "email": ["john@example.com", "jane@example.com"],
        "resume_text": ["Experienced Data Scientist with Python, SQL...", "Software Engineer proficient in Java, C++..."]
    })
    sample_csv = sample_data.to_csv(index=False)
    st.download_button(
        label="Download Sample CSV",
        data=sample_csv,
        file_name="sample_resume_data.csv",
        mime="text/csv"
    )
    
    bulk_file = st.file_uploader("Upload CSV with resume data", type=["csv"])
    
    if bulk_file and job_description:
        with st.spinner("Processing bulk upload..."):
            results = process_bulk_upload(bulk_file, job_description, job_role)
            if results:
                st.session_state["bulk_results"] = results
                st.success("Bulk processing complete!")

# Analytics Dashboard
with tab3:
    st.header("Analytics Dashboard")
    
    if "bulk_results" in st.session_state and st.session_state["bulk_results"]:
        results = st.session_state["bulk_results"]
        df = pd.DataFrame(results)
        
        # Summary Metrics
        st.subheader("Summary Analytics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Candidates", len(df))
        col2.metric("Average Match Score", round(df["match_score"].mean(), 1))
        col3.metric("Top Score", df["match_score"].max())
        
        # List View of Candidates
        st.subheader("Candidate List")
        display_df = df[["candidate_id", "name", "email", "match_score", "skills", "experience_summary"]]
        st.dataframe(display_df, use_container_width=True)
        
        # View Individual Resume
        selected_candidate = st.selectbox("Select Candidate to View Resume", options=df["candidate_id"], format_func=lambda x: df[df["candidate_id"] == x]["name"].iloc[0])
        if selected_candidate:
            candidate_data = df[df["candidate_id"] == selected_candidate].iloc[0]
            st.subheader(f"Details for {candidate_data['name']}")
            st.write(f"**Email**: {candidate_data['email']}")
            st.write(f"**Skills**: {', '.join(candidate_data['skills'])}")
            st.write(f"**Experience Summary**: {candidate_data['experience_summary']}")
            st.write(f"**Match Score**: {candidate_data['match_score']}/100")
            st.write(f"**Missing Skills**: {', '.join(candidate_data['missing_skills'])}")
            st.text_area("Resume Content", df[df["candidate_id"] == selected_candidate]["resume_text"].iloc[0], height=300)
        
        # Visualization
        import plotly.express as px
        st.subheader("Match Score Distribution")
        fig = px.histogram(df, x="match_score", nbins=20, title="Distribution of Match Scores")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload resumes in the Bulk Upload tab to view analytics.")
