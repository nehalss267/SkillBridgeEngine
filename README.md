# SkillBridgeEngine: Contextual Skill Gap Analyzer

SkillBridgeEngine is a local, privacy-first machine learning pipeline designed to analyze a candidate's resume against a target job description. It extracts technical entities, computes the semantic "skill gap" using state-of-the-art vector embeddings, and dynamically generates a personalized syllabus of free courses to bridge that gap.

## 🧠 Core Architecture
* **Data Ingestion:** `pdfplumber` for robust PDF resume parsing without breaking multi-column layouts.
* **Custom NER Pipeline:** A locally trained `spaCy` model (built on `en_core_web_md`) to accurately extract modern tech entities (`HARD_SKILL`, `SOFT_SKILL`, `TOOL`).
* **Semantic Engine:** `sentence-transformers` utilizing `BAAI/bge-base-en-v1.5` for high-accuracy, low-latency cosine similarity matching (e.g., matching "AWS" to "Amazon Web Services").
* **Vector Database:** Local `ChromaDB` for storing and retrieving high-dimensional course embeddings.

## 📂 Project Structure
```text
├── mooc_dataset.csv          # Source data for course recommendations
```

## ⚙️ Setup & Installation

**1. Install Dependencies**
Ensure you have Python 3.9+ installed, then run:
```bash
pip install pandas chromadb sentence-transformers spacy pdfplumber scikit-learn datasets
```

**2. Download the Base NLP Model**
Download the medium English vocabulary model for spaCy:
```bash
python -m spacy download en_core_web_md
```

**3. Prepare Dummy Data**
Place the following files in your root directory to test the pipeline:
* `sample_resume.pdf` (Your target resume)
* `sample_jd.txt` (The target job description in plain text)
* `mooc_dataset.csv` (Your library of available courses)

## 🚀 Usage

This project is divided into two phases: a one-time setup phase (training the models and populating the database) and an active inference phase (analyzing resumes).

### Phase 1: System Setup
Run this command **once** to download the Hugging Face job descriptions, train the custom Named Entity Recognition (NER) model, and build the local ChromaDB vector space.
```bash
python main.py --setup
```
*Note: This will create two new local folders: `/custom_ner_model` and `/chroma_db`.*

### Phase 2: Skill Gap Analysis
Once setup is complete, run the analysis engine against your resume and job description:
```bash
python main.py --analyze --resume sample_resume.pdf --jd sample_jd.txt
```

**Expected Output:**
The terminal will output the extracted skills from both documents, isolate the missing skills using semantic cosine similarity, and print a top-3 recommended course syllabus fetched directly from your local vector database.

### 📝 Google Colab / Jupyter Note
If you are running this entirely inside a Jupyter Notebook or Google Colab environment, standard `argparse` CLI commands may conflict with the kernel. Bypass `main.py` and run the functions directly in your cells:
```python
from main import run_setup, run_analysis

# 1. Run once
run_setup()

# 2. Run for each resume
run_analysis(resume_path="sample_resume.pdf", jd_path="sample_jd.txt")
```

***
