
# 📄 Resume–Job Description Matching Assistant (RAG)

This project is a Retrieval-Augmented Generation (RAG) based application that analyzes how well a resume matches a given job description.
It provides an explainable analysis highlighting matching skills, missing skills, and improvement suggestions.

---

## 🚀 Project Overview

Recruiters and candidates often manually compare resumes with job descriptions, which is time-consuming and inconsistent.
This application uses RAG and LLMs to ground the analysis on actual resume content instead of relying on generic model knowledge.

---

## 🧠 Key Features

- Supports PDF and TXT files for resumes and job descriptions
- Uses semantic retrieval to fetch relevant resume content
- Generates document-grounded, non-hallucinated responses
- Provides skill match, skill gaps, and improvement suggestions
- Interactive Streamlit web interface

---

## 🛠️ Tech Stack

- Python
- LangChain
- Retrieval-Augmented Generation (RAG)
- HuggingFace Sentence Transformers
- Chroma Vector Database
- Gemini LLM
- Streamlit

---

## 📂 Project Structure

resume_jd_rag/
├── app.py
├── README.md
└── data/
    ├── resumes/
    └── job_descriptions/
├── requirements.txt

---

## ⚙️ How It Works

1. User uploads a resume and a job description (PDF or TXT).
2. Resume content is split into chunks and converted into embeddings.
3. Embeddings are stored in a vector database.
4. Job description is used as a query to retrieve relevant resume sections.
5. Retrieved content is passed to the LLM for analysis.
6. The app generates matching skills, missing skills, and improvement suggestions.

---

## ▶️ How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Set your Google API key in app.py:
   os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

3. Run the application:
   streamlit run app.py

---

## 📊 Output

- Matching skills
- Missing skills
- Improvement suggestions

All results are based on retrieved resume content to reduce hallucinations.

---

## ⚠️ Limitations

- Intended as an assistive tool, not an automated hiring system
- Does not rank candidates or make final decisions

---

## 📌 Future Enhancements

- Skill match percentage
- Multiple resume comparison
- Resume ranking
- UI improvements

---

## 🙋‍♂️ Author

Sai Praneeth Marripelli  
MCA Graduate | AI/ML & GenAI

