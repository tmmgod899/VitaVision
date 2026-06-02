# 🌿 VitaVision — Vitamin & Mineral Health Analyzer

**AI-powered web application for vitamin and mineral lab result analysis**
تطبيق ذكي لتحليل نتائج الفيتامينات والمعادن بطريقة سهلة وواضحة

---

## 📋 About the Project | نبذة عن المشروع

**VitaVision** is an intelligent web application built with **Streamlit** that helps users analyze vitamin and mineral laboratory results using clinical reference ranges, a Machine Learning model, and AI-generated explanations.

VitaVision supports manual input, CSV upload, and Smart AI Upload, where users can upload lab files and allow AI to extract supported nutrient values before analysis.

يهدف المشروع إلى مساعدة المستخدم على فهم نتائج تحاليله بشكل مبسط وتوعوي، مع التأكيد أن التطبيق لا يغني عن الطبيب أو المختص الصحي.

---

## 🌐 Live Website

🔗 **Website:** https://www.vitavison.com

---

## ✨ Features | المميزات

| Feature                        | Description                                          |
| ------------------------------ | ---------------------------------------------------- |
| 📝 Manual Input                | Enter lab values manually                            |
| 📂 CSV Upload                  | Upload CSV files for analysis                        |
| 🤖 Smart AI Upload             | Upload files and let AI extract supported lab values |
| 🧠 AI Explanation              | Generate AI explanation for each nutrient            |
| 💬 AI Health Companion Summary | Full friendly AI summary for the patient             |
| 🔬 Rule-Based Classification   | Classifies results using medical reference ranges    |
| 🌲 Machine Learning Prediction | ML model prediction beside rule-based result         |
| 📊 Visualizations              | Charts for status distribution and reference ranges  |
| 🌐 Bilingual UI                | Arabic / English interface                           |
| 👤 Guest Mode                  | Users can analyze without registration               |
| 🔐 Login / Register            | Supabase authentication                              |
| 🛡️ Admin Role                 | Admin dashboard support                              |
| 📥 CSV Export                  | Download results as CSV                              |
| ⚠️ Medical Disclaimer          | Clear medical disclaimer before use                  |

---

## 🧪 Supported Nutrients | العناصر المدعومة

| Nutrient    | Unit   |
| ----------- | ------ |
| Zinc        | µg/dL  |
| Vitamin E   | µg/dL  |
| Vitamin A   | µg/dL  |
| Vitamin D   | ng/mL  |
| Vitamin C   | mg/dL  |
| Magnesium   | mg/dL  |
| Folate      | ng/mL  |
| Ferritin    | ng/mL  |
| Vitamin K   | ng/mL  |
| Vitamin B12 | pg/mL  |
| Vitamin B6  | nmol/L |
| Calcium     | mg/dL  |

---

## 🤖 AI + ML Workflow

VitaVision follows this workflow:

```text
User Input
↓
Manual / CSV / Smart AI Upload
↓
Data Review
↓
Rule-Based Classification
↓
Machine Learning Prediction
↓
AI Explanation
↓
AI Health Companion Summary
↓
Results + Charts + CSV Download
```

---

## 📊 Classification Output

The system classifies each nutrient into one of three categories:

| Output    | Meaning                       |
| --------- | ----------------------------- |
| Deficient | Below normal reference range  |
| Normal    | Within normal reference range |
| Excessive | Above normal reference range  |

---

## 🧠 Machine Learning Model

The application uses a trained Machine Learning model to support prediction and compare it with the rule-based clinical classification.

| Item           | Value                                   |
| -------------- | --------------------------------------- |
| Model Type     | Random Forest Classifier                |
| Input Features | Age, Gender, Nutrient, Value            |
| Output Classes | Deficient / Normal / Excessive          |
| Model File     | `models/vitavision_hybrid_model_v1.pkl` |

The rule-based result is considered the main clinical classification, while the ML model provides additional predictive support.

---

## 🗂️ Project Structure

```text
VitaVision_Project/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
├── IconVitaVision.png
│
├── models/
│   └── vitavision_hybrid_model_v1.pkl
│
└── .gitignore
```

---

## ⚙️ Requirements

```text
streamlit
pandas
plotly
scikit-learn
joblib
openai
supabase
openpyxl
```

---

## 🚀 Run Locally

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## 🔐 Environment Variables

For deployment, the following environment variables are required:

```text
OPENAI_API_KEY
OPENAI_MODEL
SUPABASE_URL
SUPABASE_ANON_KEY
```

Example:

```text
OPENAI_MODEL=gpt-4o-mini
```

Do not upload API keys or secrets to GitHub.

---

## ☁️ Deployment

The project is deployed using:

* **GitHub** for source code
* **Render** for hosting
* **GoDaddy** for domain management
* **Supabase** for authentication and database
* **OpenAI API** for AI explanations

---

## 🔐 Authentication System

VitaVision supports:

| Role  | Access                                |
| ----- | ------------------------------------- |
| Guest | Analyze without saving data           |
| User  | Login/Register and save analysis logs |
| Admin | Admin dashboard and system overview   |

---

## ⚠️ Medical Disclaimer

VitaVision is an educational and awareness tool only.
It does not provide medical diagnosis, treatment, or prescription.

Users should consult a qualified healthcare professional before making any medical decisions based on the results.

---

## 👨‍💻 Technologies Used

* Python
* Streamlit
* Pandas
* Plotly
* Scikit-learn
* OpenAI API
* Supabase
* Render
* GoDaddy

---

## 📌 Project Status

✅ Deployed
✅ Domain connected
✅ AI explanation enabled
✅ Supabase authentication added
✅ Guest mode supported
✅ Smart AI Upload added

---

## 🌿 VitaVision

**Smart Insight for Vitamin Health**
