# 🛒 Online Retail Recommendation System

This project provides an **Online Retail Recommendation & Analysis System** with two modes of operation:

1. **Normal Auto-Run (CMD Window)**
   - Runs analyses.
   - Generates **CSV outputs** and **charts (PNG)** automatically.
   - Non-interactive (batch run).

2. **Interactive Streamlit Dashboard**
   - Provides an interactive dashboard to explore data.
   - Includes popularity analysis, collaborative filtering, and frequently-bought-together insights.

---

## 📂 Project Contents
- `OnlineRetail(1).xlsx` → Dataset file  
- `retail_recommender.py` → Main script (contains both auto-run and Streamlit app code)  
- `Run.bat` → Windows batch file to run the project  
- `requirements.txt` → Python dependencies  
- `retail_outputs_*.csv / .png` → Auto-generated analysis results  
- `Online_Retail_Recommendation_System_Foreman_Report.docx` → Formal report  

---

## ▶️ How to Run

After executing Run.bat file it asks user for 2 modes 
1.CMD mode & 2. Streamlit mode

### Option 1: Run in CMD (Auto Mode)
This will run the script in **auto mode** and generate all CSV + PNG outputs.

```bash
Run.bat
```

The results will be saved as:
- `retail_outputs_popularity_global.csv`
- `retail_outputs_popularity_by_country.csv`
- `retail_outputs_popularity_by_month.csv`
- `retail_outputs_sample_user_recommendations.csv`
- `retail_outputs_fbt_recommendations.csv`
- Plus corresponding charts in `.png`

---

### Option 2: Run Streamlit Dashboard
For interactive exploration, run:

```bash
Run.bat
```

Features available in the dashboard:
- 🌍 Global Popularity Analysis  
- 🌐 Popularity by Country  
- 📅 Monthly Trends  
- 🎯 Sample User Recommendations (CF)  
- 🔗 Frequently Bought Together (FBT)  
- 🔥 Popularity-Based Recommendations  
- 🛍 Item-Based Collaborative Filtering  

---

## ⚙️ Environment
- A Python **virtual environment (`venv`)** is already included in this project.  
- No repo cloning needed; just use the existing setup.  

---

## 📊 Outputs
All results will be saved automatically in the project folder as `.csv` and `.png` files, ready for presentation.


## 🤝 Contributing
Pull requests are welcome. For major changes, open an issue first to discuss what you’d like to change.

---

## 📜 License
MIT License © 2025 Kusan Chakraborty
