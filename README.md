# Multimodal Crime Analyzer (Group 3)

### How to Run Project

For Mac users, use python3 if python does not work. 

**Setup (use Python 3.10–3.12 if possible):**
1. Open terminal in project root
2. `python -m pip install -r requirements.txt`

**For Merged CSV (Optional - Already Included):** `python main/integration/merge_data.py`

**For Dashboard:**
1. `cd main/integration`
2. `python -m streamlit run app.py`
3. Open http://localhost:8501

**Dashboard only (minimal install):** if the full `requirements.txt` fails, CSVs are already in the repo:
```powershell
python -m pip install pandas streamlit plotly
python main/integration/merge_data.py
cd main/integration
python -m streamlit run app.py
```

**Master Table:** [master_incidents.csv](main/integration/master_incidents.csv)
