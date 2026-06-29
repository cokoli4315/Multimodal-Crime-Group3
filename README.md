# Multimodal Crime Analyzer (Group 3)

A multimodal incident analysis pipeline for EDS 6397 - Generative AI & Applications Course. Four analyst modules extract structured crime and emergency data from different sources—**911 call audio**, **PDF incident reports**, **scene images**, and **social-media text**—then an integration layer merges them into one master dataset and a Streamlit dashboard for filtering, charts, and incident lookup.


| Modality | Input                    | Output                                          |
| -------- | ------------------------ | ----------------------------------------------- |
| Audio    | 911 WAV Clips + Metadata | `main/audio/911_audio_analysis.csv`             |
| Document | PDF Reports              | `main/document/structured_incident_reports.csv` |
| Image    | Crime/Emergency Photos   | `main/images/image_analysis_output.csv`         |
| Text     | Twitter Crime Reports    | `main/text/text_analyst_output.csv`             |


The merged table lives at `main/integration/master_incidents.csv` (~1,002 records) with a shared schema: `Incident_ID`, `Source`, `Event`, `Location`, `Time`, `Severity`, and `Detail`.

### How to Run Project

For Mac users, use python3 if python does not work. 

**Setup (use Python 3.10–3.12 if possible):**

1. Open terminal in project root
2. `python -m pip install -r requirements.txt`

**For Merged CSV (Optional - Already Included):** `python main/integration/merge_data.py`

**For Dashboard:**

1. `cd main/integration`
2. `python -m streamlit run app.py`
3. Open [http://localhost:8501](http://localhost:8501)

**Dashboard only (minimal install):** if the full `requirements.txt` fails, CSVs are already in the repo:

```powershell
python -m pip install pandas streamlit plotly
python main/integration/merge_data.py
cd main/integration
python -m streamlit run app.py
```

**Master Table:** [master_incidents.csv](main/integration/master_incidents.csv)