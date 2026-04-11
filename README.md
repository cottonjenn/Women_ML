# The Gender Happiness Gap — Streamlit Demo

**Project:** Predicting & Profiling Women's Wellbeing in America  
**Data Source:** General Social Survey (GSS) 1972–2024, NORC at the University of Chicago  
**Live app:** [https://women-ml.streamlit.app/](https://women-ml.streamlit.app/)

---

## Setup

1. Place your data file at `data/ACTUAL_qol.csv`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## App Sections

| Section | Description |
|---|---|
| Dataset Overview | Key statistics and variable descriptions |
| EDA: Happiness Trends | Gender happiness gap visualized over 50+ years |
| Supervised Learning | Logistic Regression + Random Forest classifiers |
| K-Means Clustering | Latent wellbeing profiles via unsupervised learning |


## Ethical Notes

- Analysis uses the recorded binary sex variable (pre-2021 data); findings do not represent the full spectrum of gender identity
- Models identify **predictive associations**, not causal relationships
- The post-2021 GSS mode shift (in-person → web) introduces a methodological discontinuity; results should be interpreted accordingly
