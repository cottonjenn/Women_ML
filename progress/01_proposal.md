# Proposal

## Candidate Project Ideas
We gave claude.ai our 3 project ideas and 8 potential datasets and prompted it to help us consider top 5 machine learning projects given the instruction criteria and 4 week timeline. We received the top 3 ideas: 

### 1. Gender equity gaps in U.S. college athletics
**Data:** EADA (ope.ed.gov) · wehoop · NBA/WNBA salary gap

**Supervised:** Regression

**Unsupervised:** K-Means Clustering

**What you build:** Predict women's athletic budget share from school features (enrollment, conference, state, sport count). Then cluster schools into equity profiles — revealing whether underfunding concentrates by region, conference, or institution type.

**Why it works in 4 weeks:** EADA data downloads as a clean CSV in minutes. No scraping, no API keys. The ML pipeline is straightforward: regression → residual analysis → clustering on funding gap features. Division of labor is natural (data cleaning / supervised model / clustering + viz).

### 2. Predicting gender attitude shifts in America
**Data:** GSS 1972–2024 (gss.norc.org) — free download

**Supervised:** Classification

**Unsupervised:** Clustering

**What you build:** Classify whether a respondent holds traditional vs. egalitarian gender views using demographic features (age, education, religion, region, income). Then cluster respondents into attitudinal profiles — do the clusters match political lines, or reveal something more surprising?

**Why it works in 4 weeks:** GSS is one of the most analysis-ready datasets in social science. Variables like fefam, fepol, fepresch are pre-coded and well-documented. The only real work is feature selection and handling missing values.

### 3. Women's happiness gap — modeling wellbeing over time
**Data:** GSS 1972–2024 · happiness, life satisfaction, work/family variables

**Supervised:** Regression

**Unsupervised:** Anomaly Detection: Isolation Forest

**What you build:** Predict self-reported happiness from work, family, economic, and attitudinal features — with sex as a key predictor. Then use Isolation Forest or LOF anomaly detection to flag respondent profiles where predicted vs. actual happiness diverges most, surfacing hidden subgroups of women whose wellbeing defies the model.

**Why it works in 4 weeks:** Same GSS dataset as Idea 2, different angle. The anomaly detection component is compact to implement and produces compelling, interpretable results. Strong narrative hook: the "paradox of declining female happiness" is well-documented in research.

## AI Influence over Project Selection

## Brief Excerpt of One AI Exchange

## Final Proposal

**Research Question:**

**Target variable for Supervised analysis:**

**Data:**

**Timeline:**

**Ethical/legal considerations:**

**Planned additional ML methods:**