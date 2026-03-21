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
We found the data sets from our own searches because we wanted to analyze something to do with women in STEM or sports or their general well being. There was an abundance of variables, so we used AI to narrow down our ideas.

## Brief Excerpt of One AI Exchange
Kimmy: If I wanted to use the GSS to practice machine learning focusing on variables about women in stem or women in sports or anything like that, what variables would you recommend?

Google Gemini: Using the General Social Survey (GSS) for machine learning is a great way to practice, though it’s important to know that while the GSS is massive, it is more "sociological" than "industrial." This means it focuses on attitudes, demographics, and broad job categories rather than specific play-by-play sports data or internal STEM corporate metrics.

For a machine learning project, you’ll likely want to set a **target variable** (like "Does this person work in a STEM field?") and use other variables as **features** to predict that outcome.

Here are the best variables for your specific interests:

#### 1. Women in STEM

The GSS uses standard census codes for occupations. To "find" women in STEM, you have to filter by gender and then look at the occupational and educational codes.

* **OCC10 (Occupation):** This is the holy grail. You will need to filter for codes related to "Computer and Mathematical," "Architecture and Engineering," and "Life, Physical, and Social Science."
* **MAJOR1 & MAJOR2 (College Major):** Added in recent decades, these tell you if the respondent’s degree was in a STEM field (e.g., Engineering, Biology).
* **CONSCI (Confidence in Science):** Great for a sentiment analysis or clustering project. Do women in STEM actually trust the scientific community more than those who aren't?
* **ADVSCI (Science and Technology Benefits):** Measures whether the respondent thinks science makes life healthier and easier.
* **SEI10 (Socioeconomic Index):** Useful for regression models to see if women in STEM have higher occupational prestige or "status" compared to other fields.

#### 2. Women in Sports

The GSS has less data on "professional athletes" but excellent data on **physical activity** and **cultural attitudes** toward gender in sports.

* **PHYSACTS (Physical Activities):** Asks how often the respondent engages in physical activities or sports. You can use this to predict "active" vs. "sedentary" lifestyles among women.
* **SATSOC (Satisfaction with Social/Leisure):** Useful to see if participation in sports/physical activity correlates with higher life satisfaction in women.
* **GENDERSC (Gender and Science/Sports):** In certain years, the GSS includes modules on whether people believe there are innate differences in ability between men and women in various fields (including sports).

#### 3. The "Gender Roles" Feature Set

If you are building a predictive model (e.g., "What factors predict a woman pursuing a high-level STEM career?"), these demographic and attitude variables are your best features:

| Variable Code | What it Measures | Why it’s good for ML |
| --- | --- | --- |
| SEX | Respondent's gender | Essential for a comparative model. |
| FECHLD | Attitude: "Working mother doesn't hurt children" | Measures traditionalism vs. progressivism. |
| FEFAM | Attitude: "Man is achiever, woman is home-maker" | High predictive power for career paths. |
| PAOCC10 / MAOCC10 | Father's/Mother's Occupation | Great for modeling "generational inheritance" in STEM. |
| EDUC / DEGREE | Years of Schooling / Highest Degree | Strongest predictor of occupation and income. |

---

#### Machine Learning Project Idea

**The "STEM Pipeline" Predictor:**
Try to build a classification model (like a Random Forest or Logistic Regression) that predicts if a respondent has a **STEM occupation (OCC10)** based on their **parents' occupations**, **years of education**, and **attitudes toward gender roles**.

This allows you to see which feature has the highest "importance"—for example, is a woman more likely to be in STEM because her mother was (environment), or because she holds progressive views on gender (attitude)?

## Final Proposal

**Research Question:** What determines women's quality of life or happiness?

**Target variable for Supervised analysis:**
Happy

**Data:** A subset of the GSS (see data folder for more info)

**Timeline:** We'll figure it out.

**Ethical/legal considerations:** Free to access, and it's literally made to analyze society, so we're helping.

**Planned additional ML methods:** Classification, Regression, etc.