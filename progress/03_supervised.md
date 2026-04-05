# Supervised Modeling Checkpoint

### 1) Problem Context and Research Question
The goal of this project is to predict individual happiness levels, specifically identifying factors that lead to being "very happy" vs otherwise ("pretty happy", "not too happy", or both), using demographic and quality-of-life variables from a subset of the GSS (General Social Survey) dataset. This analysis seeks to determine: "Which life factors (e.g., health, marital status, work status, etc.) are the strongest predictors of subjective well-being AKA happiness?"

### 2) Supervised Models Implemented
We implemented a baseline Logistic Regression model and a more complex Random Forest Classifier. 

| Model Type | Key Hyperparameters Explored | Validation Setup | Performance Metrics (Final Values) |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Default (LBFGS solver) | 80/20 Train-Test Split with SMOTE resampling | Accuracy: ~73.6% |
| **Random Forest (Baseline Model)** | `n_estimators=300`, `max_depth=15`, `min_samples_leaf=5`, `class_weight='balanced'` | 80/20 Train-Test Split & 10-fold Cross-Validation | Accuracy: 59.7% (Test), 60.15% (CV Mean) |
| **Random Forest (Optimized Model)** | `n_estimators=300`, `max_depth=15`, `min_samples_leaf=5`, `class_weight='balanced'` | 80/20 Train-Test Split & 10-fold Cross-Validation | Accuracy: 73.7% (Test), 74.2% (CV Mean) |

*Note: The Logistic Regression model focused on a binary "Very Happy" vs. "Not Happy" subset, while the Random Forests explored all three options and also explored a binary "Is Very Happy" classification across the full dataset.*

### 3) Model Comparison and Selection
* **Patterns and Trends**: Both models identified marital status and satisfaction (`hapmar`) and life excitement (`life_new`) as critical predictors. The optimized Random Forest model demonstrated slightly higher stability through cross-validation, maintaining a mean accuracy of 74.2% (slightly better than the Logistic regression accuracy of 73.6%).
* **Best Model**: The **Random Forest** performed best. While its accuracy was only slightly better to the tuned Logistic Regression, it handled the inherent class imbalance more effectively using the `balanced` class weight parameter rather than requiring external resampling like SMOTE.
* **Challenges**: We encountered significant class imbalance, 31% of the respondants are "very happy", while 55% are "pretty happy", and 14% are "not too happy". Additionally, the Logistic Regression initially faced convergence issues until data was properly preprocessed.

### 4) Explainability and Interpretability
We used **Permutation Importance** to interpret the Random Forest model. 


The analysis showed that **Marital Happiness (`hapmar`)** was by far the most influential feature, followed by **Life Excitement (`life_new`)**. This suggests that the model relies heavily on self-reported quality-of-life assessments rather than purely demographic data like age or education to predict high happiness levels. Unsurprising as social science data is typically more intangible than many physical sciences.

### 5) Final Takeaways
The supervised learning analysis confirms that subjective quality-of-life indicators are stronger predictors of happiness than objective demographic variables. By achieving ~74% accuracy, we can conclude that while happiness is complex, it is significantly correlated with specific, measurable life domains—primarily social (marriage) and physical (health) well-being. This directly answers our research question by ranking the relative importance of these life factors in predicting a "very happy" outcome.