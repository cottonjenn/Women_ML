# Unsupervised Modeling Checkpoint

### 1) Problem Context and Research Question
The goal of this project is to predict individual happiness levels, specifically identifying factors that lead to being "very happy" vs otherwise ("pretty happy", "not too happy", or both), using demographic and quality-of-life variables from a subset of the GSS (General Social Survey) dataset. This analysis seeks to determine: "Which life factors (e.g., health, marital status, work status, etc.) are the strongest predictors of subjective well-being AKA happiness?"

### 2) Unsupervised Model
We implemented a K-means clustering algorithm as an unsupervised learning method. 

| Model Type | Key Hyperparameters Explored | Validation Setup | Performance Metrics (Final Values) |
| :--- | :--- | :--- | :--- |
| **K-means Clustering** | N clsuters = 3, 4, 6 | Silhouette scores, Inertia (WCSS) | Davies-Bouldin Index for K=4: 1.5752, Purity Score for K=4: 0.5529, Davies-Bouldin Index for K=6: 1.6080, Purity Score for K=6: 0.5912  |

![Diagram](/work/images/elbow.png)

### 3) Model Comparison and Selection
* **Selecting K**: Pronounced elbows can be seen at K=4 and K=6 clusters. There was some initial interest in clustering at K=3 as well. When looking at the silhouette scores for each K, both K=4 and K=6 were better than K=3 by about 0.004-0.005. Regardless, none of the scores indicate great clustering.
Silhouette Score for k=3: 0.1808
Silhouette Score for k=4: 0.1844
Silhouette Score for k=6: 0.1856

The elbow plot shows the WCSS for each cluster. 
![Elbow](/work/images/elbow.png)

* **Best Model**: The k=4 and k=6 clustering led to virtually similar results. Using external validation against our labels, the k=6 cluster had a slightly better purity metric at ~0.591 compared to k=4 at ~0.553. However, looking at the Davies-Boulin Index (DBI) for both K's, the K=4 group was slightly better. Either way, neither group does well at clustering. The following pair plot shows the results of the K=4 group. Overlap in most of the clusters on most features indicate a weak ability to cluster.

![Pairs](/work/images/pairs.png)

* **Challenges**: The original data includes structural NAs for variables such as years_married. These NAs are helpful when using supervised methods because they can contribute to the overal learning, but when clustering it is essential for all NAs to be dropped This leads to all variables that have never been married being dropped. We may be losing some observations that could help improve clustering.

### 4) Explainability and Interpretability
To visualize the clustering, we created a tSNE visualization, plotting our clusters in a 2D field. 

![tSNE](/work/images/tsne.png)

The t-SNE plots reveal an interesting structure in the data. In the left plot (K=4), the clusters show overlap with little separation between groups, making them difficult to clearly distinguish. In the right plot (K=6), additional clusters introduce more distinctions, but there is still overlap. While increasing the number of clusters highlights potential definition, it does not improve the overall separation of the data.


### 5) Final Takeaways
The unsupervised model we chose to use adds little value to what the supervised models have already been developed. The overlap between clusters and inability to effectively group the data shows us that the data does not naturally seperate itself. This could indicate that correct classification may be difficult on this dataset. Our best supervised model was able to achieve 73.7% accuracy, which is impressive considering the data does not naturally group, as shown by the k-means clustering approach.

The fact that our best model achieves 74% accuracy despite the data not separating naturally into clusters suggests the predictive signal comes from specific feature combinations rather than latent group structure.