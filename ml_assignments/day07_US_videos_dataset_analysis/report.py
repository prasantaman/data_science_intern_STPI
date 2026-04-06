# ============================================
# FINAL REPORT - YOUTUBE TRENDING ANALYSIS
# ============================================

report = f"""
==================================================
        YOUTUBE TRENDING DATA ANALYSIS REPORT
==================================================

1. REGRESSION MODEL PERFORMANCE
--------------------------------------------------
Model Used       : Linear Regression
Target Variable  : Views

Summary:
- The regression model demonstrates good predictive capability.
- Evaluation metrics (MSE, RMSE, MAE, R²) indicate that the model 
  captures a significant portion of variance in views.

Key Predictors of Views:
- Likes (Strong Positive Influence)
- Comment Count (Strong Positive Influence)
- Dislikes (Moderate Influence)

Insight:
User engagement metrics are the primary drivers of video popularity.


2. CLASSIFICATION MODEL PERFORMANCE
--------------------------------------------------
Model Used       : Logistic Regression
Target Variable  : Viral (Top 25% Views)

Summary:
- The classification model achieved strong performance in terms of:
  Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- The model effectively distinguishes between viral and non-viral videos.

Key Predictors of Virality:
- Likes
- Comment Count
- Dislikes

Insight:
Higher engagement significantly increases the probability of a video going viral.


3. HYPOTHESIS TESTING INSIGHTS
--------------------------------------------------
Test Used        : Independent T-Test

Results:
- Extremely low p-values (≈ 0) observed for:
  Likes, Dislikes, Comment Count

Conclusion:
- Since p < 0.05, the null hypothesis is rejected.
- There is strong statistical evidence that higher engagement 
  leads to higher views.


4. EXPLORATORY INSIGHTS
--------------------------------------------------
- Views distribution is highly skewed, indicating viral outliers.
- Strong positive correlation between Likes and Views.
- Comment Count also positively correlates with Views.
- Title Length and Days Since Publish have minimal impact.


5. BUSINESS RECOMMENDATIONS
--------------------------------------------------
- Focus on increasing audience engagement (likes, comments).
- Encourage interaction through calls-to-action.
- Create emotionally engaging and discussion-driven content.
- Maintain consistency in content publishing.
- Prioritize engagement over metadata optimization.


6. FINAL CONCLUSION
--------------------------------------------------
- Engagement metrics are the strongest predictors of both views and virality.
- Machine learning models effectively captured these relationships.
- Data-driven strategies can significantly improve content performance.

==================================================
            PROJECT COMPLETED SUCCESSFULLY
==================================================
"""

# Display report
print(report)