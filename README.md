# OC-P4
# Credit Scoring Model
# Unbalanced learning

## Project scenario
*Prêt à dépenser* is a fintech startup specializing in consumer lending. They are looking to implement an AI-driven credit scoring model to evaluate loan applicants. This tool must be explainable so that customer relations officers can easily review the files for potential issues.

## Solutions
- Dataset description (different files, missing values, target analysis, etc)
- Data transformation :
  - creation of numerous synthetic features  
  - merge all files (identifying ids, perform aggregations, encode categorical features, etc)
  - model pipeline :
    - *ColumnTransformer* for preprocessing
    - *Selector* with different stpes ("multicollinearity, *variance threshold*, *F-score*-based selection, *RFECV*)
    - *sampling* with *SMOTE*
    - *model* (*Dummy*, *Random Forest*, *Logistic Regression*, *LightGBM*)
  - Evaluation with *Average Precision*
  - Decision threshold choice using cost-based formula
  - Explainability : *SHAP*
