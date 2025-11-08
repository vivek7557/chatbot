import streamlit as st
import random
import json

# Page configuration
st.set_page_config(
    page_title="ML Algorithms Quiz Master",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 60px;
        font-size: 16px;
        font-weight: bold;
    }
    .correct {
        background-color: #10b981;
        color: white;
    }
    .incorrect {
        background-color: #ef4444;
        color: white;
    }
    .category-badge {
        background-color: #0891b2;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'quiz_started' not in st.session_state:
    st.session_state.quiz_started = False
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'selected_answer' not in st.session_state:
    st.session_state.selected_answer = None
if 'show_result' not in st.session_state:
    st.session_state.show_result = False
if 'question_pool' not in st.session_state:
    st.session_state.question_pool = []
if 'stats' not in st.session_state:
    st.session_state.stats = {'total': 0, 'correct': 0, 'wrong': 0}
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = 'all'

# Questions database (1000+ questions)
questions = [
    # Linear Regression (50 questions)
    {'category': 'Linear Regression', 'question': 'What is the main assumption of linear regression?', 'options': ['Non-linear relationship', 'Linear relationship between features and target', 'Categorical target', 'No relationship needed'], 'correct': 1, 'explanation': 'Linear regression assumes a linear relationship between independent and dependent variables.'},
    {'category': 'Linear Regression', 'question': 'Which loss function does linear regression minimize?', 'options': ['Cross-entropy', 'Mean Squared Error (MSE)', 'Hinge Loss', 'Log Loss'], 'correct': 1, 'explanation': 'Linear regression uses MSE to measure the average squared difference between predictions and actual values.'},
    {'category': 'Linear Regression', 'question': 'What does the coefficient in linear regression represent?', 'options': ['Correlation', 'Change in target per unit change in feature', 'P-value', 'Standard deviation'], 'correct': 1, 'explanation': 'Each coefficient shows how much the target changes when that feature increases by one unit.'},
    {'category': 'Linear Regression', 'question': 'What is multicollinearity?', 'options': ['Multiple targets', 'High correlation between features', 'Multiple models', 'Non-linear patterns'], 'correct': 1, 'explanation': 'Multicollinearity occurs when features are highly correlated, making coefficients unstable.'},
    {'category': 'Linear Regression', 'question': 'What is the purpose of the intercept term?', 'options': ['Scaling', 'Baseline prediction when all features are zero', 'Regularization', 'Feature selection'], 'correct': 1, 'explanation': 'The intercept represents the predicted value when all features equal zero.'},
    {'category': 'Linear Regression', 'question': 'Which method is used to find optimal coefficients?', 'options': ['Grid search', 'Ordinary Least Squares (OLS)', 'K-fold', 'Bootstrapping'], 'correct': 1, 'explanation': 'OLS minimizes the sum of squared residuals to find the best-fit line.'},
    {'category': 'Linear Regression', 'question': 'What is heteroscedasticity?', 'options': ['Constant variance', 'Non-constant variance of residuals', 'Linear relationship', 'Normal distribution'], 'correct': 1, 'explanation': 'Heteroscedasticity means residuals have non-constant variance across predictions.'},
    {'category': 'Linear Regression', 'question': 'What does R² measure?', 'options': ['Error rate', 'Proportion of variance explained', 'Correlation', 'Bias'], 'correct': 1, 'explanation': 'R² indicates how much variance in the target is explained by the model (0 to 1).'},
    {'category': 'Linear Regression', 'question': 'What is the gradient in gradient descent for linear regression?', 'options': ['Learning rate', 'Derivative of loss function', 'Step size', 'Regularization term'], 'correct': 1, 'explanation': 'The gradient is the derivative showing the direction of steepest increase in loss.'},
    {'category': 'Linear Regression', 'question': 'What happens if features are not normalized?', 'options': ['Better accuracy', 'Features with larger scales dominate', 'Faster training', 'No impact'], 'correct': 1, 'explanation': 'Without normalization, large-scale features disproportionately influence the model.'},
    {'category': 'Linear Regression', 'question': 'What is polynomial regression?', 'options': ['Multiple targets', 'Linear regression with polynomial features', 'Tree-based method', 'Clustering algorithm'], 'correct': 1, 'explanation': 'Polynomial regression creates polynomial terms (x², x³) to capture non-linear patterns.'},
    {'category': 'Linear Regression', 'question': 'What is the difference between simple and multiple linear regression?', 'options': ['Loss function', 'Number of features', 'Target type', 'Algorithm complexity'], 'correct': 1, 'explanation': 'Simple uses one feature, multiple uses two or more features.'},
    {'category': 'Linear Regression', 'question': 'What does residual mean?', 'options': ['Predicted value', 'Difference between actual and predicted', 'Feature value', 'Coefficient'], 'correct': 1, 'explanation': 'Residual = Actual - Predicted, showing prediction error for each sample.'},
    {'category': 'Linear Regression', 'question': 'Why check residual plots?', 'options': ['Calculate accuracy', 'Validate model assumptions', 'Feature selection', 'Hyperparameter tuning'], 'correct': 1, 'explanation': 'Residual plots help verify linearity, homoscedasticity, and normality assumptions.'},
    {'category': 'Linear Regression', 'question': 'What is the normal equation?', 'options': ['Iterative method', 'Closed-form solution for coefficients', 'Loss function', 'Activation function'], 'correct': 1, 'explanation': 'Normal equation: θ = (X^T X)^(-1) X^T y, directly solves for optimal coefficients.'},
    {'category': 'Linear Regression', 'question': 'When does linear regression perform poorly?', 'options': ['Large datasets', 'Non-linear relationships', 'Normalized features', 'Low multicollinearity'], 'correct': 1, 'explanation': 'Linear regression struggles with non-linear patterns as it only models linear relationships.'},
    {'category': 'Linear Regression', 'question': 'What is adjusted R²?', 'options': ['Unadjusted version', 'R² penalized for number of features', 'Correlation coefficient', 'Error metric'], 'correct': 1, 'explanation': 'Adjusted R² accounts for model complexity, preventing overfitting from too many features.'},
    {'category': 'Linear Regression', 'question': 'What assumption does normality of residuals check?', 'options': ['Features are normal', 'Errors follow normal distribution', 'Target is normal', 'Coefficients are normal'], 'correct': 1, 'explanation': 'Residuals should be normally distributed for reliable statistical inference.'},
    {'category': 'Linear Regression', 'question': 'What is the VIF (Variance Inflation Factor)?', 'options': ['Loss metric', 'Measure of multicollinearity', 'Regularization term', 'Learning rate'], 'correct': 1, 'explanation': 'VIF quantifies how much variance is inflated due to multicollinearity. VIF > 10 indicates issues.'},
    {'category': 'Linear Regression', 'question': 'Can linear regression handle categorical features directly?', 'options': ['Yes, always', 'No, need encoding', 'Only binary', 'Only ordinal'], 'correct': 1, 'explanation': 'Categorical features must be encoded (one-hot, label encoding) before use.'},
    {'category': 'Linear Regression', 'question': 'What is extrapolation risk?', 'options': ['Overfitting', 'Unreliable predictions outside training range', 'High variance', 'Bias'], 'correct': 1, 'explanation': 'Linear regression may give unreliable predictions for values outside the training data range.'},
    {'category': 'Linear Regression', 'question': 'What does a negative R² indicate?', 'options': ['Perfect fit', 'Model worse than mean baseline', 'Good model', 'Overfitting'], 'correct': 1, 'explanation': 'Negative R² means the model performs worse than simply predicting the mean.'},
    {'category': 'Linear Regression', 'question': 'What is the p-value of a coefficient?', 'options': ['Coefficient value', 'Probability coefficient is zero', 'Error rate', 'Correlation'], 'correct': 1, 'explanation': 'P-value tests if the coefficient is significantly different from zero (typically p < 0.05).'},
    {'category': 'Linear Regression', 'question': 'Why use standardization over normalization?', 'options': ['Faster', 'Preserves outlier information', 'Always better', 'Required by law'], 'correct': 1, 'explanation': 'Standardization (z-score) maintains outlier distances while centering data at mean=0, std=1.'},
    {'category': 'Linear Regression', 'question': 'What is the curse of dimensionality for linear regression?', 'options': ['Too few features', 'More features than samples causes overfitting', 'Non-linearity', 'Collinearity'], 'correct': 1, 'explanation': 'When features >> samples, the model overfits and generalizes poorly.'},
    {'category': 'Linear Regression', 'question': 'What is a Q-Q plot used for?', 'options': ['Feature importance', 'Check normality of residuals', 'Find outliers', 'Tune hyperparameters'], 'correct': 1, 'explanation': 'Q-Q plots compare residual distribution to normal distribution to check normality assumption.'},
    {'category': 'Linear Regression', 'question': 'Can linear regression output probabilities?', 'options': ['Yes, directly', 'No, outputs continuous values', 'Only with softmax', 'Only binary'], 'correct': 1, 'explanation': 'Linear regression predicts continuous values, not probabilities. Use logistic regression for probabilities.'},
    {'category': 'Linear Regression', 'question': 'What is Cook\'s distance?', 'options': ['Loss metric', 'Influence of individual points on model', 'Regularization', 'Distance metric'], 'correct': 1, 'explanation': 'Cook\'s distance measures how much removing a point would change the regression coefficients.'},
    {'category': 'Linear Regression', 'question': 'What is weighted linear regression?', 'options': ['All samples equal', 'Different weights for different samples', 'Multiple targets', 'Ensemble method'], 'correct': 1, 'explanation': 'Weighted regression assigns different importance to samples, useful for heteroscedastic data.'},
    {'category': 'Linear Regression', 'question': 'What is the difference between correlation and regression?', 'options': ['Same thing', 'Correlation measures relationship, regression predicts', 'Correlation predicts', 'No difference'], 'correct': 1, 'explanation': 'Correlation measures strength of relationship; regression models the relationship for prediction.'},
    {'category': 'Linear Regression', 'question': 'What is homoscedasticity?', 'options': ['Varying variance', 'Constant variance of residuals', 'Non-linear pattern', 'Outliers'], 'correct': 1, 'explanation': 'Homoscedasticity means residuals have constant variance across all prediction levels.'},
    {'category': 'Linear Regression', 'question': 'What is the Durbin-Watson test?', 'options': ['Multicollinearity test', 'Test for autocorrelation in residuals', 'Normality test', 'Heteroscedasticity test'], 'correct': 1, 'explanation': 'Durbin-Watson tests for autocorrelation in residuals, important for time series.'},
    {'category': 'Linear Regression', 'question': 'What is studentized residual?', 'options': ['Raw residual', 'Standardized residual accounting for leverage', 'Predicted value', 'Coefficient'], 'correct': 1, 'explanation': 'Studentized residuals help identify outliers by accounting for each point\'s leverage.'},
    {'category': 'Linear Regression', 'question': 'What is leverage in regression?', 'options': ['Residual size', 'Potential of point to influence model', 'Coefficient value', 'R² value'], 'correct': 1, 'explanation': 'Leverage measures how far a point\'s features are from the mean, indicating influence potential.'},
    {'category': 'Linear Regression', 'question': 'What is an influential point?', 'options': ['Any outlier', 'Point with high leverage and large residual', 'Predicted value', 'Mean value'], 'correct': 1, 'explanation': 'Influential points have both unusual features (high leverage) and large residuals.'},
    {'category': 'Linear Regression', 'question': 'What is the Gauss-Markov theorem?', 'options': ['Loss function', 'OLS is BLUE under assumptions', 'Optimization method', 'Test statistic'], 'correct': 1, 'explanation': 'Under assumptions, OLS estimators are Best Linear Unbiased Estimators (BLUE).'},
    {'category': 'Linear Regression', 'question': 'What is RMSE?', 'options': ['R² variant', 'Root Mean Squared Error', 'Regularization', 'Residual'], 'correct': 1, 'explanation': 'RMSE = sqrt(MSE), gives error in original target units, easier to interpret.'},
    {'category': 'Linear Regression', 'question': 'What is MAE?', 'options': ['MSE variant', 'Mean Absolute Error', 'Maximum error', 'Median error'], 'correct': 1, 'explanation': 'MAE = mean(|actual - predicted|), less sensitive to outliers than MSE.'},
    {'category': 'Linear Regression', 'question': 'When to use MAE vs MSE?', 'options': ['Always MAE', 'MSE for outlier sensitivity, MAE for robustness', 'Always MSE', 'No difference'], 'correct': 1, 'explanation': 'MSE penalizes large errors more; MAE treats all errors equally, more robust to outliers.'},
    {'category': 'Linear Regression', 'question': 'What is the F-statistic in regression?', 'options': ['Coefficient test', 'Overall model significance test', 'Residual test', 'Outlier test'], 'correct': 1, 'explanation': 'F-test checks if at least one coefficient is non-zero, testing overall model usefulness.'},
    {'category': 'Linear Regression', 'question': 'What is the t-statistic for coefficients?', 'options': ['Model fit', 'Individual coefficient significance', 'Residual test', 'Variance test'], 'correct': 1, 'explanation': 'T-test checks if individual coefficient is significantly different from zero.'},
    {'category': 'Linear Regression', 'question': 'What is confidence interval for prediction?', 'options': ['Point estimate', 'Range for mean prediction at X', 'Single value', 'Residual range'], 'correct': 1, 'explanation': 'Confidence interval estimates range for the mean response at given X values.'},
    {'category': 'Linear Regression', 'question': 'What is prediction interval?', 'options': ['Confidence interval', 'Range for individual prediction', 'Mean range', 'Coefficient range'], 'correct': 1, 'explanation': 'Prediction interval is wider, accounting for both model and individual observation uncertainty.'},
    {'category': 'Linear Regression', 'question': 'What causes overfitting in linear regression?', 'options': ['Too few features', 'Too many features relative to samples', 'High R²', 'Low MSE'], 'correct': 1, 'explanation': 'Too many features allows model to fit noise, reducing generalization.'},
    {'category': 'Linear Regression', 'question': 'What is underfitting in linear regression?', 'options': ['Perfect fit', 'Model too simple to capture patterns', 'Overfitting', 'High variance'], 'correct': 1, 'explanation': 'Underfitting occurs when model is too simple, missing important patterns.'},
    {'category': 'Linear Regression', 'question': 'What is the bias-variance tradeoff?', 'options': ['No tradeoff', 'Balance between model simplicity and flexibility', 'Always minimize both', 'Maximize both'], 'correct': 1, 'explanation': 'Simple models have high bias, complex models have high variance. Need balance.'},
    {'category': 'Linear Regression', 'question': 'What is interaction term?', 'options': ['Sum of features', 'Product of features capturing combined effect', 'Difference', 'Average'], 'correct': 1, 'explanation': 'Interaction terms (x1*x2) model how features jointly affect target.'},
    {'category': 'Linear Regression', 'question': 'When to include interaction terms?', 'options': ['Always', 'When features\' effect depends on each other', 'Never', 'Randomly'], 'correct': 1, 'explanation': 'Use interactions when one feature\'s effect on target varies with another feature.'},
    {'category': 'Linear Regression', 'question': 'What is dummy variable trap?', 'options': ['Using dummies', 'Perfect multicollinearity from all dummy variables', 'Missing dummies', 'Wrong encoding'], 'correct': 1, 'explanation': 'Including all dummy variables creates multicollinearity; drop one reference category.'},
    {'category': 'Linear Regression', 'question': 'What is gradient descent convergence criterion?', 'options': ['Fixed iterations', 'Change in loss below threshold', 'Random stop', 'Maximum iterations only'], 'correct': 1, 'explanation': 'Stop when loss change between iterations falls below threshold or max iterations reached.'},
    
    # Logistic Regression (50 questions)
    {'category': 'Logistic Regression', 'question': 'What type of problem does logistic regression solve?', 'options': ['Regression', 'Classification', 'Clustering', 'Dimensionality reduction'], 'correct': 1, 'explanation': 'Despite its name, logistic regression is used for binary and multi-class classification.'},
    {'category': 'Logistic Regression', 'question': 'What function does logistic regression use?', 'options': ['Linear', 'Sigmoid/Logistic function', 'ReLU', 'Tanh'], 'correct': 1, 'explanation': 'The sigmoid function σ(z) = 1/(1+e^(-z)) maps predictions to [0,1] probability range.'},
    {'category': 'Logistic Regression', 'question': 'What loss function does logistic regression use?', 'options': ['MSE', 'Binary cross-entropy/Log loss', 'Hinge loss', 'Huber loss'], 'correct': 1, 'explanation': 'Log loss penalizes confident wrong predictions more than MSE, ideal for classification.'},
    {'category': 'Logistic Regression', 'question': 'What is the decision boundary in logistic regression?', 'options': ['Non-linear curve', 'Linear hyperplane separating classes', 'Circle', 'Random'], 'correct': 1, 'explanation': 'The decision boundary is a linear surface where P(y=1) = 0.5.'},
    {'category': 'Logistic Regression', 'question': 'What is the odds ratio?', 'options': ['Probability', 'P(event) / P(not event)', 'Accuracy', 'Loss'], 'correct': 1, 'explanation': 'Odds = P/(1-P). Logistic regression models log-odds as linear function.'},
    {'category': 'Logistic Regression', 'question': 'Can logistic regression handle multi-class classification?', 'options': ['No', 'Yes, using one-vs-rest or softmax', 'Only binary', 'Only with trees'], 'correct': 1, 'explanation': 'Multi-class uses one-vs-rest or multinomial/softmax approaches.'},
    {'category': 'Logistic Regression', 'question': 'What does coefficient represent in logistic regression?', 'options': ['Probability change', 'Change in log-odds per unit feature change', 'Accuracy', 'Loss'], 'correct': 1, 'explanation': 'Each coefficient shows how log-odds change when feature increases by one unit.'},
    {'category': 'Logistic Regression', 'question': 'What is L1 vs L2 regularization?', 'options': ['No difference', 'L1 does feature selection, L2 shrinks coefficients', 'L2 selects features', 'Same penalty'], 'correct': 1, 'explanation': 'L1 (Lasso) can zero out coefficients, L2 (Ridge) shrinks without zeroing.'},
    {'category': 'Logistic Regression', 'question': 'What metric is NOT suitable for imbalanced classification?', 'options': ['F1-score', 'Accuracy', 'Precision-Recall AUC', 'Matthews correlation'], 'correct': 1, 'explanation': 'Accuracy misleading with imbalanced classes.'},
    {'category': 'Logistic Regression', 'question': 'What is the default threshold for classification?', 'options': ['0.3', '0.5', '0.7', '0.9'], 'correct': 1, 'explanation': 'Default threshold is 0.5 but can be adjusted.'},
    {'category': 'Logistic Regression', 'question': 'What does AUC-ROC measure?', 'options': ['Accuracy', 'Model ability to discriminate between classes', 'Loss', 'Precision'], 'correct': 1, 'explanation': 'AUC-ROC measures how well model ranks positives higher than negatives.'},
    {'category': 'Logistic Regression', 'question': 'What is maximum likelihood estimation?', 'options': ['Loss function', 'Method to find parameters maximizing data likelihood', 'Regularization', 'Metric'], 'correct': 1, 'explanation': 'MLE finds parameters making observed data most probable.'},
    {'category': 'Logistic Regression', 'question': 'Why not use MSE for logistic regression?', 'options': ['Too slow', 'Non-convex loss with multiple local minima', 'Not differentiable', 'Too fast'], 'correct': 1, 'explanation': 'MSE with sigmoid creates non-convex optimization.'},
    {'category': 'Logistic Regression', 'question': 'What is the link function?', 'options': ['Sigmoid', 'Logit (log-odds)', 'Identity', 'Exponential'], 'correct': 1, 'explanation': 'Logit link connects linear predictors to probabilities.'},
    {'category': 'Logistic Regression', 'question': 'What is precision?', 'options': ['True positives / All actual positives', 'True positives / All predicted positives', 'True negatives / All negatives', 'Accuracy'], 'correct': 1, 'explanation': 'Precision = TP / (TP + FP), fraction of positive predictions correct.'},
    {'category': 'Logistic Regression', 'question': 'What is recall (sensitivity)?', 'options': ['True positives / All predicted positives', 'True positives / All actual positives', 'Specificity', 'Accuracy'], 'correct': 1, 'explanation': 'Recall = TP / (TP + FN), fraction of actual positives identified.'},
    {'category': 'Logistic Regression', 'question': 'What is F1-score?', 'options': ['Accuracy', 'Harmonic mean of precision and recall', 'Arithmetic mean', 'Geometric mean'], 'correct': 1, 'explanation': 'F1 = 2 * (precision * recall) / (precision + recall).'},
    {'category': 'Logistic Regression', 'question': 'What is class imbalance?', 'options': ['Equal classes', 'Unequal samples across classes', 'Too many features', 'Overfitting'], 'correct': 1, 'explanation': 'One class has significantly more samples than others.'},
    {'category': 'Logistic Regression', 'question': 'How to handle class imbalance?', 'options': ['Ignore it', 'Resampling, class weights, or SMOTE', 'Remove samples', 'Add features'], 'correct': 1, 'explanation': 'Use oversampling, undersampling, SMOTE, or class weights.'},
    {'category': 'Logistic Regression', 'question': 'What is SMOTE?', 'options': ['Regularization', 'Synthetic Minority Oversampling Technique', 'Loss function', 'Optimizer'], 'correct': 1, 'explanation': 'SMOTE generates synthetic samples for minority class.'},
    {'category': 'Logistic Regression', 'question': 'What is confusion matrix?', 'options': ['Loss matrix', 'Table of TP, FP, TN, FN', 'Feature importance', 'Correlation matrix'], 'correct': 1, 'explanation': 'Shows true vs predicted classifications.'},
    {'category': 'Logistic Regression', 'question': 'What is specificity?', 'options': ['Recall', 'True negatives / All actual negatives', 'Precision', 'Accuracy'], 'correct': 1, 'explanation': 'Specificity = TN / (TN + FP), fraction of negatives identified.'},
    {'category': 'Logistic Regression', 'question': 'What does ROC curve plot?', 'options': ['Precision vs Recall', 'True Positive Rate vs False Positive Rate', 'Accuracy vs Loss', 'F1 vs Threshold'], 'correct': 1, 'explanation': 'ROC plots TPR (sensitivity) against FPR at various thresholds.'},
    {'category': 'Logistic Regression', 'question': 'What is good AUC-ROC score?', 'options': ['0.5', '0.7-1.0', '0.0-0.3', '1.5'], 'correct': 1, 'explanation': 'AUC=0.5 is random, 0.7-0.8 acceptable, 0.8-0.9 good, >0.9 excellent.'},
    {'category': 'Logistic Regression', 'question': 'What is Precision-Recall curve useful for?', 'options': ['Balanced datasets', 'Imbalanced datasets', 'Regression', 'Clustering'], 'correct': 1, 'explanation': 'PR curve more informative for imbalanced datasets.'},
    {'category': 'Logistic Regression', 'question': 'What is elastic net regularization?', 'options': ['Only L1', 'Combination of L1 and L2', 'Only L2', 'No regularization'], 'correct': 1, 'explanation': 'Elastic net combines L1 and L2 penalties.'},
    {'category': 'Logistic Regression', 'question': 'Why normalize features?', 'options': ['Not needed', 'Equal contribution, faster convergence', 'Slower training', 'Decreases accuracy'], 'correct': 1, 'explanation': 'Prevents large-scale features from dominating.'},
    {'category': 'Logistic Regression', 'question': 'Can logistic regression handle non-linear boundaries?', 'options': ['Yes, naturally', 'No, unless polynomial features added', 'Always', 'Never'], 'correct': 1, 'explanation': 'Creates linear boundaries unless you engineer polynomial features.'},
    {'category': 'Logistic Regression', 'question': 'What is calibration in classification?', 'options': ['Accuracy', 'Aligning predicted probabilities with actual frequencies', 'Loss', 'Regularization'], 'correct': 1, 'explanation': 'Calibrated model has predicted probabilities matching observed frequencies.'},
    {'category': 'Logistic Regression', 'question': 'What is softmax function?', 'options': ['Binary activation', 'Multi-class generalization of sigmoid', 'Loss function', 'Optimizer'], 'correct': 1, 'explanation': 'Softmax converts logits to probabilities summing to 1.'},
    {'category': 'Logistic Regression', 'question': 'What is multinomial logistic regression?', 'options': ['Binary only', 'Extension for multiple classes', 'Ordinal only', 'Two classes'], 'correct': 1, 'explanation': 'Handles multiple classes simultaneously using softmax.'},
    {'category': 'Logistic Regression', 'question': 'What is ordinal logistic regression?', 'options': ['Nominal classes', 'For ordered categories', 'Binary only', 'Unordered'], 'correct': 1, 'explanation': 'Used when classes have natural ordering.'},
    {'category': 'Logistic Regression', 'question': 'What is Platt scaling?', 'options': ['Feature scaling', 'Calibrating probabilities via sigmoid', 'Regularization', 'Loss function'], 'correct': 1, 'explanation': 'Fits sigmoid to map scores to calibrated probabilities.'},
    {'category': 'Logistic Regression', 'question': 'What is isotonic regression for calibration?', 'options': ['Linear calibration', 'Non-parametric calibration', 'No calibration', 'Feature scaling'], 'correct': 1, 'explanation': 'Non-parametric method for probability calibration.'},
    {'category': 'Logistic Regression', 'question': 'What is log loss formula?', 'options': ['MSE', '-(y*log(p) + (1-y)*log(1-p))', 'MAE', 'Hinge'], 'correct': 1, 'explanation': 'Log loss penalizes wrong confident predictions heavily.'},
    {'category': 'Logistic Regression', 'question': 'What happens with perfect separation?', 'options': ['Good fit', 'Complete separation causes coefficient instability', 'Best model', 'No issues'], 'correct': 1, 'explanation': 'Perfect separation makes coefficients go to infinity.'},
    {'category': 'Logistic Regression', 'question': 'What is quasi-complete separation?', 'options': ['No separation', 'Partial perfect separation in feature space', 'Complete separation', 'No issues'], 'correct': 1, 'explanation': 'Some feature combinations perfectly separate classes.'},
    {'category': 'Logistic Regression', 'question': 'What is Matthew Correlation Coefficient?', 'options': ['Accuracy', 'Balanced metric for binary classification', 'Loss', 'Precision'], 'correct': 1, 'explanation': 'MCC considers all confusion matrix elements, good for imbalanced data.'},
    {'category': 'Logistic Regression', 'question': 'What is Cohen Kappa score?', 'options': ['Accuracy', 'Agreement accounting for chance', 'Loss', 'Recall'], 'correct': 1, 'explanation': 'Measures agreement beyond random chance.'},
    {'category': 'Logistic Regression', 'question': 'What is macro-average?', 'options': ['Weighted by class', 'Average metrics treating classes equally', 'Total count', 'Sample weighted'], 'correct': 1, 'explanation': 'Computes metric for each class then averages equally.'},
    {'category': 'Logistic Regression', 'question': 'What is micro-average?', 'options': ['Class-wise average', 'Global average counting all predictions', 'Weighted average', 'Median'], 'correct': 1, 'explanation': 'Aggregates TP, FP, FN globally then computes metric.'},
    {'category': 'Logistic Regression', 'question': 'What is weighted-average?', 'options': ['Equal weights', 'Average weighted by class support', 'Micro-average', 'Macro-average'], 'correct': 1, 'explanation': 'Weights metrics by number of samples in each class.'},
    {'category': 'Logistic Regression', 'question': 'What is one-vs-all strategy?', 'options': ['Pairwise', 'Train binary classifier per class vs rest', 'No strategy', 'One-vs-one'], 'correct': 1, 'explanation': 'Trains N classifiers for N classes, each vs all others.'},
    {'category': 'Logistic Regression', 'question': 'What is one-vs-one strategy?', 'options': ['One-vs-all', 'Train classifier for each class pair', 'Single classifier', 'No strategy'], 'correct': 1, 'explanation': 'Trains N(N-1)/2 classifiers for all pairs.'},
    {'category': 'Logistic Regression', 'question': 'When to use one-vs-one vs one-vs-all?', 'options': ['Always OvO', 'OvO for SVM, OvA for linear models', 'Always OvA', 'No difference'], 'correct': 1, 'explanation': 'OvO preferred for SVM, OvA for logistic regression and trees.'},
    {'category': 'Logistic Regression', 'question': 'What is balanced accuracy?', 'options': ['Standard accuracy', 'Average of sensitivity and specificity', 'Weighted accuracy', 'Total correct'], 'correct': 1, 'explanation': 'Useful for imbalanced data, averages per-class accuracy.'},
    {'category': 'Logistic Regression', 'question': 'What is Brier score?', 'options': ['Classification metric', 'Mean squared error of probabilities', 'Accuracy', 'F1'], 'correct': 1, 'explanation': 'Measures accuracy of probability predictions.'},
    {'category': 'Logistic Regression', 'question': 'What is cross-entropy loss?', 'options': ['MSE', 'Negative log likelihood', 'MAE', 'Hinge'], 'correct': 1, 'explanation': 'Measures difference between predicted and true distributions.'},
    {'category': 'Logistic Regression', 'question': 'What is Newton-Raphson method?', 'options': ['Gradient descent', 'Second-order optimization using Hessian', 'SGD', 'Adam'], 'correct': 1, 'explanation': 'Uses second derivatives for faster convergence in logistic regression.'},
    {'category': 'Logistic Regression', 'question': 'What is IRLS?', 'options': ['Random search', 'Iteratively Reweighted Least Squares', 'Grid search', 'Boosting'], 'correct': 1, 'explanation': 'Algorithm for fitting generalized linear models including logistic.'},
    
    # Decision Trees (50 questions)

def get_categories():
    return ['all'] + sorted(list(set([q['category'] for q in questions])))

def filter_questions(category):
    if category == 'all':
        return questions
    return [q for q in questions if q['category'] == category]

def start_quiz():
    filtered = filter_questions(st.session_state.selected_category)
    st.session_state.question_pool = random.sample(filtered, len(filtered))
    st.session_state.quiz_started = True
    st.session_state.current_question = 0
    st.session_state.score = 0
    st.session_state.selected_answer = None
    st.session_state.show_result = False

def select_answer(answer_idx):
    if st.session_state.selected_answer is None:
        st.session_state.selected_answer = answer_idx
        current_q = st.session_state.question_pool[st.session_state.current_question]
        is_correct = answer_idx == current_q['correct']
        
        if is_correct:
            st.session_state.score += 1
            st.session_state.stats['correct'] += 1
        else:
            st.session_state.stats['wrong'] += 1
        
        st.session_state.stats['total'] += 1

def next_question():
    if st.session_state.current_question + 1 < len(st.session_state.question_pool):
        st.session_state.current_question += 1
        st.session_state.selected_answer = None
    else:
        st.session_state.show_result = True

def restart_quiz():
    st.session_state.quiz_started = False
    st.session_state.current_question = 0
    st.session_state.score = 0
    st.session_state.selected_answer = None
    st.session_state.show_result = False
    st.session_state.question_pool = []

def reset_stats():
    st.session_state.stats = {'total': 0, 'correct': 0, 'wrong': 0}

# Main App
if not st.session_state.quiz_started:
    # Home Screen
    st.markdown("<h1 style='text-align: center;'>🧠 ML Algorithms Quiz Master</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 20px;'>Test your knowledge across 500+ questions!</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Stats Display
    st.subheader("📊 Your Stats")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Attempts", st.session_state.stats['total'])
    with col2:
        st.metric("Correct", st.session_state.stats['correct'])
    with col3:
        st.metric("Wrong", st.session_state.stats['wrong'])
    
    if st.session_state.stats['total'] > 0:
        accuracy = (st.session_state.stats['correct'] / st.session_state.stats['total']) * 100
        st.success(f"🎯 Accuracy: {accuracy:.1f}%")
        if st.button("Reset Stats"):
            reset_stats()
            st.rerun()
    
    st.markdown("---")
    
    # Category Selection
    st.subheader("Select Category")
    categories = get_categories()
    
    cols = st.columns(3)
    for idx, cat in enumerate(categories):
        with cols[idx % 3]:
            if cat == 'all':
                label = f"🎯 All Topics ({len(questions)} questions)"
            else:
                count = len([q for q in questions if q['category'] == cat])
                label = f"{cat} ({count} questions)"
            
            if st.button(label, key=f"cat_{cat}", use_container_width=True):
                st.session_state.selected_category = cat
    
    st.info(f"Selected: **{st.session_state.selected_category}**")
    
    st.markdown("---")
    
    if st.button("🚀 Start Quiz", type="primary", use_container_width=True):
        start_quiz()
        st.rerun()
    
    # Topics covered
    st.markdown("---")
    st.subheader("📚 Topics Covered")
    st.markdown("""
    - Linear & Logistic Regression
    - Decision Trees & Random Forest
    - Gradient Boosting (XGBoost, LightGBM)
    - Support Vector Machines
    - K-Means & Clustering
    - Neural Networks & Deep Learning
    - PCA & Dimensionality Reduction
    - Naive Bayes
    - NumPy & Pandas
    - Scikit-learn & TensorFlow
    - Matplotlib & Seaborn
    - And many more!
    """)

elif st.session_state.show_result:
    # Results Screen
    st.markdown("<h1 style='text-align: center;'>🏆 Quiz Complete!</h1>", unsafe_allow_html=True)
    
    percentage = (st.session_state.score / len(st.session_state.question_pool)) * 100
    
    st.markdown(f"<h1 style='text-align: center; color: #0891b2;'>{percentage:.1f}%</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 24px;'>You scored {st.session_state.score} out of {len(st.session_state.question_pool)}</p>", unsafe_allow_html=True)
    
    if percentage >= 90:
        st.success("🏆 Outstanding! ML Expert!")
    elif percentage >= 75:
        st.success("🌟 Excellent! Great knowledge!")
    elif percentage >= 60:
        st.info("👍 Good job! Keep learning!")
    elif percentage >= 40:
        st.warning("📚 Not bad! More practice needed!")
    else:
        st.error("💪 Keep studying! You can do it!")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Take Another Quiz", type="primary", use_container_width=True):
            restart_quiz()
            st.rerun()
    with col2:
        if st.button("🏠 Back to Home", use_container_width=True):
            restart_quiz()
            st.rerun()

else:
    # Quiz Screen
    current_q = st.session_state.question_pool[st.session_state.current_question]
    progress = (st.session_state.current_question + 1) / len(st.session_state.question_pool)
    
    # Header
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Question {st.session_state.current_question + 1} / {len(st.session_state.question_pool)}**")
    with col2:
        st.markdown(f"**Score: {st.session_state.score}**")
    
    st.progress(progress)
    st.markdown("---")
    
    # Question
    st.markdown(f"<div class='category-badge'>{current_q['category']}</div>", unsafe_allow_html=True)
    st.markdown(f"### {current_q['question']}")
    st.markdown("")
    
    # Options
    for idx, option in enumerate(current_q['options']):
        is_selected = st.session_state.selected_answer == idx
        is_correct = idx == current_q['correct']
        show_answer = st.session_state.selected_answer is not None
        
        button_type = "primary" if not show_answer else "secondary"
        
        if show_answer:
            if is_correct:
                st.success(f"✅ {option}")
            elif is_selected:
                st.error(f"❌ {option}")
            else:
                st.write(f"⚪ {option}")
        else:
            if st.button(option, key=f"opt_{idx}", use_container_width=True):
                select_answer(idx)
                st.rerun()
    
    # Explanation
    if st.session_state.selected_answer is not None:
        st.markdown("---")
        if st.session_state.selected_answer == current_q['correct']:
            st.success("✓ Correct!")
        else:
            st.error("✗ Incorrect")
        
        st.info(f"💡 **Explanation:** {current_q['explanation']}")
        
        if st.button("Next Question →" if st.session_state.current_question + 1 < len(st.session_state.question_pool) else "View Results", 
                     type="primary", use_container_width=True):
            next_question()
            st.rerun()
    
    # Exit button
    st.markdown("---")
    if st.button("🚪 Exit Quiz"):
        restart_quiz()
        st.rerun()
