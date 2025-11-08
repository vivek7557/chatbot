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

# Questions database (first 50 questions as example - add all 530+ from the React app)
questions = [
    # Linear Regression
    {'category': 'Linear Regression', 'question': 'What is the main assumption of linear regression?', 'options': ['Non-linear relationship', 'Linear relationship between features and target', 'Categorical target', 'No relationship needed'], 'correct': 1, 'explanation': 'Linear regression assumes a linear relationship between independent and dependent variables.'},
    {'category': 'Linear Regression', 'question': 'Which loss function does linear regression minimize?', 'options': ['Cross-entropy', 'Mean Squared Error (MSE)', 'Hinge Loss', 'Log Loss'], 'correct': 1, 'explanation': 'Linear regression uses MSE to measure the average squared difference between predictions and actual values.'},
    {'category': 'Linear Regression', 'question': 'What does the coefficient in linear regression represent?', 'options': ['Correlation', 'Change in target per unit change in feature', 'P-value', 'Standard deviation'], 'correct': 1, 'explanation': 'Each coefficient shows how much the target changes when that feature increases by one unit.'},
    {'category': 'Linear Regression', 'question': 'What is multicollinearity?', 'options': ['Multiple targets', 'High correlation between features', 'Multiple models', 'Non-linear patterns'], 'correct': 1, 'explanation': 'Multicollinearity occurs when features are highly correlated, making coefficients unstable.'},
    {'category': 'Linear Regression', 'question': 'What is the purpose of the intercept term?', 'options': ['Scaling', 'Baseline prediction when all features are zero', 'Regularization', 'Feature selection'], 'correct': 1, 'explanation': 'The intercept represents the predicted value when all features equal zero.'},
    
    # Logistic Regression
    {'category': 'Logistic Regression', 'question': 'What type of problem does logistic regression solve?', 'options': ['Regression', 'Classification', 'Clustering', 'Dimensionality reduction'], 'correct': 1, 'explanation': 'Despite its name, logistic regression is used for binary and multi-class classification.'},
    {'category': 'Logistic Regression', 'question': 'What function does logistic regression use?', 'options': ['Linear', 'Sigmoid/Logistic function', 'ReLU', 'Tanh'], 'correct': 1, 'explanation': 'The sigmoid function σ(z) = 1/(1+e^(-z)) maps predictions to [0,1] probability range.'},
    {'category': 'Logistic Regression', 'question': 'What loss function does logistic regression use?', 'options': ['MSE', 'Binary cross-entropy/Log loss', 'Hinge loss', 'Huber loss'], 'correct': 1, 'explanation': 'Log loss penalizes confident wrong predictions more than MSE, ideal for classification.'},
    
    # Decision Trees
    {'category': 'Decision Trees', 'question': 'What is a decision tree?', 'options': ['Linear model', 'Tree structure with if-then rules', 'Neural network', 'Clustering method'], 'correct': 1, 'explanation': 'Decision trees recursively split data based on feature values to create a tree of decisions.'},
    {'category': 'Decision Trees', 'question': 'What is entropy in decision trees?', 'options': ['Loss', 'Measure of impurity/randomness', 'Accuracy', 'Depth'], 'correct': 1, 'explanation': 'Entropy measures disorder: 0 (pure) to 1 (maximum disorder). Formula: -Σ p*log2(p)'},
    {'category': 'Decision Trees', 'question': 'What is Gini impurity?', 'options': ['Entropy', 'Probability of incorrect classification', 'Accuracy', 'Depth metric'], 'correct': 1, 'explanation': 'Gini measures impurity: 1 - Σ(pi²). Lower is purer. CART algorithm uses Gini.'},
    
    # Random Forest
    {'category': 'Random Forest', 'question': 'What is Random Forest?', 'options': ['Single tree', 'Ensemble of decision trees', 'Neural network', 'Linear model'], 'correct': 1, 'explanation': 'Random Forest builds multiple decision trees and aggregates their predictions.'},
    {'category': 'Random Forest', 'question': 'What is bagging?', 'options': ['Feature selection', 'Bootstrap Aggregating - sampling with replacement', 'Boosting', 'Pruning'], 'correct': 1, 'explanation': 'Bagging creates diverse trees by training each on a random sample (with replacement) of data.'},
    
    # SVM
    {'category': 'SVM', 'question': 'What is the goal of SVM?', 'options': ['Minimize error', 'Find hyperplane maximizing margin', 'Build trees', 'Cluster data'], 'correct': 1, 'explanation': 'SVM finds the optimal hyperplane that maximizes the margin between classes.'},
    {'category': 'SVM', 'question': 'What are support vectors?', 'options': ['All points', 'Data points closest to decision boundary', 'Outliers', 'Centers'], 'correct': 1, 'explanation': 'Support vectors are the critical points on the margin boundary that define the hyperplane.'},
    
    # K-Means
    {'category': 'K-Means', 'question': 'What type of algorithm is K-Means?', 'options': ['Supervised', 'Unsupervised clustering', 'Semi-supervised', 'Reinforcement'], 'correct': 1, 'explanation': 'K-Means is unsupervised, grouping data without labels based on similarity.'},
    {'category': 'K-Means', 'question': 'What is K in K-Means?', 'options': ['Features', 'Number of clusters', 'Samples', 'Iterations'], 'correct': 1, 'explanation': 'K is a hyperparameter specifying how many clusters to create.'},
    
    # Neural Networks
    {'category': 'Neural Networks', 'question': 'What is a perceptron?', 'options': ['Deep network', 'Single-layer linear classifier', 'Clustering', 'Tree'], 'correct': 1, 'explanation': 'Perceptron is the simplest neural network: single layer performing linear classification.'},
    {'category': 'Neural Networks', 'question': 'What is an activation function?', 'options': ['Loss function', 'Non-linear transformation in neurons', 'Optimizer', 'Metric'], 'correct': 1, 'explanation': 'Activation functions introduce non-linearity, enabling networks to learn complex patterns.'},
    {'category': 'Neural Networks', 'question': 'What is the ReLU activation?', 'options': ['Sigmoid', 'max(0, x)', 'Tanh', 'Linear'], 'correct': 1, 'explanation': 'ReLU: f(x)=max(0,x), simple and effective, addresses vanishing gradient but has dying ReLU problem.'},
    
    # NumPy
    {'category': 'NumPy', 'question': 'What is NumPy?', 'options': ['Plotting library', 'Numerical computing library for arrays', 'ML framework', 'Database'], 'correct': 1, 'explanation': 'NumPy provides efficient multi-dimensional array operations in Python.'},
    {'category': 'NumPy', 'question': 'What is an ndarray?', 'options': ['List', 'N-dimensional array object', 'DataFrame', 'Dictionary'], 'correct': 1, 'explanation': 'ndarray is NumPy\'s core data structure: fast, fixed-type, multi-dimensional array.'},
    
    # Pandas
    {'category': 'Pandas', 'question': 'What is Pandas?', 'options': ['NumPy extension', 'Data manipulation library with DataFrames', 'Plotting tool', 'Database'], 'correct': 1, 'explanation': 'Pandas provides DataFrame and Series for structured data manipulation and analysis.'},
    {'category': 'Pandas', 'question': 'What is a DataFrame?', 'options': ['Array', '2D labeled data structure (table)', 'List', 'Dictionary'], 'correct': 1, 'explanation': 'DataFrame is a 2D table with labeled rows and columns, like Excel or SQL table.'},
    
    # Scikit-learn
    {'category': 'Scikit-learn', 'question': 'What is scikit-learn?', 'options': ['Deep learning', 'ML library for classical algorithms', 'Database', 'Plotting'], 'correct': 1, 'explanation': 'Scikit-learn provides implementations of regression, classification, clustering, and preprocessing.'},
    {'category': 'Scikit-learn', 'question': 'What is train_test_split()?', 'options': ['Trains model', 'Splits data into train/test sets', 'Tests model', 'Validates model'], 'correct': 1, 'explanation': 'train_test_split() randomly splits data: X_train, X_test, y_train, y_test.'},
    
    # PCA
    {'category': 'PCA', 'question': 'What is PCA?', 'options': ['Supervised', 'Unsupervised dimensionality reduction', 'Classification', 'Clustering'], 'correct': 1, 'explanation': 'PCA reduces dimensions by projecting data onto directions of maximum variance.'},
    
    # Gradient Boosting
    {'category': 'Gradient Boosting', 'question': 'What is boosting?', 'options': ['Parallel training', 'Sequential training correcting previous errors', 'Bagging', 'Random sampling'], 'correct': 1, 'explanation': 'Boosting builds models sequentially, each focusing on errors from previous models.'},
    {'category': 'Gradient Boosting', 'question': 'What is XGBoost?', 'options': ['Random Forest variant', 'Extreme Gradient Boosting with regularization', 'Neural network', 'Clustering'], 'correct': 1, 'explanation': 'XGBoost is optimized GB with L1/L2 regularization, sparsity handling, and speed optimizations.'},
    
    # Naive Bayes
    {'category': 'Naive Bayes', 'question': 'What is Naive Bayes based on?', 'options': ['Decision trees', 'Bayes theorem with independence assumption', 'Neural networks', 'SVM'], 'correct': 1, 'explanation': 'Naive Bayes applies Bayes theorem: P(y|X) = P(X|y)P(y)/P(X), assuming features are independent.'},
    
    # Add more questions here following the same pattern...
]

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
