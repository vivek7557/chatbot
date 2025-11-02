import streamlit as st
import random

# Page configuration
st.set_page_config(
    page_title="ML Mastery Challenge",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 0.5rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .quiz-card {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    .correct {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .incorrect {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .explanation {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'game_state' not in st.session_state:
    st.session_state.game_state = 'menu'
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'streak' not in st.session_state:
    st.session_state.streak = 0
if 'total_answered' not in st.session_state:
    st.session_state.total_answered = 0
if 'answered_questions' not in st.session_state:
    st.session_state.answered_questions = []
if 'selected_answer' not in st.session_state:
    st.session_state.selected_answer = None
if 'game_questions' not in st.session_state:
    st.session_state.game_questions = []

# Static questions bank
static_questions = [
    {
        "question": "You're training a model and notice high training accuracy (98%) but low validation accuracy (65%). What's the most likely issue?",
        "options": [
            "Underfitting - model is too simple",
            "Overfitting - model memorized training data",
            "Data leakage between sets",
            "Learning rate is too high"
        ],
        "correct": 1,
        "explanation": "This classic scenario shows overfitting. The model performs exceptionally well on training data but fails to generalize to new data. Solutions include: regularization, dropout, more training data, or reducing model complexity.",
        "difficulty": "Medium"
    },
    {
        "question": "Which regularization technique randomly drops neurons during training to prevent co-adaptation?",
        "options": [
            "L1 Regularization (Lasso)",
            "L2 Regularization (Ridge)",
            "Dropout",
            "Early Stopping"
        ],
        "correct": 2,
        "explanation": "Dropout randomly sets a fraction of neurons to zero during training, forcing the network to learn robust features that work with different random subsets of neurons.",
        "difficulty": "Easy"
    },
    {
        "question": "What does the vanishing gradient problem affect most?",
        "options": [
            "Decision Trees",
            "Deep neural networks with sigmoid/tanh activations",
            "Support Vector Machines",
            "K-Nearest Neighbors"
        ],
        "correct": 1,
        "explanation": "In deep networks, gradients multiply through layers during backpropagation. Sigmoid/tanh have small derivatives (<1), causing gradients to exponentially shrink in early layers.",
        "difficulty": "Medium"
    },
    {
        "question": "In ensemble learning, what's the difference between bagging and boosting?",
        "options": [
            "Bagging trains sequentially, boosting trains in parallel",
            "Bagging trains in parallel, boosting trains sequentially focusing on errors",
            "They're the same thing with different names",
            "Bagging only works with neural networks"
        ],
        "correct": 1,
        "explanation": "Bagging trains multiple models independently in parallel on random subsets (e.g., Random Forest). Boosting trains models sequentially, each focusing on correcting previous models' errors.",
        "difficulty": "Hard"
    },
    {
        "question": "What's the purpose of a confusion matrix?",
        "options": [
            "To confuse the model during training",
            "To visualize true positives, false positives, true negatives, and false negatives",
            "To calculate training loss",
            "To normalize input features"
        ],
        "correct": 1,
        "explanation": "A confusion matrix shows classification results: true positives, false positives, true negatives, and false negatives. Essential for understanding model errors.",
        "difficulty": "Easy"
    },
    {
        "question": "What is the bias-variance tradeoff?",
        "options": [
            "More data always reduces both bias and variance",
            "Simpler models have low bias and high variance",
            "Complex models have low bias but high variance",
            "Bias and variance are unrelated concepts"
        ],
        "correct": 2,
        "explanation": "Complex models fit training data well (low bias) but are sensitive to training set variations (high variance). Simple models miss patterns (high bias) but are more stable (low variance).",
        "difficulty": "Medium"
    },
    {
        "question": "What's the main purpose of an attention mechanism in transformers?",
        "options": [
            "To reduce model size",
            "To allow the model to focus on relevant parts of the input",
            "To replace activation functions",
            "To eliminate the need for training data"
        ],
        "correct": 1,
        "explanation": "Attention mechanisms let models weigh the importance of different input parts. In transformers, self-attention allows each position to attend to all positions, capturing long-range dependencies.",
        "difficulty": "Medium"
    },
    {
        "question": "What does the Adam optimizer combine?",
        "options": [
            "SGD and dropout",
            "Momentum and RMSprop",
            "L1 and L2 regularization",
            "Gradient descent and genetic algorithms"
        ],
        "correct": 1,
        "explanation": "Adam combines momentum (exponentially weighted average of gradients) and RMSprop (adaptive learning rates per parameter). This makes it efficient for handling sparse gradients.",
        "difficulty": "Medium"
    },
    {
        "question": "What's the purpose of data augmentation in image classification?",
        "options": [
            "To increase model complexity",
            "To artificially expand training data and improve generalization",
            "To reduce training time",
            "To normalize pixel values"
        ],
        "correct": 1,
        "explanation": "Data augmentation creates variations of training images (flips, rotations, crops) to artificially expand the dataset. This helps the model learn invariant features and generalize better.",
        "difficulty": "Easy"
    },
    {
        "question": "What's the difference between precision and recall?",
        "options": [
            "They measure the same thing",
            "Precision: correct positives / predicted positives; Recall: correct positives / actual positives",
            "Precision is for regression, recall is for classification",
            "Recall is always higher than precision"
        ],
        "correct": 1,
        "explanation": "Precision asks 'Of all predicted positives, how many were correct?' Recall asks 'Of all actual positives, how many did we catch?' High precision = few false alarms; high recall = few misses.",
        "difficulty": "Medium"
    }
]

# Dynamic question templates
def generate_overfitting_question():
    train_acc = random.randint(75, 98)
    val_acc = random.randint(45, 70)
    gap = train_acc - val_acc
    
    return {
        "question": f"Your model achieves {train_acc}% training accuracy but only {val_acc}% validation accuracy. What's the primary issue?",
        "options": [
            "The model is underfitting",
            "The model is overfitting",
            "The learning rate is too low",
            "The batch size is incorrect"
        ],
        "correct": 1 if gap > 15 else 0,
        "explanation": f"With training accuracy at {train_acc}% and validation at {val_acc}%, this is {'a clear case of overfitting' if gap > 15 else 'possibly underfitting'}. {'The model memorized training patterns but fails to generalize.' if gap > 15 else 'Consider increasing model complexity.'}",
        "difficulty": "Medium"
    }

def generate_kfold_question():
    k = random.choice([3, 5, 10])
    samples = random.choice([600, 800, 1000, 1200, 1500])
    
    return {
        "question": f"You're performing {k}-fold cross-validation on a dataset with {samples} samples. How many samples are in each validation fold?",
        "options": [
            f"{samples // k} samples",
            f"{samples // (k + 1)} samples",
            f"{samples // (k - 1)} samples",
            f"{samples // 2} samples"
        ],
        "correct": 0,
        "explanation": f"In {k}-fold CV, data is split into {k} equal parts. With {samples} samples: {samples}/{k} = {samples // k} samples per validation fold.",
        "difficulty": "Easy"
    }

def generate_optimizer_question():
    optimizer = random.choice(["SGD", "Adam", "RMSprop", "AdaGrad"])
    issue = random.choice(["oscillating wildly", "converging too slowly"])
    lr = random.choice([0.1, 0.5, 1.0]) if issue == "oscillating wildly" else random.choice([0.0001, 0.00001, 0.000001])
    
    return {
        "question": f"You're using the {optimizer} optimizer with learning rate {lr}. Training is {issue}. What should you do?",
        "options": ["Decrease the learning rate", "Increase the learning rate", "Add more layers", "Remove regularization"] if issue == "oscillating wildly" else ["Increase the learning rate", "Decrease the learning rate", "Switch optimizer", "Add batch norm"],
        "correct": 0,
        "explanation": f"{'Oscillation indicates the learning rate is too high' if issue == 'oscillating wildly' else 'Slow convergence suggests learning rate is too low'}. {'Reduce it for stable steps.' if issue == 'oscillating wildly' else 'Increase it for faster progress.'}",
        "difficulty": "Easy"
    }

def generate_imbalance_question():
    positive = random.choice([2, 5, 8, 10, 15, 25])
    negative = 100 - positive
    
    return {
        "question": f"Your dataset has {negative}% negative samples and {positive}% positive samples. Which metric is most appropriate?",
        "options": ["Accuracy", "F1-Score or AUC-ROC", "Mean Squared Error", "Mean Absolute Error"],
        "correct": 1 if positive < 20 else 0,
        "explanation": f"With {'severe' if positive < 20 else 'relatively balanced'} class imbalance ({positive}% positive), {'accuracy is misleading. F1-Score and AUC-ROC handle imbalance better.' if positive < 20 else 'accuracy is reasonable, but F1-Score provides additional insights.'}",
        "difficulty": "Medium"
    }

def generate_batch_size_question():
    batch_size = random.choice([8, 16, 32, 64, 128, 256, 512])
    
    return {
        "question": f"You're training with batch size {batch_size}. What's a key characteristic of this choice?",
        "options": ["More stable gradients but may converge to sharper minima", "Noisier gradients but may find flatter minima", "Always better performance", "No effect on training"] if batch_size >= 128 else ["Noisier gradients but may find flatter minima", "More stable gradients and faster convergence", "Always worse performance", "Uses less memory always"],
        "correct": 0,
        "explanation": f"{'Large' if batch_size >= 128 else 'Small'} batch sizes ({batch_size}) provide {'more stable gradients but may converge to sharp minima (worse generalization)' if batch_size >= 128 else 'noisier gradients, potentially finding flatter minima (better generalization)'}.",
        "difficulty": "Hard"
    }

def select_questions():
    """Select mix of static and dynamic questions"""
    num_static = min(6, len(static_questions))
    num_dynamic = 6
    
    # Get random static questions
    shuffled_static = random.sample(static_questions, num_static)
    
    # Generate dynamic questions
    generators = [
        generate_overfitting_question,
        generate_kfold_question,
        generate_optimizer_question,
        generate_imbalance_question,
        generate_batch_size_question
    ]
    
    dynamic_questions = []
    for _ in range(num_dynamic):
        generator = random.choice(generators)
        dynamic_questions.append(generator())
    
    # Combine and shuffle
    all_questions = shuffled_static + dynamic_questions
    random.shuffle(all_questions)
    
    return all_questions

def start_game():
    """Start a new game"""
    st.session_state.game_questions = select_questions()
    st.session_state.game_state = 'playing'
    st.session_state.current_question = 0
    st.session_state.score = 0
    st.session_state.streak = 0
    st.session_state.answered_questions = []
    st.session_state.selected_answer = None

def submit_answer(answer_idx):
    """Submit an answer"""
    q = st.session_state.game_questions[st.session_state.current_question]
    is_correct = answer_idx == q['correct']
    
    if is_correct:
        st.session_state.score += 1
        st.session_state.streak += 1
    else:
        st.session_state.streak = 0
    
    st.session_state.answered_questions.append({
        'correct': is_correct,
        'question_num': st.session_state.current_question
    })
    st.session_state.total_answered += 1
    st.session_state.selected_answer = answer_idx

def next_question():
    """Move to next question"""
    if st.session_state.current_question < len(st.session_state.game_questions) - 1:
        st.session_state.current_question += 1
        st.session_state.selected_answer = None
    else:
        st.session_state.game_state = 'results'

# Main app logic
if st.session_state.game_state == 'menu':
    st.markdown("<div class='quiz-card'>", unsafe_allow_html=True)
    st.markdown("# 🧠 ML Mastery Challenge")
    st.markdown("### Test your intermediate machine learning knowledge!")
    st.markdown("---")
    
    st.info("**Unlimited ML questions with dynamic generation!**\n\n✓ Curated question bank\n✓ Dynamic template-based generation\n✓ Real-world scenarios\n✓ Never see the same question twice")
    
    if st.session_state.total_answered > 0:
        st.success(f"**Total Questions Answered: {st.session_state.total_answered}**")
    
    if st.button("🚀 Start New Challenge", key="start"):
        start_game()
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.game_state == 'playing':
    q = st.session_state.game_questions[st.session_state.current_question]
    
    # Progress bar
    progress = (st.session_state.current_question + 1) / len(st.session_state.game_questions)
    st.progress(progress)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Question {st.session_state.current_question + 1}/{len(st.session_state.game_questions)}**")
    with col2:
        if st.session_state.streak > 0:
            st.markdown(f"⚡ **{st.session_state.streak} Streak**")
    with col3:
        st.markdown(f"🏆 **Score: {st.session_state.score}**")
    
    st.markdown("<div class='quiz-card'>", unsafe_allow_html=True)
    
    # Difficulty badge
    st.markdown(f"**Difficulty:** `{q['difficulty']}`")
    
    # Question
    st.markdown(f"## {q['question']}")
    
    # Options
    if st.session_state.selected_answer is None:
        for idx, option in enumerate(q['options']):
            if st.button(option, key=f"opt_{idx}", use_container_width=True):
                submit_answer(idx)
                st.rerun()
    else:
        # Show results
        for idx, option in enumerate(q['options']):
            is_correct = idx == q['correct']
            is_selected = idx == st.session_state.selected_answer
            
            if is_correct:
                st.markdown(f"<div class='correct'>✅ {option}</div>", unsafe_allow_html=True)
            elif is_selected:
                st.markdown(f"<div class='incorrect'>❌ {option}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div>{option}</div>", unsafe_allow_html=True)
        
        # Explanation
        st.markdown(f"<div class='explanation'><strong>💡 Explanation:</strong><br>{q['explanation']}</div>", unsafe_allow_html=True)
        
        # Next button
        if st.button("➡️ Next Question" if st.session_state.current_question < len(st.session_state.game_questions) - 1 else "📊 See Results", key="next"):
            next_question()
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.game_state == 'results':
    percentage = (st.session_state.score / len(st.session_state.game_questions)) * 100
    
    if percentage >= 90:
        message = "🌟 ML Expert! Outstanding performance!"
    elif percentage >= 75:
        message = "🎯 Great job! You know your ML!"
    elif percentage >= 60:
        message = "👍 Good work! Keep practicing!"
    else:
        message = "📚 Keep learning! Review the concepts and try again!"
    
    st.markdown("<div class='quiz-card'>", unsafe_allow_html=True)
    
    st.markdown("# 🏆 Challenge Complete!")
    st.markdown(f"## {message}")
    st.markdown("---")
    
    # Score display
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Score", f"{st.session_state.score}/{len(st.session_state.game_questions)}")
    with col2:
        st.metric("Percentage", f"{percentage:.0f}%")
    
    st.success(f"**Lifetime Total: {st.session_state.total_answered} questions answered**")
    
    # Results grid
    st.markdown("### Question Results:")
    cols = st.columns(4)
    for idx, result in enumerate(st.session_state.answered_questions):
        with cols[idx % 4]:
            if result['correct']:
                st.markdown(f"<div style='background:#d4edda; padding:1rem; border-radius:0.5rem; text-align:center; font-weight:bold; color:#28a745;'>{idx + 1}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background:#f8d7da; padding:1rem; border-radius:0.5rem; text-align:center; font-weight:bold; color:#dc3545;'>{idx + 1}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("♾️ Play Again (New Questions)", key="play_again"):
        start_game()
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
