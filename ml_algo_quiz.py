import random
import sys

def generate_bulk_questions():
    # Core ML algorithms and tools lists
    algos = [
        "Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting",
        "AdaBoost", "XGBoost", "LightGBM", "K-Nearest Neighbors", "Naive Bayes", "Support Vector Machine",
        "K-Means", "Hierarchical Clustering", "DBSCAN", "Apriori Algorithm", "Principal Component Analysis",
        "Independent Component Analysis", "t-SNE", "Autoencoders", "Neural Networks", "Convolutional Neural Network",
        "Recurrent Neural Network", "Long Short Term Memory", "Transformer", "BERT", "GPT", "YOLO", "FastText",
        "CatBoost", "Ridge Regression", "Lasso Regression", "ElasticNet", "Bagging", "Stacking", "Voting Classifier",
        "Isolation Forest", "One-Class SVM", "Prophet", "ARIMA", "SARIMA", "VAR", "LightFM", "Surprise", "DeepAR"
    ]
    tools = [
        "Pandas", "NumPy", "Matplotlib", "Seaborn", "Scikit-learn", "TensorFlow", "PyTorch", "Keras", "JAX",
        "Statsmodels", "Plotly", "Altair", "Snowflake", "SQL", "BigQuery", "Airflow", "MLflow", "DVC",
        "Docker", "Kubernetes", "Spark", "Hadoop", "Kafka", "AWS Sagemaker", "Google AI Platform", "Azure ML",
        "Jupyter Notebook", "Colab", "Streamlit", "Dash", "FastAPI", "Elasticsearch", "Neo4j", "Redshift",
        "PostgreSQL", "MySQL", "Tableau", "Power BI", "Metabase", "DataDog", "Prometheus", "Looker"
    ]
    concepts = [
        "Bias-Variance Tradeoff", "Hyperparameter Tuning", "Cross-Validation", "Ensemble Methods",
        "Feature Engineering", "Dimensionality Reduction", "Overfitting", "Underfitting", "Regularization",
        "Bootstrap Sampling", "Grid Search", "Random Search", "Early Stopping", "Batch Normalization",
        "Attention Mechanism", "Activation Functions", "Confusion Matrix", "ROC Curve", "Precision-Recall",
        "AUC Score", "F1 Score", "Clustering Validation", "Business Intelligence", "ETL Pipeline",
        "Data Normalization", "One-Hot Encoding", "Word Embeddings", "Tokenization", "NER", "Sentiment Analysis"
    ]
    # MCQ answer templates; correct, plus distractors
    templates = [
        ("What is the primary purpose of {}?", [
            "{} is mainly used for supervised learning tasks.",
            "It is primarily used for unsupervised learning.",
            "It is used for reinforcement learning.",
            "It is a visualization tool."
        ]),
        ("Which library is best known for implementing {}?", [
            "Scikit-learn",
            "Matplotlib",
            "NLTK",
            "TensorBoard"
        ]),
        ("Which of the following best describes {}?", [
            "A machine learning algorithm or technique.",
            "A data visualization library.",
            "A cloud ML platform.",
            "A database solution."
        ]),
        ("{} is primarily:", [
            "A supervised algorithm.",
            "An unsupervised algorithm.",
            "A deep learning framework.",
            "A visualization library."
        ]),
        ("Which domain is {} most commonly used in?", [
            "Machine Learning",
            "Web Development",
            "Database Management",
            "Mobile Apps"
        ]),
        ("Which of these is a key feature of {}?", [
            "Improves model accuracy by combining multiple learners.",
            "Provides data storage solutions.",
            "Handles real-time data streams.",
            "Offers web deployment for ML models."
        ]),
        ("What is {} NOT used for?", [
            "Image recognition tasks.",
            "Training neural networks.",
            "Data visualization.",
            "Natural language processing."
        ])
    ]
    # Flatten topics
    topics = algos + tools + concepts

    # Generate 1000+ questions by cycling through topics and templates
    questions = []
    num_questions = 1050
    for i in range(num_questions):
        topic = random.choice(topics)
        question_template, answers = random.choice(templates)
        correct_answer = answers[0].format(topic)
        distractors = [a.format(topic) for a in answers[1:]]
        qtype = "mcq"
        all_answers = [correct_answer] + distractors
        random.shuffle(all_answers)
        questions.append({
            "question": question_template.format(topic),
            "answers": all_answers,
            "correct": correct_answer
        })
    return questions

def run_quiz():
    questions = generate_bulk_questions()
    random.shuffle(questions)
    print("Welcome to the Machine Learning Algorithms & Data Science Tools Quiz!")
    print("You will be asked 10 random questions each round. Answer by selecting the choice number.\n")
    score = 0
    num_per_round = 10
    try:
        for i in range(num_per_round):
            q = questions[i]
            print(f"Q{i+1}: {q['question']}")
            for idx, option in enumerate(q["answers"], 1):
                print(f"  {idx}. {option}")
            answer = input("Your choice (1-4, or 'q' to quit): ").strip()
            if answer.lower() == "q":
                print(f"Quitting. Your score: {score}/{i}")
                sys.exit()
            try:
                if q["answers"][int(answer)-1] == q["correct"]:
                    print("✅ Correct!\n")
                    score += 1
                else:
                    print(f"❌ Incorrect. Correct answer: {q['correct']}\n")
            except Exception:
                print("Invalid input. Skipping question.\n")
        print(f"Quiz complete. Your score: {score}/{num_per_round}")
    except KeyboardInterrupt:
        print(f"\nAborted. Your score: {score}/{i}")

if __name__ == "__main__":
    run_quiz()