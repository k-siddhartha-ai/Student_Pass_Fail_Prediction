#1) predict whether the student will pass or fail using study hours
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# Streamlit Page Config
st.set_page_config(
    page_title="Student Pass / Fail Prediction",
    layout="centered"
)

st.title("Student Pass / Fail Prediction")
st.write(
    "This app predicts whether a student will **PASS or FAIL** "
    "based on **study hours per day** using a Decision Tree model."
)

# Dataset

hours = np.array([[1], [2], [3], [4], [5], [6]])
results = np.array([0, 0, 1, 1, 1, 1])  # 0 = Fail, 1 = Pass

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    hours,
    results,
    test_size=0.3,
    random_state=42,
    stratify=results
)

# Train Model
model = DecisionTreeClassifier(
    max_depth=2,
    criterion="gini",
    random_state=42
)
model.fit(X_train, y_train)

# User Input
study_hours = st.slider(
    "Select study hours per day",
    min_value=0,
    max_value=10,
    value=3
)

prediction = model.predict([[study_hours]])[0]
probability = model.predict_proba([[study_hours]])[0]
confidence = np.max(probability) * 100

# Prediction Output
st.subheader("Prediction Result")

if prediction == 1:
    st.success("The student is likely to PASS")
else:
    st.error("The student is likely to FAIL")

st.info(f" Prediction Confidence: **{confidence:.2f}%**")

# Decision Tree Visualization (Clean – No clutter)
st.subheader("Decision Tree Visualization")

fig, ax = plt.subplots(figsize=(12, 6))

plot_tree(
    model,
    feature_names=["Study Hours"],
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True,
    impurity=False,
    proportion=False,
    ax=ax
)

ax.set_axis_off()
plt.tight_layout()
st.pyplot(fig)

