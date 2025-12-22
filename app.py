import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Pass/Fail Prediction", layout="centered")

st.title("Student Pass / Fail Prediction")
st.write(
    "This app predicts whether a student will PASS or FAIL "
    "based on study hours using a Decision Tree model."
)

hours = np.array([[1], [2], [3], [4], [5], [6]])
result = np.array([0, 0, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(
    hours, result, test_size=0.3, random_state=42, stratify=result
)

model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(X_train, y_train)

study_hours = st.slider("Select study hours", 0, 10, 3)
prediction = model.predict([[study_hours]])[0]

st.subheader("Prediction Result")
if prediction == 1:
    st.success("The student is likely to PASS")
else:
    st.error("The student is likely to FAIL")

fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(
    model,
    feature_names=["Study Hours"],
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True,
    ax=ax
)

st.pyplot(fig)
