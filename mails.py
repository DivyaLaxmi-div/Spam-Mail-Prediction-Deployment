import streamlit as st
import pickle

# Load the trained model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Streamlit App Title
st.title("📧 Spam Mail Classifier")
st.write("Enter an email message below and the app will predict whether it is **Spam** or **Not Spam**.")

# Text Input from User
input_mail = st.text_area("✉️ Enter the email message here:")

# Predict Button
if st.button("Predict"):
    if input_mail.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        # Transform input using vectorizer
        input_data = vectorizer.transform([input_mail])

        # Predict Spam Probability
        probability = model.predict_proba(input_data)[0][1]  # Probability of Spam

        # Threshold-based Classification (No Confidence shown)
        if probability > 0.6:
            st.error("🚨 The email is Spam")
        else:
            st.success("✅ The email is Not Spam")
