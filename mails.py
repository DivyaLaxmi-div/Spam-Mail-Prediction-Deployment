import streamlit as st
import pickle

model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

st.title("📧 Spam Mail Classifier")
st.write("Enter an email message below and the app will predict whether it is **Spam** or **Not Spam**.")

input_mail = st.text_area("Enter the email message here:")

if st.button("Predict"):
    if input_mail.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        input_data = vectorizer.transform([input_mail])
        prediction = model.predict(input_data)
        if prediction[0] == 0:
            st.success("✅ The email is **Not Spam**")
        else:
            st.error("🚨 The email is **Spam**")
