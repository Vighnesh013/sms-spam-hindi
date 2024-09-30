import streamlit as st
import pickle



# Load the model and label encoder
try:
    svm = pickle.load(open('svm.pkl', 'rb'))
    print("SVM model loaded successfully.")
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    print("TF-IDF vectorizer loaded successfully.")
    lbl = pickle.load(open('lbl.pkl', 'rb'))
    print("Label encoder loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    print(f"Error: {e}")



def predict_emails(text):
    # Ensure 'tfidf' is the vectorizer instance, not a transformed matrix
    txt_vect = tfidf.transform([text])  # Transform the input text
    prediction = svm.predict(txt_vect)   # Make prediction using the model
    return lbl.inverse_transform(prediction)[0]  # Return the predicted label


def main():
    st.title("Email Spam/Ham Classification")
    st.write("Enter an email below to classify if it's spam or ham!")

    html_temp = """
    <div style="background-color:#25246; padding:10px">
        <h2 style="color:white; text-align:center;"> Spam Email Classification </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)  # Render the HTML code as HTML

    # Getting the input from the user
    input_text = st.text_input("Enter the message")

    if st.button("Click to predict"):
        if input_text:
            output = predict_emails(input_text)

            if output == "spam":
                st.markdown("""
                <div style="background-color:#F4D03F; padding:10px">
                    <h2 style="color:white; text-align:center;"> This Email is Spam </h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color:#F4D03F; padding:10px">
                    <h2 style="color:white; text-align:center;"> This Email is Ham </h2>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter a message to classify.")

if __name__ == "__main__":
    main()
