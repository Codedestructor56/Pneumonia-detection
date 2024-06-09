import streamlit as st
from PIL import Image
import numpy as np
import os


from eda import display_eda

# Define the two pages
def main():
    st.title("Pneumonia Detection App")

    page = st.sidebar.selectbox("Choose a page", ["Image Input", "EDA"])

    if page == "Image Input":
        image_input_page()
    elif page == "EDA":
        eda_page()

def image_input_page():
    st.header("Image Input")

    uploaded_file = st.file_uploader("Choose an X-ray image...", type="jpeg")

    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded X-ray Image', use_column_width=True)
        
        # Preprocess the image and get the model prediction (dummy implementation)
        st.write("Classifying...")
        prediction = dummy_model_predict(image)
        st.write(f"Prediction: {prediction}")

def dummy_model_predict(image):
    # Dummy model prediction
    return "PNEUMONIA" if np.random.rand() > 0.5 else "NORMAL"

def eda_page():
    st.header("Exploratory Data Analysis")
    display_eda()

if __name__ == "__main__":
    main()
