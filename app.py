import streamlit as st
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from models import CustomSVM, CustomLogisticRegression, knn_predict
from eda import display_eda, load_image_paths

@st.cache_resource
def load_models():
    knn = KNeighborsClassifier(n_neighbors=3)
    svm = SVC()
    log_reg = LogisticRegression()
    custom_svm = CustomSVM()
    custom_log_reg = CustomLogisticRegression()
    return knn, svm, log_reg, custom_svm, custom_log_reg

knn, svm, log_reg, custom_svm, custom_log_reg = load_models()

def main():
    st.title("Pneumonia Detection App")

    page = st.sidebar.selectbox("Choose a page", ["Image Input", "EDA"])

    if page == "Image Input":
        image_input_page()
    elif page == "EDA":
        eda_page()

def preprocess_images(image_paths):
    images = []
    for path in image_paths:
        image = Image.open(path).convert('L') 
        image_resized = image.resize((64, 64))
        images.append(np.array(image_resized).flatten())
    return np.array(images)

def load_and_preprocess_data():
    normal_images, pneumonia_images = load_image_paths()
    image_paths = normal_images + pneumonia_images
    labels = [0] * len(normal_images) + [1] * len(pneumonia_images)
    X = preprocess_images(image_paths)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    selector = SelectKBest(f_classif, k=500)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train_selected)
    X_test_normalized = scaler.transform(X_test_selected)

    return X_train_normalized, X_test_normalized, y_train, y_test, selector

X_train_normalized, X_test_normalized, y_train, y_test, selector = load_and_preprocess_data()

def preprocess_uploaded_image(image, selector):
    image_resized = image.resize((64, 64)).convert('L')
    image_flattened = np.array(image_resized).flatten().reshape(1, -1)
    return selector.transform(image_flattened)

def image_input_page():
    st.header("Image Input")

    uploaded_file = st.file_uploader("Choose an X-ray image...", type="jpeg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded X-ray Image', use_column_width=True)
        processed_image = preprocess_uploaded_image(image, selector)

        if st.button("Predict with sklearn KNN"):
            knn.fit(X_train_normalized, y_train)
            predict_with_model(knn, processed_image)
        if st.button("Predict with scratch KNN"):
            predict_with_scratch_knn(processed_image)
        if st.button("Predict with sklearn SVM"):
            svm.fit(X_train_normalized, y_train)
            predict_with_model(svm, processed_image)
        if st.button("Predict with custom SVM"):
            custom_svm.fit(X_train_normalized, y_train)
            predict_with_model(custom_svm, processed_image)
        if st.button("Predict with sklearn Logistic Regression"):
            log_reg.fit(X_train_normalized, y_train)
            predict_with_model(log_reg, processed_image)
        if st.button("Predict with custom Logistic Regression"):     
            custom_log_reg.fit(X_train_normalized, y_train)
            predict_with_model(custom_log_reg, processed_image)

def predict_with_model(model, image):
    prediction = model.predict(image)
    st.write(f"Prediction: {'PNEUMONIA' if prediction[0] == 1 else 'NORMAL'}")

def predict_with_scratch_knn(image):
    prediction = knn_predict(X_train_normalized, y_train, image, k=3)
    st.write(f"Prediction with scratch KNN: {'PNEUMONIA' if prediction[0] == 1 else 'NORMAL'}")

def eda_page():
    st.header("Exploratory Data Analysis")
    display_eda()

if __name__ == "__main__":
    main()
