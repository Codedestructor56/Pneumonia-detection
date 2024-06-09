import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image
import os
import cv2

st.set_option('deprecation.showPyplotGlobalUse', False)
# Load data
def load_image_paths():
    normal_images = [os.path.join('data/NORMAL', f) for f in os.listdir('data/NORMAL') if f.endswith('.jpeg')]
    pneumonia_images = [os.path.join('data/PNEUMONIA', f) for f in os.listdir('data/PNEUMONIA') if f.endswith('.jpeg')]
    return normal_images, pneumonia_images

normal_images, pneumonia_images = load_image_paths()

def get_image_stats(image_paths):
    dimensions = []
    mean_values = []
    for image_path in image_paths:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
        dimensions.append(img_array.shape)
        mean_values.append(np.mean(img_array))
    return dimensions, mean_values

normal_dimensions, normal_means = get_image_stats(normal_images)
pneumonia_dimensions, pneumonia_means = get_image_stats(pneumonia_images)

def display_eda():
    # Class Distribution
    st.subheader("Class Distribution")
    class_distribution()

    # Image Dimensions
    st.subheader("Image Dimensions")
    image_dimensions()

    # Mean Pixel Values
    st.subheader("Mean Pixel Values")
    mean_pixel_values()

    # Aspect Ratio Analysis
    st.subheader("Aspect Ratio Analysis")
    aspect_ratio_analysis()

    # Intensity Histograms
    st.subheader("Intensity Histogram - Normal")
    intensity_histogram(normal_images, 'Normal')

    st.subheader("Intensity Histogram - Pneumonia")
    intensity_histogram(pneumonia_images, 'Pneumonia')

    # Brightness and Contrast(mean and standard deviation)
    st.subheader("Brightness and Contrast")
    brightness_contrast_analysis()

def class_distribution():
    data = {'Class': ['NORMAL'] * len(normal_images) + ['PNEUMONIA'] * len(pneumonia_images)}
    df = pd.DataFrame(data)
    sns.countplot(x='Class', data=df)
    plt.title('Distribution of Classes')
    st.pyplot()

def image_dimensions():
    normal_dims = pd.DataFrame(normal_dimensions, columns=['Height', 'Width'])
    pneumonia_dims = pd.DataFrame(pneumonia_dimensions, columns=['Height', 'Width'])
    
    st.write("Normal Images Dimensions")
    st.write(normal_dims.describe())

    st.write("Pneumonia Images Dimensions")
    st.write(pneumonia_dims.describe())

def mean_pixel_values():
    data = {'Class': ['NORMAL'] * len(normal_means) + ['PNEUMONIA'] * len(pneumonia_means),
            'Mean Pixel Value': normal_means + pneumonia_means}
    df = pd.DataFrame(data)
    sns.boxplot(x='Class', y='Mean Pixel Value', data=df)
    plt.title('Mean Pixel Values')
    st.pyplot()

def get_aspect_ratios(dimensions):
    aspect_ratios = [dim[1] / dim[0] for dim in dimensions]  
    return aspect_ratios

def aspect_ratio_analysis():
    normal_aspect_ratios = get_aspect_ratios(normal_dimensions)
    pneumonia_aspect_ratios = get_aspect_ratios(pneumonia_dimensions)

    plt.figure(figsize=(12, 6))
    sns.histplot(normal_aspect_ratios, kde=True, color='blue', label='Normal')
    sns.histplot(pneumonia_aspect_ratios, kde=True, color='red', label='Pneumonia')
    plt.title('Aspect Ratio Distribution')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Frequency')
    plt.legend()
    st.pyplot()

def intensity_histogram(image_paths, title):
    plt.figure(figsize=(10, 5))
    for image_path in image_paths[:10]:  
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
        plt.hist(img_array.ravel(), bins=256, alpha=0.5)
    plt.title(f'Intensity Histogram - {title}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    st.pyplot()

def calculate_brightness_contrast(image_paths):
    brightness = []
    contrast = []
    for image_path in image_paths:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
        brightness.append(np.mean(img_array))
        contrast.append(np.std(img_array))
    return brightness, contrast

def brightness_contrast_analysis():
    normal_brightness, normal_contrast = calculate_brightness_contrast(normal_images)
    pneumonia_brightness, pneumonia_contrast = calculate_brightness_contrast(pneumonia_images)

    plt.figure(figsize=(12, 6))
    sns.histplot(normal_brightness, kde=True, color='blue', label='Normal')
    sns.histplot(pneumonia_brightness, kde=True, color='red', label='Pneumonia')
    plt.title('Brightness Distribution')
    plt.xlabel('Brightness')
    plt.ylabel('Frequency')
    plt.legend()
    st.pyplot()

    plt.figure(figsize=(12, 6))
    sns.histplot(normal_contrast, kde=True, color='blue', label='Normal')
    sns.histplot(pneumonia_contrast, kde=True, color='red', label='Pneumonia')
    plt.title('Contrast Distribution')
    plt.xlabel('Contrast')
    plt.ylabel('Frequency')
    plt.legend()
    st.pyplot()

def edge_detection_analysis():
    plt.figure(figsize=(10, 10))
    for i, image_path in enumerate(normal_images[:5] + pneumonia_images[:5]):
        plt.subplot(4, 5, i + 1)
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
        edges = cv2.Canny(img_array, threshold1=100, threshold2=200)
        plt.imshow(edges, cmap='gray')
        plt.title('Edge Detection')
        plt.axis('off')
    st.pyplot()

