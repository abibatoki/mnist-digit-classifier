# 🔢 MNIST Digit Classifier

> 🚀 **Streamlit + TensorFlow | Deployed on Hugging Face Spaces with Docker**

This project is a personal milestone for me. I trained a convolutional neural network (CNN) to recognize handwritten digits from the popular MNIST dataset and then deployed it using **Streamlit**, **Docker**, and **Hugging Face Spaces**. The goal was to create a fully interactive app that lets anyone draw a digit and get an instant prediction.

---

## 🚀 Live Demo

Try it out here:  
🔗 **[https://huggingface.co/spaces/abibatoki/mnist_digit_classifier](https://huggingface.co/spaces/abibatoki/mnist_digit_classifier)**

---

## 📦 Code and Resources Used

**Python Version**: 3.8.

**Libraries**:
- `tensorflow`, `numpy`
- `pillow`
- `streamlit`, `streamlit-drawable-canvas`

**Deployment Stack**:
- Streamlit frontend  
- Docker container  
- Hugging Face Spaces  

---

## 🧠 Problem Statement

The MNIST dataset is often referred to as the “Hello World” of deep learning. While it’s not used in real-world applications, it provides a foundational exercise in building and deploying a neural network for image classification. The aim was to move beyond model training and get hands-on with **serving machine learning models** through a user-friendly web interface.

---

## 🏗️ Model Building

1. **Data Preprocessing**
   - Loaded and reshaped image data  
   - Normalized pixel values  
   - One-hot encoded the target labels (0–9)  

2. **Model Architecture**
   - CNN with two convolutional layers  
   - MaxPooling and Dropout layers to reduce overfitting  
   - Dense output layer with 10 units and softmax activation
   - Combination of **ReLU** and **tanh** activation functions to improve learning performance  

3. **Training**
   - Trained for 5 epochs with batch size 100
   - Achieved **98.52% validation accuracy on the train set** 
   - Achieved **97.72% accuracy on the test set**  
   - Saved using `model.save("mnist_model.h5", include_optimizer=False)` to ensure clean loading in production  

---

## 🌐 App Features

- ✏️ **Drawable Canvas**: Users can sketch a digit with their mouse or finger  
- 📷 Image is captured and preprocessed in real time  
- 🤖 Model makes a live prediction and displays the most likely digit  
- 🧼 No backend server or database needed — the app runs entirely in the browser using Streamlit  

---

## 🧠 Lessons Learned

- How to build a CNN from scratch with Keras  
- How to save and load models effectively for inference  
- Streamlit's interactive canvas and event-handling features  
- Docker basics for packaging a model-serving app  
- The deployment process on Hugging Face Spaces  

---

Thanks for reading

Feel free to reach out or [connect on LinkedIn]([https://www.linkedin.com/](https://www.linkedin.com/in/abibatoki/) if you want to discuss or collaborate.
