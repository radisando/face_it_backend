## <b>PROJECT NAME</b>
Emotion Recognition Web App (face_it):  
🔗 <a href="https://www.youtube.com/watch?v=ST0ZGqhGCCw" target="_blank">Live Demo</a>    🔗 <a href="https://face-it.streamlit.app/" target="_blank">Try the App!</a> <br>



## <b>DESCRIPTION</b>
As the final project of my Data Science & AI bootcamp at Le Wagon, we built an AI-powered web application that recognizes human emotions from facial expressions in real time. Working in a 5-person team, we focused on two main aspects:

• <b>Model performance:</b> training deep learning models, reducing bias, and improving accuracy.

• <b>Scalable deployment:</b> building a Streamlit frontend and FastAPI backend, fully containerized with Docker.

The pipeline takes an image uploaded by the user, applies preprocessing (grayscale, resize, normalize), and predicts the emotion using a Convolutional Neural Network (CNN) and transfer learning with ResNet50. Our model reached around <b>70% accuracy</b> across seven emotion categories.



## <b>GOAL</b>
• Design an AI system that is not only accurate, but also fair and explainable.

• We placed special emphasis on reducing misclassifications for subtle emotions (like Fear vs. Disgust), while ensuring the model could run in real-world, production-ready scenarios.


## <b>DATASET</b>
• FER-2013 dataset (~40k labeled face images, PNG/JPG)

• Applied augmentation to rebalance underrepresented classes

• Seven emotion categories:

😊 Happiness  |  😢 Sadness  |  😱 Fear  |  😡 Anger  |  🤢 Disgust  |  😲 Surprise  |  😐 Neutral <br>



## <b>KEY FEATURES</b>
📸 Upload face images for recognition

🧹 Automated preprocessing pipeline (grayscale, resize, normalization)

🤖 Deep learning classification into 7 emotions (CNN + ResNet50) 

🪞 Explainability with Grad-CAM & SHAP

🌐 Fully deployed as a real-time web application (Streamlit + FastAPI + Docker)
