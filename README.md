### 📘 TensorFlow vs PyTorch

TensorFlow and PyTorch are two of the most widely used deep learning frameworks:

 1. TensorFlow uses a static computation graph by default (though eager execution is now supported), whereas PyTorch relies on a dynamic computation graph, making it more intuitive for development.

 2. PyTorch is easier to debug thanks to its native Pythonic control flow, while TensorFlow's static graphs can make debugging more challenging.

 3. TensorFlow excels in deployment with mature tools like TensorFlow Lite and TensorFlow Serving, while PyTorch's deployment options like TorchServe are still developing.

 4. TensorFlow has a large community, especially in production environments, whereas PyTorch is widely favored in academic and research circles.


### When to use:

TensorFlow: Best suited for production environments, mobile/embedded apps, or when using tools like TensorBoard.

PyTorch: Preferred for research, prototyping, or when ease of debugging and experimentation is important.

### Jupyter Notebooks in AI development:

- Exploratory Data Analysis (EDA):

Jupyter allows interactive data visualization, making it ideal for exploring datasets before model development.

- Model Prototyping and Experimentation:

Developers can iteratively build and tweak models, view results instantly, and document their workflow in the same environment.

SpaCy enhance NLP tasks compared to basic Python string operations:

spaCy provides advanced linguistic features such as part-of-speech tagging, named entity recognition (NER), lemmatization, and dependency parsing. These go far beyond what native Python string methods (.split(), .replace(), etc.) can do, enabling more accurate and scalable NLP pipelines.

### Comparative Analysis

- Scikit-learn is designed for classical machine learning tasks such as SVMs and decision trees, while TensorFlow is built for deep learning models like CNNs and RNNs.

- Scikit-learn is very beginner-friendly and easy to use, whereas TensorFlow has a steeper learning curve due to its complexity.

- Scikit-learn has strong support in academic settings and ML education, while TensorFlow is more prominent in production environments and advanced AI research.


1. Named Entity Recognition (NER) Example Using spaCy

Entities identified: 'Amazon' (ORG), 'Echo' (PRODUCT)

📷![Results](https://github.com/user-attachments/assets/8fe5e2e7-6797-4866-b580-64c1daa9e542)


2. Model Output/Performance Evaluation

  Accuracy Achieved: ~93.6%

  Confusion Matrix and Classification Report: Well-balanced performance across labels.

📷 ![Summary](https://github.com/user-attachments/assets/c06e62c2-d9c0-4a50-8f3b-7ee242161ff6)



### 🤖 Ethical Reflection

Building and deploying NLP models comes with ethical considerations:

- Bias in Language Data:

Training on biased datasets can reinforce stereotypes or misrepresentations.

- User Privacy:

Sentiment analysis on user reviews must ensure compliance with data privacy laws like GDPR.

- Transparency and Interpretability:

Models, especially in customer-facing apps, should offer explanations for their outputs to build user trust.

### Best Practices:

1. Use diverse datasets.

2. Implement privacy-preserving data pipelines.

3. Provide model cards explaining training data, limitations, and intended use
