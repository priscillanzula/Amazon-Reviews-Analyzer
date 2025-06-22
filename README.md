
### 📚 Overview
This project is an end-to-end Natural Language Processing (NLP) solution that classifies Amazon product reviews into sentiment categories. Built as part of a PLP Academy assignment, it showcases how to clean and process text data, build a machine learning classifier, apply named entity recognition (NER), and reflect on ethical aspects of NLP systems.

###  What I Accomplished

Text Classification with Scikit-learn

Cleaned and vectorized Amazon product reviews using CountVectorizer

Trained a Multinomial Naive Bayes (MultinomialNB) classifier

Achieved ~93.6% accuracy on test data

Evaluated model with:

📈 Accuracy score

🧾 Classification report

📊 Confusion matrix

🧠 Named Entity Recognition with 

Used spaCy's pre-trained pipeline to detect named entities in review text

Identified useful entity types like:

ORG → Amazon

PRODUCT → Echo



### Tools & Technologies

1. Python - Core programming language
2. Jupyter -	Interactive development environment
3. Scikit-learn -	Machine learning model building
4. spaCy	- Advanced NLP (NER, lemmatization)
5. Matplotlib -	Data visualization


### Sample Results

🔍 NER Output

Text: “The Amazon Echo is amazing.”

Entities:

Amazon → ORG

Echo → PRODUCT

🧾 Classification Report

Achieved strong precision and recall across all sentiment classes:

Accuracy: ~93.6%

###  Ethical Reflection

NLP isn't just technical — it's also about responsibility.

Key ethical considerations explored in this project:

- Bias in Language Data
  
Training on biased reviews can reinforce stereotypes or misinformation.

- User Privacy
  
Respecting user data and ensuring compliance with privacy laws like GDPR.

- Transparency and Trust
  
Building models that offer understandable, explainable results.

###  Best Practices Followed

Used a representative dataset

Ensured results were interpretable

Reflected on responsible AI use in real-world settings

 
 ###  How to Run This Project
1. Clone the Repository

   git clone https://github.com/priscillanzula/Spacy-pytorch-tensorflow.git

3. Set Up Environment

   pip install -r requirements.txt

4. Launch the Notebook

   jupyter notebook amazon.ipynb

### 🙌 Acknowledgments
Huge thanks to PLP Academy for the mentorship and structured learning.

Gratitude to the open-source community for Scikit-learn, spaCy, and Jupyter.

### 🔗 Connect With Me

Let’s talk about ML, NLP, or your next AI project:

📬 LinkedIn https://www.linkedin.com/in/priscilla-nzula/
