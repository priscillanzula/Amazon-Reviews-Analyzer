import streamlit as st
import joblib
import pandas as pd
import spacy
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="🛍️",
    layout="wide"
)

# Load models with caching and fallback training
@st.cache_resource
def load_or_create_models():
    """Load existing models or create new ones with sample data"""
    try:
        # Try to load existing models
        if os.path.exists('classifier.pkl') and os.path.exists('vectorizer.pkl'):
            model = joblib.load('classifier.pkl')
            vectorizer = joblib.load('vectorizer.pkl')
            try:
                nlp = spacy.load('en_core_web_sm')
            except OSError:
                st.warning("⚠️ spaCy model 'en_core_web_sm' not found. Install it with: python -m spacy download en_core_web_sm")
                nlp = None
            st.success("✅ Existing models loaded successfully!")
            return model, vectorizer, nlp
        else:
            st.info("🔄 Model files not found. Creating new models with sample data...")
            return create_sample_models()
    except Exception as e:
        st.warning(f"⚠️ Error loading models: {e}")
        st.info("🔄 Creating new models...")
        return create_sample_models()

def create_sample_models():
    """Create models using sample review data"""
    # Sample training data (you should replace this with your actual dataset)
    sample_reviews = [
        # Positive reviews
        "This product is amazing! Great quality and fast delivery.",
        "Excellent purchase, highly recommend!",
        "Love it! Works perfectly and looks great.",
        "Outstanding quality, exceeded my expectations.",
        "Perfect product, exactly what I needed.",
        "Great value for money, very satisfied.",
        "Awesome product, will buy again!",
        "Fantastic quality, highly recommend to everyone.",
        "Best purchase I've made, love it!",
        "Excellent service and great product quality.",
        
        # Negative reviews  
        "Terrible product, waste of money.",
        "Poor quality, broke after one day.",
        "Disappointed with this purchase.",
        "Not worth the price, very poor quality.",
        "Awful product, doesn't work as advertised.",
        "Bad quality, returning immediately.",
        "Worst purchase ever, completely useless.",
        "Poor construction, fell apart quickly.",
        "Don't buy this, complete waste of money.",
        "Horrible product, very disappointed.",
        
        # Neutral reviews
        "It's okay, nothing special but does the job.",
        "Average product, neither good nor bad.",
        "Decent quality for the price I guess.",
        "It works but could be better.",
        "Mediocre product, meets basic expectations.",
        "Fair quality, nothing to complain about.",
        "Standard product, does what it says.",
        "Reasonable purchase, gets the job done.",
        "Adequate quality, not amazing but okay.",
        "Acceptable product, serves its purpose."
    ]
    
    # Labels (1=positive, 0=negative, 2=neutral)
    sample_labels = (
        [1] * 10 +  # positive
        [0] * 10 +  # negative  
        [2] * 10    # neutral
    )
    
    # Create and train vectorizer
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(sample_reviews)
    
    # Train classifier
    model = MultinomialNB()
    model.fit(X, sample_labels)
    
    # Save models
    joblib.dump(model, 'classifier.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    # Load spaCy model
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        st.warning("⚠️ spaCy model 'en_core_web_sm' not found. Named Entity Recognition will be disabled.")
        st.info("To enable NER, run: python -m spacy download en_core_web_sm")
        nlp = None
    
    st.success("✅ New models created and trained with sample data!")
    st.info("💡 For better accuracy, replace the sample data with your actual Amazon review dataset.")
    
    return model, vectorizer, nlp

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🛍️ Amazon Review Sentiment Analyzer</h1>
    <p>Powered by Machine Learning & Natural Language Processing</p>
</div>
""", unsafe_allow_html=True)

# Load models
try:
    model, vectorizer, nlp = load_or_create_models()
except Exception as e:
    st.error(f"❌ Critical error loading models: {e}")
    st.stop()

# Sidebar
st.sidebar.title("📊 About This App")
st.sidebar.info("""
This app uses:
- **Multinomial Naive Bayes** for sentiment classification
- **CountVectorizer** for text preprocessing  
- **spaCy** for Named Entity Recognition (if available)
- Trained on sample review data
""")

st.sidebar.title("🎯 How to Use")
st.sidebar.markdown("""
1. Enter a product review in the text area
2. Click 'Analyze Review' 
3. View sentiment prediction & confidence
4. Explore named entities found in text
5. See probability distribution
""")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Product Review")
    review_text = st.text_area(
        "",
        placeholder="Example: The Amazon Echo is amazing! Great sound quality and easy setup. Highly recommend this product.",
        height=150,
        key="review_input"
    )
    
    # Example reviews
    st.subheader("💡 Try These Examples")
    examples = [
        "The Amazon Echo is amazing! Great sound quality and easy setup.",
        "Terrible product. Broke after one week. Very disappointed.",
        "It's okay, nothing special but does the job I guess."
    ]
    
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    with col_ex1:
        if st.button("😊 Positive Example"):
            st.session_state.review_input = examples[0]
            st.rerun()
    with col_ex2:
        if st.button("😞 Negative Example"):
            st.session_state.review_input = examples[1]
            st.rerun()
    with col_ex3:
        if st.button("😐 Neutral Example"):
            st.session_state.review_input = examples[2]
            st.rerun()

with col2:
    st.subheader("📈 Model Info")
    
    # Model info (replace with actual metrics if available)
    st.info("""
    **Model Status:** ✅ Active
    
    **Features:**
    - Sentiment Classification
    - Confidence Scoring
    - Named Entity Recognition
    - Text Analysis
    """)

# Analysis button
if st.button('🔍 Analyze Review', type='primary', use_container_width=True):
    if review_text.strip():
        # Show loading spinner
        with st.spinner('Analyzing review...'):
            try:
                # Sentiment prediction
                text_vectorized = vectorizer.transform([review_text])
                prediction_num = model.predict(text_vectorized)[0]
                probabilities = model.predict_proba(text_vectorized)[0]
                confidence = max(probabilities)
                
                # Map numeric predictions to labels
                label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
                prediction = label_map.get(prediction_num, 'unknown')
                
                # Get class labels
                classes = [label_map.get(i, f'class_{i}') for i in model.classes_]
                prob_dict = dict(zip(classes, probabilities))
                
                # NER analysis (only if spaCy is available)
                entities = []
                if nlp is not None:
                    try:
                        doc = nlp(review_text)
                        entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
                    except Exception as e:
                        st.warning(f"NER analysis failed: {e}")
                
                # Display results
                st.success("Analysis Complete!")
                
                # Results columns
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.subheader("🎯 Sentiment Analysis")
                    
                    # Sentiment display with styling
                    sentiment_class = f"sentiment-{prediction}"
                    sentiment_emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
                    
                    st.markdown(f"""
                    <div class="{sentiment_class}">
                        <h3>{sentiment_emoji.get(prediction, "🤔")} {prediction.title()}</h3>
                        <p><strong>Confidence:</strong> {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability distribution
                    fig_prob = go.Figure(data=[
                        go.Bar(x=list(prob_dict.keys()), 
                               y=list(prob_dict.values()),
                               marker_color=['#28a745' if k==prediction else '#6c757d' for k in prob_dict.keys()])
                    ])
                    fig_prob.update_layout(
                        title="Confidence Distribution",
                        xaxis_title="Sentiment",
                        yaxis_title="Probability",
                        height=300
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
                
                with result_col2:
                    st.subheader("🧠 Named Entity Recognition")
                    
                    if nlp is None:
                        st.info("NER disabled - spaCy model not available")
                        st.markdown("""
                        <div style="text-align: center; padding: 2rem;">
                            <span style="font-size: 3rem;">📦</span>
                            <p>Install spaCy model to enable NER:</p>
                            <code>python -m spacy download en_core_web_sm</code>
                        </div>
                        """, unsafe_allow_html=True)
                    elif entities:
                        # Create entity dataframe
                        entity_df = pd.DataFrame(entities, columns=['Entity', 'Type', 'Start', 'End'])
                        
                        # Display entities as colored badges
                        for _, row in entity_df.iterrows():
                            entity_type_colors = {
                                'PERSON': '#e3f2fd',
                                'ORG': '#f3e5f5', 
                                'PRODUCT': '#e8f5e8',
                                'GPE': '#fff3e0',
                                'MONEY': '#fce4ec'
                            }
                            color = entity_type_colors.get(row['Type'], '#f5f5f5')
                            st.markdown(f"""
                            <div style="background-color: {color}; padding: 8px; margin: 4px 0; border-radius: 5px; border-left: 3px solid #666;">
                                <strong>{row['Entity']}</strong> → {row['Type']}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Entity types chart
                        entity_counts = entity_df['Type'].value_counts()
                        if len(entity_counts) > 0:
                            fig_entities = px.pie(
                                values=entity_counts.values, 
                                names=entity_counts.index,
                                title="Entity Types Distribution"
                            )
                            fig_entities.update_layout(height=300)
                            st.plotly_chart(fig_entities, use_container_width=True)
                    else:
                        st.info("No named entities found in this review.")
                        st.markdown("""
                        <div style="text-align: center; padding: 2rem;">
                            <span style="font-size: 4rem;">🔍</span>
                            <p>Try a review mentioning brands, products, or locations!</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Detailed analysis
                st.subheader("📊 Detailed Analysis")
                
                analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
                
                with analysis_col1:
                    st.metric("Review Length", f"{len(review_text)} characters")
                    st.metric("Word Count", f"{len(review_text.split())} words")
                
                with analysis_col2:
                    st.metric("Entities Found", len(entities))
                    # Calculate sentiment strength
                    sentiment_strength = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                    st.metric("Prediction Strength", sentiment_strength)
                
                with analysis_col3:
                    # Text statistics
                    sentences = len([s for s in review_text.split('.') if s.strip()])
                    st.metric("Sentences", sentences)
                    avg_word_length = np.mean([len(word) for word in review_text.split()]) if review_text.split() else 0
                    st.metric("Avg Word Length", f"{avg_word_length:.1f}")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.info("Please check the error details above.")
    else:
        st.warning("⚠️ Please enter a review to analyze!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Built with ❤️ using Streamlit, scikit-learn, and spaCy</p>
    <p>Deploy this app on <a href="https://streamlit.io/cloud">Streamlit Cloud</a> for free!</p>
</div>
""", unsafe_allow_html=True)