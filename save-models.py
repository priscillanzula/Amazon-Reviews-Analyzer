# Add this code to the end of your Jupyter notebook to save models
import joblib
import pickle

# Example: Save your trained sentiment analysis model
# Replace 'your_model' and 'your_vectorizer' with your actual variable names

def save_sentiment_models(model, vectorizer, model_path='classifier.pkl', vectorizer_path='vectorizer.pkl'):
    """
    Save the trained sentiment analysis model and vectorizer
    
    Args:
        model: Your trained classifier (e.g., MultinomialNB)
        vectorizer: Your fitted vectorizer (e.g., CountVectorizer)
        model_path: Path to save the model
        vectorizer_path: Path to save the vectorizer
    """
    try:
        # Save the model
        joblib.dump(model, model_path)
        print(f"✅ Model saved to {model_path}")
        
        # Save the vectorizer
        joblib.dump(vectorizer, vectorizer_path)
        print(f"✅ Vectorizer saved to {vectorizer_path}")
        
        # Verify the saved files
        print("\n📁 Saved files:")
        import os
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / 1024 / 1024  # Size in MB
            print(f"   {model_path} ({size:.2f} MB)")
        if os.path.exists(vectorizer_path):
            size = os.path.getsize(vectorizer_path) / 1024 / 1024  # Size in MB
            print(f"   {vectorizer_path} ({size:.2f} MB)")
            
    except Exception as e:
        print(f"❌ Error saving models: {e}")

# Example usage (add this to your notebook):
"""
# After training your model, use this to save:
save_sentiment_models(your_trained_model, your_fitted_vectorizer)

# Or manually:
joblib.dump(model, 'classifier.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
"""

# Test loading function
def test_load_models(model_path='classifier.pkl', vectorizer_path='vectorizer.pkl'):
    """Test if the saved models can be loaded correctly"""
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        print("✅ Models loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Vectorizer type: {type(vectorizer).__name__}")
        
        # Test prediction with sample text
        sample_text = ["This product is amazing!"]
        vectorized = vectorizer.transform(sample_text)
        prediction = model.predict(vectorized)
        probability = model.predict_proba(vectorized)
        
        print(f"   Test prediction: {prediction[0]}")
        print(f"   Test probability: {probability[0]}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Model saving utility")
    print("Add this code to your notebook after training your model.")
    
    # Uncomment and modify these lines in your notebook:
    save_sentiment_models(model, vectorizer)
    test_load_models()