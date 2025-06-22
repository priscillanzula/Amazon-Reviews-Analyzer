import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2
from streamlit_drawable_canvas import st_drawable_canvas
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
import os

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

# Load or create MNIST model with caching
@st.cache_resource
def load_or_create_mnist_model():
    """Load existing model or create a new one"""
    model_path = 'mnist_model.pkl'
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            st.success("✅ Existing MNIST model loaded successfully!")
            return model
        except Exception as e:
            st.warning(f"⚠️ Could not load existing model: {e}")
            st.info("Creating a new model...")
    
    # Create and train a new model
    st.info("🔄 Training a new MNIST model (this may take a moment)...")
    
    with st.spinner("Downloading and training MNIST model..."):
        try:
            # Load MNIST data from sklearn
            mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
            X, y = mnist.data, mnist.target.astype(int)
            
            # Use a subset for faster training (first 10000 samples)
            X_subset = X[:10000] / 255.0  # Normalize
            y_subset = y[:10000]
            
            # Train Random Forest (faster than neural networks for demo)
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_subset, y_subset)
            
            # Save model
            joblib.dump(model, model_path)
            st.success("✅ New MNIST model trained and saved!")
            
            return model
            
        except Exception as e:
            st.error(f"❌ Error creating model: {e}")
            st.info("Using a simple fallback model...")
            
            # Fallback: create a dummy model that just predicts random digits
            class DummyModel:
                def predict(self, X):
                    return np.random.randint(0, 10, size=len(X))
                def predict_proba(self, X):
                    # Return random probabilities that sum to 1
                    probs = np.random.random((len(X), 10))
                    return probs / probs.sum(axis=1, keepdims=True)
            
            return DummyModel()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #4ECDC4;
        text-align: center;
        margin: 1rem 0;
    }
    .digit-display {
        font-size: 4rem;
        font-weight: bold;
        color: #FF6B6B;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🔢 MNIST Digit Classifier</h1>
    <p>Draw a digit and let AI recognize it!</p>
</div>
""", unsafe_allow_html=True)

# Load model
try:
    model = load_or_create_mnist_model()
except Exception as e:
    st.error(f"❌ Critical error with model: {e}")
    st.stop()

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This MNIST classifier can recognize handwritten digits (0-9).

**How it works:**
1. Draw a digit in the canvas
2. The image is preprocessed to 28x28 pixels
3. AI model predicts the digit
4. View confidence scores for all digits
""")

st.sidebar.title("📊 Model Info")
st.sidebar.write(f"**Model Type:** {type(model).__name__}")

if hasattr(model, 'n_estimators'):
    st.sidebar.write(f"**Trees:** {model.n_estimators}")

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("✏️ Draw a Digit")
    
    # Drawing canvas
    canvas_result = st_drawable_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",  # Transparent fill
        stroke_width=20,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=400,
        height=400,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Control buttons
    col_clear, col_predict = st.columns(2)
    
    with col_clear:
        if st.button("🗑️ Clear Canvas", use_container_width=True):
            st.rerun()
    
    with col_predict:
        predict_button = st.button("🔍 Predict Digit", type="primary", use_container_width=True)

with col2:
    st.subheader("🎯 Prediction Results")
    
    if canvas_result.image_data is not None and predict_button:
        # Process the drawn image
        input_image = canvas_result.image_data
        
        # Convert to PIL Image
        img = Image.fromarray(input_image.astype('uint8'), 'RGBA')
        img = img.convert('L')  # Convert to grayscale
        
        # Resize to 28x28 (MNIST size)
        img = img.resize((28, 28), Image.LANCZOS)
        
        # Invert colors (MNIST has white digits on black background)
        img = ImageOps.invert(img)
        
        # Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = img_array / 255.0
        
        # Reshape for model prediction
        img_reshaped = img_array.reshape(1, -1)  # Flatten to 1D
        
        # Make prediction
        try:
            prediction = model.predict(img_reshaped)[0]
            probabilities = model.predict_proba(img_reshaped)[0]
            confidence = max(probabilities)
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <div class="digit-display">{prediction}</div>
                <h3>Predicted Digit: {prediction}</h3>
                <p><strong>Confidence:</strong> {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show processed image
            st.subheader("📷 Processed Image (28x28)")
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(img_array, cmap='gray')
            ax.axis('off')
            ax.set_title(f"Input to Model")
            st.pyplot(fig)
            plt.close(fig)  # Clean up
            
            # Confidence scores for all digits
            st.subheader("📊 Confidence Scores")
            
            # Create bar chart
            digits = list(range(10))
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#FF6B6B' if i == prediction else '#4ECDC4' for i in digits]
            bars = ax.bar(digits, probabilities, color=colors)
            
            ax.set_xlabel('Digit')
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Confidence for Each Digit')
            ax.set_xticks(digits)
            
            # Add value labels on bars
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)  # Clean up
            
            # Top 3 predictions
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            st.subheader("🏆 Top 3 Predictions")
            
            for i, idx in enumerate(top_3_indices):
                medal = ["🥇", "🥈", "🥉"][i]
                st.write(f"{medal} **Digit {idx}**: {probabilities[idx]:.1%}")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Make sure you drew something on the canvas.")
    
    elif canvas_result.image_data is not None:
        st.info("👆 Click 'Predict Digit' to classify your drawing!")
    else:
        st.info("👈 Draw a digit on the canvas to get started!")

# Example digits section
st.markdown("---")
st.subheader("💡 Example Digits to Try")

example_cols = st.columns(5)
example_digits = ['0️⃣', '1️⃣', '2️⃣', '3️⃣', '4️⃣']
example_tips = [
    "Draw a clear circle",
    "Draw a straight line", 
    "Make the curves distinct",
    "Add the top curve",
    "Don't forget the top line"
]

for i, (col, digit, tip) in enumerate(zip(example_cols, example_digits, example_tips)):
    with col:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: #f0f8ff; border-radius: 10px;">
            <div style="font-size: 2rem;">{digit}</div>
            <p style="font-size: 0.8rem; margin: 0;">{tip}</p>
        </div>
        """, unsafe_allow_html=True)

# More examples
example_cols2 = st.columns(5)
example_digits2 = ['5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣']
example_tips2 = [
    "Square top, curved bottom",
    "Circle with a stem",
    "Diagonal line with serif",
    "Two circles stacked",
    "Circle with a stem"
]

for i, (col, digit, tip) in enumerate(zip(example_cols2, example_digits2, example_tips2)):
    with col:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: #f0f8ff; border-radius: 10px;">
            <div style="font-size: 2rem;">{digit}</div>
            <p style="font-size: 0.8rem; margin: 0;">{tip}</p>
        </div>
        """, unsafe_allow_html=True)

# Tips section
st.markdown("---")
st.subheader("💡 Tips for Better Recognition")

tip_col1, tip_col2 = st.columns(2)

with tip_col1:
    st.markdown("""
    **Drawing Tips:**
    - Draw digits large and bold
    - Use the full canvas space
    - Keep digits centered
    - Make lines thick and clear
    """)

with tip_col2:
    st.markdown("""
    **For Better Accuracy:**
    - Draw digits similar to handwritten style
    - Avoid decorative fonts
    - Keep proportions realistic
    - Clear the canvas between digits
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🧠 Powered by Machine Learning | Built with Streamlit</p>
    <p>Model automatically downloads and trains on first run</p>
</div>
""", unsafe_allow_html=True)