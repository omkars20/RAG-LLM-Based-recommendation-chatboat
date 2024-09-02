product_data = {
    "Wireless Earbuds": "These wireless earbuds provide excellent sound quality and comfort, with long battery life and noise-canceling features.",
    "Smartphone": "A sleek smartphone with a powerful processor, high-resolution camera, and an all-day battery.",
    "Laptop": "A lightweight laptop with a high-resolution display, fast processor, and long battery life, perfect for work and play.",
    "Smartwatch": "A stylish smartwatch with health monitoring features, including heart rate tracking, sleep analysis, and workout detection.",
    "Bluetooth Speaker": "A portable Bluetooth speaker with deep bass, long battery life, and water-resistant design, ideal for outdoor use."
}

# Convert the product data into a list of tuples
product_pairs = [(name, description) for name, description in product_data.items()]

from sentence_transformers import SentenceTransformer

# Load a pre-trained model for vectorization
model = SentenceTransformer('all-MiniLM-L6-v2')

# Vectorize the product descriptions
product_descriptions = [pair[1] for pair in product_pairs]
product_vectors = model.encode(product_descriptions)

import faiss
import numpy as np

product_vectors = np.array(product_vectors).astype('float32')

faiss_index = faiss.IndexFlatL2(product_vectors.shape[1])
faiss_index.add(product_vectors)

# Print the index details
print("FAISS index built successfully.")
print("Number of vectors in the index:", faiss_index.ntotal)

def get_closest_product(user_input):
    input_vector = model.encode([user_input]).astype('float32')
    distances, indices = faiss_index.search(input_vector, 1)
    return product_pairs[indices[0][0]]

def generate_recommendation(user_input, product_name, product_description):
    # A simple generative model with no external API
    recommendation =  (f"Based on your input, we recommend the {product_name}. "
                       f"It's {product_description.lower()}")
    return recommendation

from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['query']
        closest_product = get_closest_product(user_input)
        recommendation = generate_recommendation(user_input, closest_product[0], closest_product[1])
        return render_template('index.html', query=user_input, recommendation=recommendation)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
