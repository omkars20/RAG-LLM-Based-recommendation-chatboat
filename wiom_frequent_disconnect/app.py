import openai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
from flask import Flask, request, render_template

# Initialize the OpenAI client with the API key
client = OpenAI(api_key='api-key')
# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare the knowledge base
documents = [
    {
        "id": "doc_001", 
        "content": "67% of our customers are first-time WiFi owners. We need to educate them on 2.4G and 5G signals and their respective advantages and limitations, the range and speed limitations of WiFi, how many devices can be connected for optimal performance, and how to change their WiFi password to avoid unwanted users.", 
        "tags": ["customer_education", "wifi_signals", "password_change"]
    },
    {
        "id": "doc_002", 
        "content": "For our ONTs, the ideal optical power range is: Receive Power (Rx): -27 dBm to -8 dBm, Transmit Power (Tx): -8 dBm to +2 dBm. Symptoms when ONTs’ optical power is outside the ideal range include poor connection stability, slower speeds, and intermittent connectivity.", 
        "tags": ["partner_infrastructure", "optical_power", "connection_stability"]
    },
    {
        "id": "doc_003", 
        "content": "Sometimes there can be an issue in the router hardware or firmware that impacts internet connectivity. Symptoms include: Only ONT power light glowing with no SSID visible, SSID disappearing with Wiom Net and RALINK signal showing, device intermittently resetting, or PON LED disappearing.", 
        "tags": ["router_issues", "hardware_problems", "SSID_disappearing", "PON_LED"]
    },
    {
        "id": "script_001", 
        "content": "Sir, maine aapko proper guide kiya hai, aur abhi aapka internet bhi sahi chal raha hai. Kya aapki iss shikayat ko mai on-call resolve kar sakta hoon? Aap Se request hai ke bataye hue tarike se hi Wi-Fi use karein agle 24 ghanto ke liye. Agar aapko koi issue aata hai to aap hume dubara call kar sakte hain aur shikayat register kar sakte hain.", 
        "tags": ["on_call_resolution", "script"]
    },
    {
        "id": "script_002", 
        "content": "Sir, aapko ye issue baar-baar face karna pad raha hai, iske liye hum maafi chahte hain. Mai aapki yeh ticket priority mein ground team ko bhej raha hoon, jisse aapka yeh issue jaldi se jaldi solve kiya ja sake.", 
        "tags": ["escalate_to_partner", "script"]
    },
    {
        "id": "process_001", 
        "content": "Identify the Device Type: Determine whether the device in question is an ONT or a router. If it’s an ONT, check if the power is within the specified range. If the power is out of range, it may be a partner infrastructure issue. Create a ticket for Optical Power Out of Range and forward it to the partner without further troubleshooting.", 
        "tags": ["process_flow", "device_identification", "ONT"]
    },
    {
        "id": "process_002", 
        "content": "Router Troubleshooting: Ensure the patch cord is connected directly to the ONT and is not bent or tampered with. Restart the router if necessary. If the LOS light is red, it may indicate a wire breakage or technical issue, requiring further action.", 
        "tags": ["process_flow", "router_troubleshooting"]
    },
    {
        "id": "process_003", 
        "content": "Check the Router Lights: If there is a red LOS light, it indicates a possible wire breakage or technical issue. Ensure that the router is restarted properly, and if the problem persists, escalate the issue.", 
        "tags": ["process_flow", "router_lights", "LOS_light"]
    }
]


# Generate embeddings for the documents
document_texts = [doc['content'] for doc in documents]
document_embeddings = embedding_model.encode(document_texts, convert_to_tensor=True)
document_embeddings = document_embeddings.cpu().detach().numpy().astype('float32')

# Create FAISS index
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings)

# Mapping from index to documents
index_to_doc = {i: doc for i, doc in enumerate(documents)}

# Function to embed the query
def embed_query(query):
    query_embedding = embedding_model.encode([query], convert_to_tensor=True)
    return query_embedding.cpu().detach().numpy().astype('float32')

# Function to retrieve documents
def retrieve_documents(query_embedding, top_k=3):
    distances, indices = index.search(query_embedding, top_k)
    results = [index_to_doc[idx] for idx in indices[0]]
    return results

# Function to generate a response using OpenAI
def generate_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4"
        messages=[
            {"role": "system", "content": "You are a customer support assistant for Wiom, an affordable internet service provider."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.7
    )
    # Correctly extract the content from the response object
    message = response.choices[0].message.content.strip()
    return message

# Function to construct the prompt
def construct_prompt(query, documents):
    context = "\n\n".join([f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(documents)])
    prompt = f"""
Context:
{context}

Customer Query:
{query}

Provide a clear and concise answer to the customer's query using the context provided.
"""
    return prompt

# Function to get support response
def get_support_response(query):
    query_embedding = embed_query(query)
    relevant_docs = retrieve_documents(query_embedding)
    prompt = construct_prompt(query, relevant_docs)
    response = generate_response(prompt)
    return response

# Flask app setup
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def support():
    if request.method == 'POST':
        query = request.form['query']
        response = get_support_response(query)
        return render_template('support.html', query=query, response=response)
    return render_template('support.html')

if __name__ == '__main__':
    app.run(debug=True)