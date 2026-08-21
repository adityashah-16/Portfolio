import ollama
import streamlit as st
import numpy as np

st.title("My First RAG Bot")

# Read the programming file
with open("programming.txt", "r") as file:
    text = file.read()

# Split text into chunks
chunks = text.split("\n\n")

# Create embeddings for each chunk
chunk_vectors = []

for chunk in chunks:
    response = ollama.embed(
        model="nomic-embed-text",
        input=chunk
    )

    vector = response["embeddings"]
    chunk_vectors.append(vector)


# User question
question = st.chat_input("Ask Something...")

if question:

    # Display user question
    with st.chat_message("user"):
        st.write(question)

    # Create embedding for question
    response = ollama.embed(
        model="nomic-embed-text",
        input=question
    )

    question_vector = response["embeddings"][0]

    # Similarity search
    scores = []

    for vector in chunk_vectors:
        similarity = np.dot(question_vector, vector) / (
            np.linalg.norm(question_vector)
            * np.linalg.norm(vector)
        )

        scores.append(similarity)

    # Find the best matching chunk
    best_index = np.argmax(scores)
    best_chunk = chunks[best_index]

    # Create prompt
    prompt = f"""
Answer the question based on the context below.

Context:
{best_chunk}

Question:
{question}

Answer clearly and only use the information from the context.
"""

    # Ask Ollama
    response = ollama.chat(
        model="gemma",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    # Display answer
    with st.chat_message("assistant"):
        st.write(answer)