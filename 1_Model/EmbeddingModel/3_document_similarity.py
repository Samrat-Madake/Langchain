from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]
query = 'tell me about kohli'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)

'''
🧠 What’s happening step-by-step

1️⃣ Convert text to numbers (embeddings)

Each document and the query are converted into vectors using
sentence-transformers/all-MiniLM-L6-v2.

2️⃣ Compare query with all documents

cosine_similarity measures how close the query vector is to each document vector

Score ranges from –1 to 1 (closer to 1 = more similar)

3️⃣ Pick the best match

The document with the highest similarity score is selected

🧪 For your query
tell me about kohli


The model correctly matches:

Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.


because “kohli” is semantically closest to that document.

📝 One-line takeaway (Exam-ready)

The code uses text embeddings and cosine similarity to retrieve the most semantically relevant document for a given query.
'''