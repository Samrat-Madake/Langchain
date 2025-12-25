from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant",temperature=1.5)
#  max_completion_tokens=512 : Used as parameter to limit the response length in token

result = llm.invoke("What is the capital of Maharashtra?")
print(result.content)  # Expected output: "The capital of Maharashtra is Mumbai."s
    