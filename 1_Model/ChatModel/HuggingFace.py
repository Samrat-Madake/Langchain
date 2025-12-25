from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    # repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=100
)

chat_model = ChatHuggingFace(llm=llm)

result = chat_model.invoke("What is the capital of India?")
print(result.content)
