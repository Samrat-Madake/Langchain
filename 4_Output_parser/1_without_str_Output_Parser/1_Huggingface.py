from langchain_core.prompts import PromptTemplate

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="HuggingFaceH4/zephyr-7b-beta",
    # repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Doest work as structured output not supported so we have to use Ouputput Parser
    task="text-generation",
    max_new_tokens=100
)

model = ChatHuggingFace(llm=llm)

# 1️⃣ Detailed Report Prompt
template1 = PromptTemplate(
    template=(
        "Write a detailed report about the following topic:\n\n"
        "{topic}\n\n"
        "Include an introduction, main content with subheadings, "
        "and a conclusion."
    ),
    input_variables=["topic"],
)

# 2️⃣ Summary Prompt
template2 = PromptTemplate(
    template=(
        "Summarize the following text concisely in 10 lines:\n\n"
        "{text}"
    ),
    input_variables=["text"],
)

# Step 1: Generate detailed report
prompt1 = template1.format(topic="Black Holes")
response1 = model.invoke(prompt1)

print("Detailed Report:\n")
print(response1.content)

# Step 2: Generate summary
prompt2 = template2.format(text=response1.content)
response2 = model.invoke(prompt2)

print("\nSummary:\n")
print(response2.content)