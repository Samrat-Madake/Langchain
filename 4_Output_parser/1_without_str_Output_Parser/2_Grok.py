from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=700
)

# 1️⃣ Detailed report prompt
template1 = PromptTemplate(
    template=(
        "Write a detailed report about the following topic:\n\n"
        "{topic}\n\n"
        "Include introduction, subheadings, and conclusion."
    ),
    input_variables=["topic"],
)

# 2️⃣ Summary prompt
template2 = PromptTemplate(
    template=(
        "Summarize the following text concisely:\n\n"
        "{text}"
    ),
    input_variables=["text"],
)

# Step 1
prompt1 = template1.format(topic="Black Holes")
response1 = llm.invoke(prompt1)

print("Detailed Report:\n")
print(response1.content)

# Step 2
prompt2 = template2.format(text=response1.content)
response2 = llm.invoke(prompt2)

print("\nSummary:\n")
print(response2.content)
