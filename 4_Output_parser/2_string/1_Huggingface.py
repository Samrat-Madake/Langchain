from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    # repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    temperature=0.7
)
model = ChatHuggingFace(llm=llm)
# Prompts
template1 = PromptTemplate(
    template=(
        "Write a detailed report about the following topic:\n\n"
        "{topic}\n\n"
        "Include introduction, subheadings, and conclusion."
    ),
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template=(
        "Summarize the following text concisely:\n\n"
        "{text}"
    ),
    input_variables=["text"],
)

parser = StrOutputParser()

# ✅  Chain
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "black hole"})
print(result)
