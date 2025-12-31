from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

# from langchain_groq import ChatGroq
# llm = ChatGroq(model="llama-3.1-8b-instant",temperature=1.5)


llm = HuggingFaceEndpoint(
    # repo_id="mistralai/Mistral-7B-Instruct-v0.2", // Chattemplate model
    # repo_id="HuggingFaceH4/zephyr-7b-beta",          // Chattemplate model
    # repo_id="Qwen/Qwen3-4B-Instruct-2507",
    # repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

#  Create review output structure
class Review(TypedDict):
    summary: str
    sentiment: str

# structured_model = llm.with_structured_output(Review)

# result = structured_model.invoke("""
# The Logitech Pop Icon Keys stands out as one of the best wireless keyboards of 2025, praised for its comfortable typing experience, compact design, and long battery life. It easily pairs with up to three Bluetooth devices, making it highly versatile for use across computers, tablets, and phones. Its small size reduces desk clutter and enhances portability, while the colorful, rounded keys add a playful aesthetic. Though the logo on the spacebar and small arrow keys may not appeal to everyone, these are minor drawbacks in an otherwise excellent keyboard.
# """)
result = model.invoke("""
Return the answer ONLY in valid JSON with keys:
summary (string), sentiment (string)
Text :
The Logitech Pop Icon Keys stands out as one of the best wireless keyboards of 2025, praised for its comfortable typing experience, compact design, and long battery life. It easily pairs with up to three Bluetooth devices, making it highly versatile for use across computers, tablets, and phones. Its small size reduces desk clutter and enhances portability, while the colorful, rounded keys add a playful aesthetic. Though the logo on the spacebar and small arrow keys may not appeal to everyone, these are minor drawbacks in an otherwise excellent keyboard.
""")

print(result.content)
