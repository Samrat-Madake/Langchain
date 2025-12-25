from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)


messages=[
    SystemMessage(content='You are a Physics expert assistant that provides concise explanations.'),
    HumanMessage(content='Tell me about the theory of relativity.')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)


#  What we have done in this program 
'''
1. Imported necessary classes and functions from langchain_core and langchain_huggingface libraries.
2. Loaded environment variables using load_dotenv().
3. Created an instance of HuggingFaceEndpoint with a specified model repository and task.
4. Created a ChatHuggingFace model using the HuggingFaceEndpoint instance.
5. Defined a list of messages representing a conversation, including system and human messages.
6. Invoked the model with the list of messages to get a response.
7. Appended the AI's response to the messages list as an AIMessage.
8. Printed the complete list of messages, showing the conversation history.
'''

