from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import JsonOutputParser
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

parser = JsonOutputParser()
#  Template 
template = PromptTemplate(
    template="Generate a Fictional character with name age country hobby:\n""{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# prompt = template.format()
# result = llm.invoke(prompt)
# parsed_output = parser.parse(result.content)


#  Using Chain 
chain  = template | model | parser
parsed_output = chain.invoke({}) # No input variables so empty dict {} has to be passed
print("Generated Fictional Character:\n", parsed_output)

print(type(parsed_output))
