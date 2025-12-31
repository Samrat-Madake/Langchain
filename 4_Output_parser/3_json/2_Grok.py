from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import JsonOutputParser

llm = ChatGroq(model="llama-3.1-8b-instant",temperature=0.5)

parser = JsonOutputParser()
#  Template 
template = PromptTemplate(
    template="Generate a Fictional character with name age country hobby:\n""{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

#  Using Chain 
chain  = template | llm | parser
parsed_output = chain.invoke({}) # No input variables so empty dict {} has to be passed
print("Generated Fictional Character:\n", parsed_output)

print(type(parsed_output))

'''
we cannot enforce the schema on json output \

suppose our Query : 5 facts on Black Holes
we want
{
fact1 : "....",
fact2 : "....",
fact3 : "....",
fact4 : "....",
fact5 : "...."
}

but llm doesnt necessarily follow the schema and may give output like below
facts:["....","....","....","....","...."] }
So we have to validate the output schema using pydantic or other validation libraries
'''