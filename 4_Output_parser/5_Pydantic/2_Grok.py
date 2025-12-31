from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1
)

class Person(BaseModel):

    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

# template = PromptTemplate(
#     template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
#     input_variables=['place'],
#     partial_variables={'format_instruction':parser.get_format_instructions()}
# )
template = PromptTemplate(
    template=(
        "Generate the name, age (>18), and city of a fictional {place} person.\n\n"
        "{format_instruction}\n\n"
        "RULES:\n"
        "- Output ONLY valid JSON\n"
        "- Do NOT include explanations\n"
        "- Do NOT include Python code\n"
        "- Do NOT include markdown\n"
        "- Do NOT include backticks"
    ),
    input_variables=["place"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)


# prompt = template.invoke({'place':'sri lankan'})
# print("Prompt:\n", prompt)

chain = template | llm | parser

final_result = chain.invoke({'place':'Papua new Gunei'})

print(final_result)

json_result = final_result.model_dump_json()
print("JSON Result:\n", json_result)