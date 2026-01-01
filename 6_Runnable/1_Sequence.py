from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
)

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain = prompt1 | llm | parser | prompt2 | llm | parser

print(chain.invoke({'topic':'AI'}))

# chain.get_graph().print_ascii() 


graph = chain.get_graph()

mermaid = graph.draw_mermaid()
print(mermaid)

graph.draw_mermaid_png(
    output_file_path="chain_graph.png",
)



# Output 
'''
     +-------------+       
     | PromptInput |       
     +-------------+       
            *
    +----------------+
    | PromptTemplate |
    +----------------+
            *
      +----------+
      | ChatGroq |
      +----------+
            *
   +-----------------+
   | StrOutputParser |
   +-----------------+
            *
+-----------------------+
| StrOutputParserOutput |
+-----------------------+
            *
    +----------------+
    | PromptTemplate |
    +----------------+
            *
      +----------+
      | ChatGroq |
      +----------+
            *
   +-----------------+
   | StrOutputParser |
   +-----------------+
            *
+-----------------------+
| StrOutputParserOutput |
+-----------------------+

'''