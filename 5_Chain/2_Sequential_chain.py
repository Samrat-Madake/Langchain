from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

#  Model
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
)

# Template 1 
template1 = PromptTemplate(
    template=
        "Write a detailed report on {topic}.",
    input_variables=["topic"],
    )

# Template 2
template2 = PromptTemplate(
    template=
        "Generate 5 pointer summary from the {text}.",
    input_variables=["text"],
    )


# StrOutputParser
parser = StrOutputParser()

# Chain
chain = template1 | llm | parser | template2 | llm | parser

# Invoke
result = chain.invoke({'topic': 'Semiconductor technology'})
print(result)

# Visualise Chain
chain.get_graph().print_ascii()


#  OUTPUT
'''
Here's a 5-pointer summary of the **Semiconductor Technology Report**:

1. **Revolutionary Impact**: Semiconductor technology has transformed modern electronics, computers, smartphones, and other devices, changing the way we live, work, and communicate.

2. **Types of Semiconductors**: There are various types of semiconductors, including Bipolar Junction Transistors (BJTs), Field-Effect Transistors (FETs), Insulated-Gate Bipolar Transistors (IGBTs), Metal-Oxide-Semiconductor Field-Effect Transistors (MOSFETs), and Light-Emitting Diodes (LEDs), each with its unique applications.

3. **Applications and Industries**: Semiconductors are used extensively in various industries, including computers, smartphones, telecommunications, automotive, and medical devices, driving their growth and development.

4. **Challenges and Future Trends**: The semiconductor industry faces challenges such as shrinking transistor sizes, increasing complexity, and power consumption, but is expected to continue growing due to trends like 5G, artificial intelligence (AI), Internet of Things (IoT), and quantum computing.

5. **Innovation and Advancements**: The industry is innovating new materials and technologies to address the challenges it faces, such as developing new semiconductor materials and exploring quantum computing, ensuring its continued growth and evolution.    
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