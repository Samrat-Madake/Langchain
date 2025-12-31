from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser

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
        "Summarize the following text concisely in 10 Lines:\n\n"
        "{text}"
    ),
    input_variables=["text"],
)

parser = StrOutputParser()

chain = template1 | llm | parser | template2 | llm | parser 

result = chain.invoke({'topic': 'black hole'})

print(result)


'''
👉 Without StrOutputParser, chaining breaks because:
PromptTemplate expects strings as input
Chat model returns AIMessage

BAM !!!!
'''

'''
OUTPUT 


Here is a concise summary of the text in 10 lines:

Black holes are regions of spacetime with incredibly strong gravitational pull.
Their event horizon marks the boundary beyond which nothing, including light, can escape.
A black hole forms when a massive object, like a star, collapses in on itself.
Its density warps spacetime, creating an event horizon and a singularity at its center.
There are four types of black holes: stellar, supermassive, intermediate-mass, and primordial.
Stellar black holes have masses between 1.4 and 20 solar masses.
Supermassive black holes are found at the centers of galaxies and have massive masses.
Observational evidence for black holes includes X-rays, gamma rays, and radio waves from hot gas and matter.
The study of black holes has implications for quantum gravity and cosmology.
It has also led to a deeper understanding of the fundamental laws of physics in extreme environments.
'''