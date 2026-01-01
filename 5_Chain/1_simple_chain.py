from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1️⃣ Template
template = PromptTemplate(
    template="Write 5 interesting facts about {topic}.",
    input_variables=["topic"],
)

# 2️⃣ Model
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
)

# 3️⃣ StrOutputParser
parser = StrOutputParser()

# 4️⃣ Chain
chain = template | llm | parser

# 5️⃣ Invoke
result = chain.invoke({'topic': 'space exploration'})
print(result)

# 6️⃣ Visulaise Chain
chain.get_graph().print_ascii()


#  OUTPUT
'''
Here are 5 interesting facts about space exploration:

1. **The Farthest Human-Made Object**: The Voyager 1 spacecraft, launched in 1977, is the farthest human-made object in space, with a distance of over 14.5 billion miles (23.3 billion kilometers) from Earth. It is now in the interstellar medium, the region of space outside our solar system.

2. **The International Space Station**: The International Space Station (ISS) is the largest human-made object in space, with a mass of over 450,000 kilograms (1 million pounds). It orbits the Earth at an altitude of around 250 miles (400 kilometers) and has been continuously occupied by astronauts since November 2000.

3. **The Fastest Spacecraft**: The Helios 2 spacecraft, launched in 1976, holds the record for the fastest spacecraft ever built, reaching a speed of over 157,000 miles per hour (253,000 kilometers per hour) as it flew by the Sun. This speed is so high that it would allow the spacecraft to travel from Earth to the Sun in just 6 hours.

4. **The Longest Spacewalk**: The longest spacewalk in history was performed by Russian cosmonaut Alexei Leonov in 1965, lasting 12 hours and 9 minutes. During this record-breaking spacewalk, Leonov and his colleague, Pavel Belyayev, performed a series of experiments and tests to prepare for future space missions.

5. **The Coldest Place in Space**: The Boomerang Nebula is a cloud of gas and dust located about 5,000 light-years from Earth, and it is the coldest place in space, with a temperature of just -272°C (-458°F). This extreme cold is caused by the slow expansion of the nebula, which creates a region of space where the gas is cooled to nearly absolute zero.

These facts showcase the incredible achievements and discoveries of space exploration, from the farthest human-made objects to the coldest places in space.
     +-------------+       
     | PromptInput |
     +-------------+
            *
            *
            *
    +----------------+
    | PromptTemplate |
    +----------------+
            *
            *
            *
      +----------+
      | ChatGroq |
      +----------+
            *
            *
            *
   +-----------------+
   | StrOutputParser |
   +-----------------+
            *
            *
            *
+-----------------------+
| StrOutputParserOutput |
+-----------------------+
'''