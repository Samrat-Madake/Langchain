from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser # For Parsing output into String

from langchain_core.runnables import RunnableParallel # For Parallel Chain

from dotenv import load_dotenv
load_dotenv()

#  LLM 
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4,
)

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': prompt1| llm | parser,
    'linkedin': prompt2| llm | parser
})

result = parallel_chain.invoke({'topic':'Quantum Pyhsics'})
    
print("Tweet:", result['tweet'], "\n\n")
print("LinkedIn:", result['linkedin'])

# parallel_chain.get_graph().print_ascii() 

graph = parallel_chain.get_graph()

mermaid = graph.draw_mermaid()
print(mermaid)
graph.draw_mermaid_png(
    output_file_path="6_Runnable/2_Parallel.png",
)
# OUTPUT

'''
Tweet: "Mind-bending moment: Quantum Physics shows that particles can be in 2 places at once, and that information can be instantaneously transmitted across vast distances. The mysteries of the universe are still unraveling... #QuantumPhysics #Science #Mystery" 


LinkedIn: **Unlocking the Secrets of the Universe: Exploring the Fascinating World of Quantum Physics**

As we continue to push the boundaries of human knowledge, the field of quantum physics remains one of the most intriguing and rapidly evolving areas of research. From the mysteries of entanglement to the potential of quantum computing, the implications of quantum physics are vast and far-reaching.

**What is Quantum Physics?**

Quantum physics is a branch of physics that studies the behavior of matter and energy at the smallest scales, where the rules of classical physics no longer apply. At these scales, particles can exist in multiple states simultaneously, and their behavior is governed by the principles of wave-particle duality and superposition.        

**Key Concepts in Quantum Physics:**

1. **Entanglement**: The phenomenon where two or more particles become connected in such a way that their properties are correlated, regardless of the distance between them.
2. **Superposition**: The ability of particles to exist in multiple states simultaneously, which is a fundamental aspect of quantum computing.
3. **Quantum Tunneling**: The ability of particles to pass through barriers or gaps, even if they don't have enough energy to do so classically.
4. **Wave-Particle Duality**: The ability of particles to exhibit both wave-like and particle-like behavior.    

**Applications of Quantum Physics:**

1. **Quantum Computing**: The potential to develop computers that are exponentially faster and more powerful than classical computers.
2. **Cryptography**: The use of quantum mechanics to create unbreakable encryption methods.
3. **Materials Science**: The development of new materials with unique properties, such as superconductors and nanomaterials.
4. **Medical Imaging**: The use of quantum physics to develop new imaging techniques, such as MRI and PET scans.

**Join the Conversation:**

As we continue to explore the mysteries of quantum physics, we invite you to join the conversation. What are your thoughts on the potential applications of quantum physics? How do you see this field evolving in the coming years?

**Share your insights and experiences in the comments below!**

**Follow us for more updates on the latest developments in quantum physics and other cutting-edge fields!**     

#QuantumPhysics #QuantumComputing #Entanglement #Superposition #WaveParticleDuality #QuantumTunneling #MaterialsScience #MedicalImaging #Cryptography #Research #Innovation #FutureOfScience


'''

# Visualization of the above chain
'''
        +-------------------------------+
        | Parallel<tweet,linkedin>Input |
        +-------------------------------+
                ***             ***
              **                   **
            **                       **
+----------------+              +----------------+
| PromptTemplate |              | PromptTemplate |
+----------------+              +----------------+
          *                             *
          *                             *
          *                             *
    +----------+                  +----------+
    | ChatGroq |                  | ChatGroq |
    +----------+                  +----------+
          *                             *
          *                             *
          *                             *
+-----------------+            +-----------------+
| StrOutputParser |            | StrOutputParser |
+-----------------+            +-----------------+
                ***             ***
                   **         **
                     **     **
        +--------------------------------+
        | Parallel<tweet,linkedin>Output |
        +--------------------------------+

'''