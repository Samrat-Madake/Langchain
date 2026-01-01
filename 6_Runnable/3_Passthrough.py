from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the following joke - {topic}',
    input_variables=['topic']
)

# Chain 1 : Prompt1 -> LLM -> Parser
joke_generator_chain = prompt1 | llm | parser 

# Chain 2 : Parallel Chain
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough() ,
    'explanation': prompt2| llm | parser
})

#  Final chain 
final_chain = joke_generator_chain | parallel_chain
result = final_chain.invoke({'topic':'AI'})
print("Result: \n", result)


#  Visualization of the above chain
final_chain.get_graph().print_ascii() 


graph = final_chain.get_graph()
mermaid = graph.draw_mermaid()
print(mermaid)

graph.draw_mermaid_png(
    output_file_path="6_Runnable/3.png",
)
# OUTPUT

'''
Result: 
 {'joke': 'Why did the AI program go to therapy? \n\nBecause it was struggling to process its emotions.', 
 
 'explanation': 'This joke is a play on words and relies on the multiple meanings of the phrase "struggling to process its emotions."\n\nIn the context of a human who is struggling emotionally, "processing emotions" is a common idiomatic expression that means dealing with and managing one\'s feelings, especially difficult or painful ones.\n\nHowever, in the context of an AI program, "processing" has a different meaning. In computer science, processing refers to the computation or calculation performed by a computer or AI system. An AI program "processes" data by analyzing, interpreting, and generating output based on that data.\n\nSo, in this joke, the punchline "struggling to process its emotions" is a clever play on words, as it references both the human emotional struggle and the AI program\'s computational abilities. It\'s a lighthearted and humorous way to poke fun at the idea that AI systems, which are typically thought of as being logical and analytical, might struggle with emotions in a way that humans do.\n\nOverall, the joke requires a basic understanding of computer science and AI, as well as a bit of wordplay, to appreciate the humor.'}



'''

#  Visualization of the above chain
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
       +---------------------------------+
       | Parallel<joke,explanation>Input |
       +---------------------------------+
              **                  ***
            **                       **
+----------------+                     **
| PromptTemplate |                      *
+----------------+                      *
          *                             *
    +----------+                        *
    | ChatGroq |                        *
    +----------+                        *
          *                             *
+-----------------+             +-------------+
| StrOutputParser |             | Passthrough |
+-----------------+             +-------------+
                ***            ***
                   **        **
      +----------------------------------+
      | Parallel<joke,explanation>Output |
      +----------------------------------+

'''