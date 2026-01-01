from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda


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

# Chain 1 - Joke Generation
joke_generation_chain = prompt1 | llm | parser

# Chain 2 - Word Count using RunnableLambda
def word_count(text):
    return len(text.split())    


#  Parallel 
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})


# Final Chain combining both
chain = joke_generation_chain | parallel_chain

result = chain.invoke({'topic':'computers'})
print("result:", result)

# Visualize the chain
# chain.get_graph().print_ascii()

graph = chain.get_graph()

mermaid = graph.draw_mermaid()
print(mermaid)

graph.draw_mermaid_png(
    output_file_path="6_Runnable/4.png",
)
# OUTPUT
'''
result: {'joke': 'Why did the computer go to the doctor?\n\nBecause it had a virus.', 'word_count': 13}
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
    +--------------------------------+
    | Parallel<joke,word_count>Input |
    +--------------------------------+
            **               **
          **                   **
+-------------+            +------------+
| Passthrough |            | word_count |
+-------------+            +------------+
              **           **
                **       **
   +---------------------------------+
   | Parallel<joke,word_count>Output |
   +---------------------------------+

'''