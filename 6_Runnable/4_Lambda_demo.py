from langchain_core.runnables import RunnableLambda

def word_count(text):
    return len(text.split())

runnable_word_count = RunnableLambda(word_count)

result = runnable_word_count.invoke("This is a sample sentence for counting words.")
print("Word Count:", result)

# Explanation:
'''
 In this code, we define a simple function `word_count` that takes a string input and returns the number of words in that string by splitting it on spaces.

 We then create a `RunnableLambda` instance using this function, allowing us to use it as a runnable component in a LangChain pipeline.

'''