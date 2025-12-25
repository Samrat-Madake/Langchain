# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv

# load_dotenv()

# llm = ChatGoogleGenerativeAI(
#     model="gemini-pro",
#     temperature=0
# )

# result = llm.invoke("What is the capital of India?")
# print(result.content)


# import google.generativeai as genai
# from dotenv import load_dotenv
# import os

# load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# model = genai.GenerativeModel("gemini-1.5-flash")
# response = model.generate_content("What is the capital of India?")
# print(response.text)
