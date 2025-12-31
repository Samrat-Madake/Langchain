from dotenv import load_dotenv
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq

load_dotenv()

# Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",   # stable, free
    temperature=0
)

# Pydantic schema (FULLY SUPPORTED)
class Review(BaseModel):
    key_themes: List[str] = Field(..., description="Key themes discussed in the review")
    summary: str = Field(..., description="Concise 2–3 sentence summary of the review")
    sentiment: Literal["pos", "neg"] = Field(..., description="Overall sentiment of the review")
    pros: Optional[List[str]] = Field(None, description="Pros mentioned in the review")
    cons: Optional[List[str]] = Field(None, description="Cons mentioned in the review")
    name: Optional[str] = Field(None, description="Name of the reviewer if mentioned")

# Structured output model
structured_model = llm.with_structured_output(Review)

# Input text
result = structured_model.invoke("""
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse!
The Snapdragon 8 Gen 3 processor makes everything lightning fast—gaming, multitasking, and editing photos.
The 5000mAh battery lasts a full day, and the 45W fast charging is very convenient.

The 200MP camera is outstanding, especially night mode and zoom up to 30x.
However, the phone is bulky for one-handed use, has unnecessary bloatware, and is very expensive.

Review by Nitish Singh
""")

print(result)


#  Why ... matters (detailed)
'''
In Pydantic:
... = “This value is required”
No default = “This value may be missing”
'''