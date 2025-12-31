from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class student(BaseModel):
    name: str='sam' # default value
    age : Optional[int] = None # optional field
    email : EmailStr
    cgpa : float = Field(gt=0,lt=10,default=4,description="A decimal value representing the student's cumulative grade point average") 
#  create a dic 
new_student = {"name": "John Doe", "age": '21', "email": "john@gmail.com"} # value is not a valid email address: An email address must have an @-sign.
# new_student = {}

#  create an instance of the model using the dictionary
student_1 = student(**new_student)
# print(type(student_1.age)) #  age will be of type int : if not then pydantic will convert it to int: type coercion
print(student_1)

#  Converting Object to dictionary
student_dict = dict(student_1)
print("Dictionary",student_dict)


#  Converting Object to json
student_json = student_1.model_dump_json()
print(student_json)