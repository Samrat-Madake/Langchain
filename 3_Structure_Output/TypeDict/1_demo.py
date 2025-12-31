from typing import TypedDict


class Person(TypedDict):
    name: str
    age: int


new_person: Person = {
    'name': 'sam' ,
    'age': 30
}
print(new_person)