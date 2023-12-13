# this chatbot was made using a tutorial, and the tutorial guy made it using chatGPT lol,
# anyway to use correctly make a knowledge_base.json file and do this inside
# {
#     "questions": [
#        
#     ]
# }

import json
from difflib import get_close_matches

# this function returns the data from the 'knowledge_base.json' file into the program
def load_knowledge_base(file_path:str) -> dict:
    with open (file_path, 'r') as file:
        data: dict = json.load(file)
    return data

# this function loads the answer that the user intered into the json file
def save_knowledge_base(file_path:str, data:dict):
    with open (file_path, 'w') as file:
        json.dump(data, file, indent=2)

def find_best_match(user_question:str, questions: list[str]) -> str | None:
    matches:list = get_close_matches(user_question, questions, n=1, cutoff=0.6)
# n=1 will return the top 1 answer, cutoff is kind of like an accuracy measure, the default is 0.6, do not put it high cus then it will start returning
# exact matches and we don't want that in our chatbot
    return matches[0] if matches else None

# this function will get the answer for the question if that question existed in the json file
def get_answer_for_question(question:str, knowledge_base:dict) -> str | None:
    for q in knowledge_base['questions']:
        if q['question']==question:
            return q['answer']

def chat_bot():
    # loading the data
    knowledge_base:dict = load_knowledge_base('knowledge_base1.json')

    while True:
        user_input:str = input('You: ')
        if user_input.lower()=='quit':
            break
            
        # this will look for the best match inside of our json file
        best_match:str|None = find_best_match(user_input, [q['question'] for q in knowledge_base['questions']])

        if best_match:
            answer:str = get_answer_for_question(best_match, knowledge_base)
            print('Bot: ', answer)
        else:
            print("Bot: I don't know the answer, can you teach me?")
            new_answer:str = input('type the answer or "skip" to skip')

            if new_answer.lower() != 'skip':
                knowledge_base['questions'].append({'question': user_input, 'answer': new_answer})
                save_knowledge_base('knowledge_base.json', knowledge_base)
                print('thank you I have learnt something new!')

chat_bot()
