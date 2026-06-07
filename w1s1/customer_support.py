from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")





COMPANY = "SpeakTech"                       # change it → persona changes

conversation =[
    SystemMessage(content=( f"You are a customer support agent for {COMPANY}, an electronics store. Be concise. 2-3 sentences max."))
              ]

def chat_turn(user_input):

    conversation.append(HumanMessage(content=user_input))

    response = llm.invoke(conversation)

    conversation.append(response)          # memory lives HERE — you own it
    
    print("-----------------------------------------------")
    print(f"User: {user_input}")
    print(f"Agent: {response.text}")
    print("-----------------------------------------------")

 

chat_turn("What is your return policy?")

chat_turn("How long do I have to return something?")

chat_turn("I bought a laptop 25 days ago. Can I still return it?")