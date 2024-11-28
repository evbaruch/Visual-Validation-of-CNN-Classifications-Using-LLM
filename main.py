from LmmApi.LLMInterface import LLMInterface
from LmmApi.chatGpt4o import ChatGPT4O
import os
from dotenv import load_dotenv
#from LmmApi.llama32Vision11b import llama32Vision11b

"""
LLMInterface is a class that acts as a bridge between the client code and the LLM strategies.
It allows the client code to switch between different strategies without changing its implementation.
The client code can interact with the LLM strategies through the LLMInterface class.

IN order to use the LLMInterface class, we set it startegy to the desired LLM strategy and then call the get_response method to get the response from the LLM strategy.

The LLMInterface class has two methods:
1. set_strategy(strategy): Sets the LLM strategy to be used.
2. get_response(self, prompt: str, image: str) -> str : Generates a response using the current strategy.


so to sum up , to use the LLMInterface class, we need to:
1. Create an instance of the LLMInterface class.
2. Set the strategy to the desired LLM strategy.
3. Call the get_response method to get the response from the LLM strategy.
"""

def foo1():
    # Initialize strategies
    
    
    # chatGpt4o Env setup ========================
    
    # Load environment variables from the .env file
    load_dotenv()
    # Initialize the OpenAI client with the provided API key
    api_key = os.getenv("OPENAI_API_KEY")
    
    # ============================================
    
    
    openai = ChatGPT4O(api_key=api_key)
    
    #llama = llama32Vision11b(api_key="claude-api-key")

    # Use OpenAI GPT
    llm_context = LLMInterface(strategy=openai)
    print(llm_context.get_response("what in this image?", "./images/dog.110.jpg"))

    # # Switch to Claude
    # llm_context.set_strategy(llama)
    # print(llm_context.get_response("What is the capital of France?"))




if __name__ == "__main__":
    foo1()
    
    
