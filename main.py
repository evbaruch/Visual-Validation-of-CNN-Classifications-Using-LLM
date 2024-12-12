from LmmApi.LLMInterface import LLMInterface
from LmmApi.chatGpt4o import chatGpt4o
from LmmApi.llama32Vision11b import llama32Vision11b

def foo1():
    # Initialize strategies
    openai = chatGpt4o(api_key="openai-api-key")
    llama = llama32Vision11b()

    # Use OpenAI GPT
    llm_context = LLMInterface(openai)
    print(llm_context.get_response("What is the capital of France?"))

    # Switch to Claude
    llm_context.set_strategy(llama)
    print(llm_context.get_response("What's in the picture?", 'images\Dog2.jpg'))




if __name__ == "__main__":
    foo1()
