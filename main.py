from LmmApi import LLMInterface
from LmmApi import chatGpt4o
from LmmApi import llama32Vision11b

def foo1():
    # Initialize strategies
    openai = chatGpt4o(api_key="openai-api-key")
    llama = llama32Vision11b(api_key="claude-api-key")

    # Use OpenAI GPT
    llm_context = LLMInterface(strategy=openai)
    print(llm_context.get_response("What is the capital of France?"))

    # Switch to Claude
    llm_context.set_strategy(llama)
    print(llm_context.get_response("What is the capital of France?"))




if __name__ == "__main__":
    foo1()
