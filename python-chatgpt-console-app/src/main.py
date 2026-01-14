import os
from chatgpt_client import ChatGPTClient
from context_manager import ContextManager


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API key: ")
    context_manager = ContextManager()
    chatgpt_client = ChatGPTClient(api_key)

    print("Welcome to the ChatGPT Console Application!")
    print("Type 'exit' to quit the application.")

    # Model selection
    model = input("Enter model name (default: gpt-3.5-turbo, or try gpt-4): ").strip()
    if not model:
        model = "gpt-3.5-turbo"

    # System prompt
    system_prompt = input("Enter a system prompt to guide ChatGPT's behavior (or leave blank): ").strip()
    if system_prompt:
        context_manager.add_to_history({"role": "system", "content": system_prompt})

    while True:
        # Role selection
        role = input("Enter role (user/assistant/system, default: user): ").strip().lower()
        if not role:
            role = "user"
        if role not in ["user", "assistant", "system"]:
            print("Invalid role. Please enter 'user', 'assistant', or 'system'.")
            continue

        user_input = input(f"{role.capitalize()}: ")
        if user_input.lower() == 'exit':
            print("Exiting the application. Goodbye!")
            break

        # Add message to context history with selected role
        context_manager.add_to_history({"role": role, "content": user_input})

        # Only get response if user is the role
        if role == "user":
            messages = context_manager.get_history()
            try:
                print("ChatGPT: ", end="", flush=True)
                assistant_message = chatgpt_client.stream_response(messages, model=model)
            except Exception as e:
                print(f"Error: {e}")
                continue
            context_manager.add_to_history({"role": "assistant", "content": assistant_message})

if __name__ == "__main__":
    main()
