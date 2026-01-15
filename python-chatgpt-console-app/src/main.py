
import os
from chatgpt_client import ChatGPTClient
from context_manager import ContextManager
from utils import format_message, format_response, handle_error, print_separator



def main():
    # For LM Studio, api_key is not required, but can be set to 'lm-studio' by default
    api_key = "lm-studio"
    context_manager = ContextManager()
    chatgpt_client = ChatGPTClient(api_key)


    print_separator()
    print("Welcome to the ChatGPT Console Application!")
    print("Type 'exit' to quit the application.")
    print_separator()


    # Model selection
    model = input("Enter model name (default: meta-llama-3.1-8b-instruct): ").strip()
    if not model:
        model = "meta-llama-3.1-8b-instruct"

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
                print(format_message(user_input))
                print_separator()
                print("Generating response...\n")
                assistant_message = chatgpt_client.stream_response(messages, model=model)
                print_separator()
                print(format_response(assistant_message))
            except Exception as e:
                handle_error(e)
                continue
            context_manager.add_to_history({"role": "assistant", "content": assistant_message})

if __name__ == "__main__":
    main()
