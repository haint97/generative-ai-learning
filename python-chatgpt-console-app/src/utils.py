
# ANSI color codes
USER_COLOR = '\033[96m'      # Cyan
ASSISTANT_COLOR = '\033[92m' # Green
ERROR_COLOR = '\033[91m'     # Red
RESET_COLOR = '\033[0m'
SEPARATOR = f"{USER_COLOR}{'-'*10} Conversation {'-'*10}{RESET_COLOR}"

def format_message(user_input):
    return f"{USER_COLOR}User: {user_input}{RESET_COLOR}"

def format_response(response):
    return f"{ASSISTANT_COLOR}ChatGPT: {response}{RESET_COLOR}"

def handle_error(error):
    print(f"{ERROR_COLOR}Error: {error}{RESET_COLOR}")

def print_separator():
    print(SEPARATOR)
