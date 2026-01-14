
from openai import OpenAI

class ChatGPTClient:

    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)


    def send_message(self, messages, stream=False, model="gpt-3.5-turbo"):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream
        )
        return response

    def stream_response(self, messages, model="gpt-3.5-turbo"):
        response = self.send_message(messages, stream=True, model=model)
        collected_chunks = []
        for chunk in response:
            chunk_message = getattr(chunk.choices[0].delta, 'content', '')
            print(chunk_message, end='', flush=True)
            collected_chunks.append(chunk_message)
        print()  # for newline after streaming
        return ''.join(collected_chunks)

    def get_response(self, user_message):
        messages = [{"role": "user", "content": user_message}]
        response = self.send_message(messages)
        return response.choices[0].message.content
