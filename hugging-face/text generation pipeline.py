from transformers import pipeline

# Initialize the pipeline for text generation using the GPT-2 model
gpt2_pipeline = pipeline(task="text-generation", model="openai-community/gpt2")

# Generate text outputs
# max_new_tokens=10: Limits the generation to 10 new words/tokens
# num_return_sequences=2: Asks the model to generate 2 different possibilities
results = gpt2_pipeline("What if AI", max_new_tokens=10, num_return_sequences=2)

for result in results:
    print(result['generated_text'])
