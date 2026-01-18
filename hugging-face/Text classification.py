from transformers import pipeline

# Create a pipeline for grammar checking
grammar_checker = pipeline(
  task="text-classification",
  model="abdulmatinomotoso/English_Grammar_Checker"
)

# Check grammar of the input text
output = grammar_checker("I will walk dog")
print(output)



# Create the pipeline
classifier = pipeline(task="text-classification", model="cross-encoder/qnli-electra-base")

# Predict the output
# The model analyzes if the second part of the text answers the question in the first part.
output = classifier("Where is the capital of France?, Brittany is known for its stunning coastline.")

print(output)





text = "AI-powered robots assist in complex brain surgeries with precision."

# Create the pipeline
classifier = pipeline(task="zero-shot-classification", model="facebook/bart-large-mnli")

# Create the categories list
categories = ["politics", "science", "sports"]

# Predict the output
output = classifier(text, categories)

# Print the top label and its score
print(f"Top Label: {output['labels'][0]} with score: {output['scores'][0]}")
