from datasets import load_dataset

# Load the "validation" split of the TIGER-Lab/MMLU-Pro dataset
my_dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")

# Display dataset details
print(my_dataset)



# Filter the documents
# We use .filter() with a lambda function to keep only rows containing "football"
filtered = my_dataset.filter(lambda row: "football" in row["text"])

# Create a sample dataset
# We use .select() to pick specific rows by index (here, just the first one)
example = filtered.select(range(1))

print(example[0]["text"])
