# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer for the pretrained model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Load the pretrained model
# Utilize the device_map argument to automatically allocate the model to available hardware, like a GPU
model = AutoModelForCausalLM.from_pretrained(
          "meta-llama/Llama-2-7b-chat-hf", 
          device_map='auto')  

# Define a new prompt for the model to generate a sci-fi story
prompt = "Please generate a short story set in a futuristic world."

# Convert the prompt into a format the model understands using the tokenizer, and move it to the GPU
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Use the model to generate text, setting a maximum token generation limit of 2000
outputs = model.generate(inputs["input_ids"], max_new_tokens=2000)

# Decode the generated tokens into text, skipping any special tokens like [CLS], [SEP], etc.
sci_fi_story = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated sci-fi story
print(sci_fi_story)
