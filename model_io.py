# Set OpenAI API Key
import os
# Import OpenAI model interface from LangChain
from langchain import llms
from langchain import prompts

# Creating the original template
template = """You are a professional copywriter for a toy store.\n
Could you provide an engaging short description for the {toy_name} priced at {price} dollars?
{format_instructions}"""

# Create a model instance
model = llms.OpenAI(model_name='text-davinci-003')

# Import structured output parser and ResponseSchema
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Define the response schema we want to receive
response_schemas = [
    ResponseSchema(name="description", description="Description of the flower"),
    ResponseSchema(name="reason", description="Why this particular description was chosen")
]

# Create an output parser
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Get format instructions
format_instructions = output_parser.get_format_instructions()

# Create a prompt from the original template, including the output parser's instructions
prompt = prompts.PromptTemplate.from_template(template, partial_variables={"format_instructions": format_instructions})

# Define more toys
toys = ["lego", "barbie", "teddy bear", "doll"]
prices = [50, 20, 30, 40]

# Create an empty DataFrame to store the results
import pandas as pd
df = pd.DataFrame(columns=["toy", "price", "description", "reason"])  # Declare the column names first

for toy, price in zip(toys, prices):
    # Input prompt
    input = prompt.format(toy_name=toy, price=price)

    # Get the output from the model
    output = model(input)

    # Parse the output
    parsed_output = output_parser.parse(output)
    parsed_output['toy'] = toy
    parsed_output['price'] = price

    # Save the parsed output to the DataFrame
    df.loc[len(df)] = parsed_output

# Print the DataFrame
print(df.to_dict(orient='records'))

# Save to csv
df.to_csv("toys_with_descriptions.csv", index=False)