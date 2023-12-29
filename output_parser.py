# ------Part 1
# Creating an instance of the model
from langchain import llms
model = llms.OpenAI(model_name='text-davinci-003')

# ------Part 2
# Creating an empty DataFrame to store the results
import pandas as pd
df = pd.DataFrame(columns=["toy_name", "price", "description", "reason"])

# Preparing the data
toys = ["Teddy Bear", "Lego Set", "Remote Control Car"]
prices = ["25", "40", "30"]

# Defining the desired data format to be received
from pydantic import BaseModel, Field
class ToyDescription(BaseModel):
    toy_name: str = Field(description="Name of the toy")
    price: int = Field(description="Price of the toy")
    description: str = Field(description="Description copy of the toy")
    reason: str = Field(description="Reason for writing the copy this way")

# ------Part 3
# Creating an Output Parser
from langchain.output_parsers import PydanticOutputParser
output_parser = PydanticOutputParser(pydantic_object=ToyDescription)

# Retrieving Format Instructions
format_instructions = output_parser.get_format_instructions()
# Printing the instructions
#print("Output format:", format_instructions)

# ------Part 4
# Creating the Prompt Template
from langchain import prompts
prompt_template = """As a professional copywriter for a toy store,
could you provide an attractive and brief description for a {toy} priced at {price} dollar?
{format_instructions}"""

# Creating a prompt based on the template, incorporating the instructions from the output parser
prompt = prompts.PromptTemplate.from_template(prompt_template, 
       partial_variables={"format_instructions": format_instructions}) 

# Printing the prompt
#print("Prompt:", prompt)

# ------Part 5
for toy, price in zip(toys, prices):
    # Preparing the model's input based on the prompt
    input = prompt.format(toy=toy, price=price)
    # Printing the prompt
    print("Prompt:", input)

    # Obtaining the model's output
    output = model(input)

    # Parsing the model's output
    parsed_output = output_parser.parse(output)
    parsed_output_dict = parsed_output.dict()  # Converting from Pydantic format to dictionary

    # Adding the parsed output to the DataFrame
    df.loc[len(df)] = parsed_output_dict

# Printing the data in dictionary format
print("Generated data:", df.to_dict(orient='records'))
