samples = [
  {
    "toy_type": "Building Blocks",
    "occasion": "Learning",
    "ad_copy": "Building blocks, perfect for sparking creativity, are the ideal choice for educational play."
  },
  {
    "toy_type": "Teddy Bear",
    "occasion": "Comfort",
    "ad_copy": "Teddy bears, a cuddly companion, offer comfort and friendship to children."
  },
  {
    "toy_type": "Lego sports car",
    "occasion": "Birthday",
    "ad_copy": "Lego sport cars bring thrilling adventures to playtime, perfect for a birthday gift."
  },
  {
    "toy_type": "Puzzle",
    "occasion": "Brain Teasing",
    "ad_copy": "Puzzles challenge the mind and offer hours of problem-solving fun for all ages."
  }
]

from langchain.prompts.prompt import PromptTemplate
template = "Toy Type: {toy_type}\nOccasion: {occasion}\nAd Copy: {ad_copy}"
prompt_sample = PromptTemplate(
    input_variables=["toy_type", "occasion", "ad_copy"], 
    template=template)

# 3. Create a FewShotPromptTemplate object
from langchain.prompts.few_shot import FewShotPromptTemplate
prompt = FewShotPromptTemplate(
    examples=samples,
    example_prompt=prompt_sample,
    suffix="Toy Type: {toy_type}\nOccasion: {occasion}",
    input_variables=["toy_type", "occasion"]
)
#print(prompt.format(toy_type="Miniature Train", occasion="Playtime"))

# 4. Pass the prompt to the large model
from langchain.llms import OpenAI
model = OpenAI(model_name='text-davinci-003')
result = model(prompt.format(toy_type="Miniature Train", occasion="Playtime"))
#print(result)

# 5. Utilize the Example Selector
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Initialize the Example Selector
example_selector = SemanticSimilarityExampleSelector.from_examples(
    samples,
    OpenAIEmbeddings(),
    Chroma,
    k=1
)

# Create a FewShotPromptTemplate object using the Example Selector
prompt = FewShotPromptTemplate(
    example_selector=example_selector, 
    example_prompt=prompt_sample, 
    suffix="Toy Type: {toy_type}\nOccasion: {occasion}", 
    input_variables=["toy_type", "occasion"]
)
print(prompt.format(toy_type="Toy Car", occasion="Birthday"))
