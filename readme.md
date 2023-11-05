+ TODO:
  + Process all original dataset to the same format.
  + Add function and instruction of direct prompt augmentation.
  + Add function of one round request.
  + To speed up, try to make request asynchronous.

## Instructions
### 1. Preparations 
+ Set ```OPENAI_API_KEY``` to your OpenAI api key in the environment variable.

### 2. Run
+ You can run by ```python gennerate.py```. Information about parameters can by found by ```python generate.py -h```.

### 3. Prompt
+ Prompt augmentation
  + Prompts can be found under ```.\instruction```
  + I only implement augmentation by generating context first (by-context). It first generates context, asks questions in different ways and answers the questions.
  + Multi-round is like our interaction with ChatGPT on the website. Chat history is recorded.
+ Prompt selection
  + I only implement selection on snowballing untruthfulness. It first asks question, then verifies the answer by asking the explanation in the answer.
