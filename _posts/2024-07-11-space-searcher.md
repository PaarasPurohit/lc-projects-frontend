---
layout: post
section-type: post
has-comments: true
title: Getting Started with LangChain - Celestial Object Searcher
category: tech
---

# Introduction

This project was done using [this](https://www.youtube.com/watch?v=_FpT1cwcSLg&list=PLZoTAELRMXVORE4VF7WQ_fAl0L1Gljtar&index=2) video.

The link to the working project will be coming soon.

# Getting Started with Gemini

Since ChatGPT needed to be paid for, I used Gemini's API. After creating an API key, I started building my environment. 

I built my project off a Flask template, since I wanted it to be easily deployed through AWS. Here's what `requirements.txt` looked like:
Flask
requests
SQLAlchemy
Werkzeug
Flask_Login
Flask_SqlAlchemy
Flask_Migrate
Flask_Restful
Flask_Cors
PyJWT
pandas
numpy
matplotlib
seaborn
scikit-learn
openai
langchain
streamlit
google-generativeai
ipython
After running `pip install -r requirements.txt`, I was able to start. I followed this [documentation](https://ai.google.dev/gemini-api/docs/get-started/tutorial?lang=python#setup) to do what Krish did in the video without paying for any services.

# Imports & Configuration

First, I made `constants.py` and put my Gemini API key there. Then, I imported all the necessary libraries in a file called `demo.py`. This is what my imports looked like:


```python
import os
import json
from constants import gemini_api_key

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.memory import ConversationBufferMemory
from langchain.chains import SequentialChain
```

After that, I realized that since this isn't Google Colab, I couldn't use its local libraries, so I did some research. Below is how you can configure your API key to set up in your code (In line 4, I also went ahead and instantiated a model for us to use):


```python
GOOGLE_API_KEY = gemini_api_key
genai.configure(api_key=GOOGLE_API_KEY)

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
```

# Using LangChain's PromptTemplate Feature

In order to better take input from the user, LangChain has a built-in class called `PromptTemplate`. It allows you to set a variable and a prompt using that variable, then can call your LLM to generate a response. It can then later output those prompt responses as one parent chain.

This is useful if you want the user to input only a specific type of query but still get a lot of information.

To do that, we need to instantiate the `PromptTemplate` class for every prompt we want, specifying the input variables and template prompts, like so:


```python
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about the celestial object by the name of {name}"
)

second_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="What are the statistics for the celestial object {name}"
)

third_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Important additional information for celestial object {name}"
)
```

# Using LangChain's ConversationBufferMemory

It's good practice when working with AI to store conversations in memory. 

With our instantiated AI model, we can use LangChain's `ConversationBufferMemory` to store prompts and responses in the conversation memory.

To do that, we can instantiate `ConversationBufferMemory` for each prompt we give to the model, storing each conversation memory with its memory key:


```python
object_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
stats_memory = ConversationBufferMemory(input_key='name', memory_key='stats_history')
additional_info_memory = ConversationBufferMemory(input_key='name', memory_key='info_history')
```

# Chains

Now, it's time to actually get responses from our model. To do that, we'll instantiate LLM chains and specify our model as the LLM, our prompts, make sure they're verbose, set the output key, and map the memory keys to the ones we made above:


```python
chain = LLMChain(
    llm=llm,
    prompt=first_input_prompt,
    verbose=True,
    output_key='object_info',
    memory=object_memory
)

chain2 = LLMChain(
    llm=llm,
    prompt=second_input_prompt,
    verbose=True,
    output_key='stats',
    memory=stats_memory
)

chain3 = LLMChain(
    llm=llm,
    prompt=third_input_prompt,
    verbose=True,
    output_key='additional_info',
    memory=additional_info_memory
)
```

It'll be really painful to keep referencing multiple chains, especially if you want their data and there are more than just three, which is the case most of the time. Fortunately, LangChain has a class called `SequentialChain` that combines multiple chains.

To do that, let's instantiate `SequentialChain` to store all our chains in one parent chain:


```python
parent_chain = SequentialChain(
    chains = [
        chain,
        chain2,
        chain3
    ],
    input_variables = [
        'name'
    ],
    output_variables = [
        'object_info',
        'stats',
        'additional_info'
    ],
    verbose=True
)
```

# Getting Results

You could print the result in the console, but I found it best to store it in JSON. You could keep it as simple as a console app, but JSON allows me to then use the `Flask` library I imported earlier to then put the data on a server to later be fetched by a frontend I create.

To do that, let's first create a method that returns the LLM's output in JSON:


```python
def get_celestial_object_info(name):
    result = parent_chain({'name': name})
    
    output_json = {
        "object-information": result['object_info'],
        "statistics": result['stats'],
        "additional-information": result['additional_info'],
        "object-name-buffer": object_memory.buffer,
        "statistics-buffer": stats_memory.buffer,
        "additional-information-buffer": additional_info_memory.buffer
    }
    
    return output_json

```

Then, in `main.py`, route a sub-URL to call this method and return the returned JSON as the response.

For the purpose of my app, I made it a GET method, where the request parameter in the URL is the celestial object for query:


```python
@app.route('/api/space', methods=['GET'])
def space_info():
    name = request.args.get('name')
    if name:
        response = get_celestial_object_info(name)
        return jsonify(response)
    else:
        return jsonify({"error": "Missing 'name' parameter"}), 400
```

# Results

Here's the console output on the backend:

```
> Entering new SequentialChain chain...


> Entering new LLMChain chain...
Prompt after formatting:
Tell me about the celestial object by the name of Jupiter

> Finished chain.


> Entering new LLMChain chain...
Prompt after formatting:
What are the statistics for the celestial object Jupiter

> Finished chain.


> Entering new LLMChain chain...
Prompt after formatting:
Important additional information for celestial object Jupiter

> Finished chain.

> Finished chain.
127.0.0.1 - - [19/Jul/2024 20:08:18] "GET /api/space?name=Jupiter HTTP/1.1" 200 -
```

And when I make a `GET` request to `http://127.0.0.1:8086/api/space?name=Jupiter` for example, here's it working:

<img src="{{ site.baseurl }}/img/Screenshot 2024-07-19 200917.png">

# Use Cases

Since the JSON returns markdown, you could use `Markdownify` to put the results nicely on a frontend for the user to see.