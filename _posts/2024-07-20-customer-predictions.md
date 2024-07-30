---
layout: post
section-type: post
has-comments: true
title: Predicting Customer Reviews with LangChain's Shot-Generative AI
category: tech
---

# Introduction

Today, I created an application that takes in a customer's review and returns a predicted review from 1 to 5 that accurately represents the customer's review.

This project was done using [this](https://www.youtube.com/watch?v=_FpT1cwcSLg&list=PLZoTAELRMXVORE4VF7WQ_fAl0L1Gljtar&index=3) video.

The link to the working project will be coming soon.

# Imports & Configuration

First, I imported the necessary libraries. The only new import here is `FewShotTemplate`, which is generally used if you want to include your prompt, essentially giving you control over the prompt engineering:


```python
from constants import gemini_api_key

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
```

Then, I instantiated the Gemini model for the program:


```python
GOOGLE_API_KEY = gemini_api_key
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
```

# Custom Prompt Engineering

The essence of prompt engineering includes wording prompts in better ways and also training AI models to respond better according to user specifications. These specifications come when we give examples of responses to the AI. To do that, I created a JSON object called `examples`:


```python
examples = [
    {
        "description":"Outstanding service!",
        "review":"5"
    },
    {
        "description":"Not very good service...",
        "review":"1"
    },
    {
        "description":"I had an okay time.",
        "review":"3"
    }
]
```

Then, I created a simple `PromptTemplate` with two inputs, `description` and `review`, for each column in the JSON examples:


```python
examples = [
    {
        "description":"Outstanding service!",
        "review":"5"
    },
    {
        "description":"Not very good service...",
        "review":"1"
    },
    {
        "description":"I had an okay time.",
        "review":"3"
    }
]

example_formatter_template = """Description: {description}
Review: {review}
"""
```


I used a different kind of template. Instead of putting the instructions right into the template, I created a template for the output itself to use. You'll see why now.

# Langchain's FewShotPromptTemplate

The `FewShotPromptTemplate` class is different than a regular `PromptTemplate` because it allows the user to specify more parameters. One of these parameters is `examples`, which allowed me to set my own examples and do prompt engineering.

The more examples you add in the JSON, the more accurate it'll be to your specifications.

One thing to consider with `FewShotPromptTemplate` is that when passing a `PromptTemplate` for the `example_prompt`, the `template` parameter inside the `PromptTemplate` needs to be for the output itself, not as an instruction. The instruction itself comes in the `FewShotPromptTemplate` parameter `prefix`:


```python
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt = example_prompt,
    prefix="Based on the input, give a review. It should be a number from 1 to 5 and accurately reflect the description input\n",
    suffix="Review: {input}\nDescription: ",
    input_variables=["input"],
    example_separator="\n",
)
```

# Output

Similar to how I did last time, I'm returning the output as JSON so I can fetch it later. To do that, I first created a chain that uses the instantiated Gemini LLM to respond to `few_shot_prompt`. After that, I made a simple method to return the results in JSON:


```python
chain = LLMChain(
    llm = llm,
    prompt = few_shot_prompt
)

def get_predicted_review(input):
    result = chain({'input': input})
    
    output_json = {
        "description": result['input'],
        "predicted_review": result['text'],
    }
    
    return output_json
```

In `main.py`, I created a GET routing to call this method:


```python
@app.route('/api/customer', methods=['GET'])
def customer_info():
    desc = request.args.get('description')
    if desc:
        response = get_predicted_review(desc)
        return jsonify(response)
    else:
        return jsonify({"error": "Missing 'name' parameter"}), 400
```

# Testing

<img src="{{ site.baseurl }}/img/Screenshot (151).png">
