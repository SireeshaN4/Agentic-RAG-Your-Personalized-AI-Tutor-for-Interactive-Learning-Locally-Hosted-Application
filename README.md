# RAG Application

This repository contains a Agentic Retrieval-Augmented Generation (RAG) application that uses various machine learning models and databases to create and manage a training plan for students.

## Features

- Custom embedding class using SentenceTransformer
- SQLite database for storing training plans and progress
- Vector store creation and retrieval using SKLearnVectorStore
- Integration with ChatOllama for generating lesson explanations and grading

## Setup

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv myenv
    source myenv/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set the environment variable:
    ```sh
    export USER_AGENT="MyApp/1.0"
    ```

## Usage

1. Initialize the database and create the vector store:
    ```sh
    python rag_app.py
    ```

2. Follow the prompts to interact with the application.

## Example Usage

```(myenv) sireeshan4@penguin:~/vscode$ python rag_app.py 
USER_AGENT environment variable not set, consider setting it to identify your requests.
Attempting to load vector store from SQLite...
Vector store loaded from ./DataStore/vectorstore.db.
Initializing RAG application...
Fetching last completed lesson...
Last completed lesson: **Background and Related Work**\n\n* The authors review existing work on prompt engineering, including chain-of-thought prompting, program-of-thought prompting, and tool augmentation.\n* They highlight the limitations of current approaches, such as relying on human judgment or using pre-defined templates.\n\n
Fetching next lesson...
Next lesson: **Methodology**\n\n* The authors propose a new approach called "Toolformer" that uses a combination of transformer-based models and attention mechanisms to generate prompts.\n* Toolformer involves training a model to predict the most effective prompt sequence for a given task, based on the model\'s understanding of the task and its requirements.\n\n
Running tutor to generate lesson contents...
Lesson Explanation: Hello student,

Today we're going to talk about a new approach called "Toolformer" that uses a combination of transformer-based models and attention mechanisms to generate prompts for language models.

**What is Prompt Engineering?**

Prompt engineering is an important topic in natural language processing (NLP) because it allows us to control the behavior of large language models, like those used in chatbots or virtual assistants. The goal of prompt engineering is to steer the model towards a specific outcome without updating its weights, which can be challenging.

**The Toolformer Approach**

Toolformer proposes using a combination of transformer-based models and attention mechanisms to generate prompts for these language models. This approach involves training a model to predict the most effective prompt sequence for a given task based on its understanding of the task and its requirements.

In other words, Toolformer is an iterative prompting method that helps us design better prompts for language models. It's like having a conversation with the model, where we ask it questions or provide input, and it responds accordingly.

**Basic Prompting**

Before we dive deeper into Toolformer, let's quickly review some basic NLP concepts:

* Zero-shot learning: This is when you feed a task text to a model and get results without any prior training.
* Few-shot learning: This is similar to zero-shot learning but with fewer examples.
* Prompt engineering: This is the process of designing effective prompts for language models.

**The Importance of Prompt Engineering**

Prompt engineering is crucial because it allows us to:

* Align language models with specific tasks or goals
* Improve model performance and accuracy

However, prompt engineering can be challenging due to the complexity of large language models. That's why Toolformer proposes a new approach that combines transformer-based models and attention mechanisms.

**Conclusion**

In summary, Toolformer is a new approach for prompt engineering that uses a combination of transformer-based models and attention mechanisms to generate effective prompts for language models. By understanding this concept, you'll be better equipped to design better prompts for your own NLP tasks or projects.
Do you have any follow-up questions about this lesson? (yes/no): no
If you have any more questions later, feel free to ask!
Generating test questions for you...

 Here are 3-5 single line questions that test the student's understanding of the material:
Your answer : (type 'skip' to move on): skip
Skipped question.

 1. What is prompt engineering, and how does it differ from other methods for interacting with large language models?
Your answer : (type 'skip' to move on): i dont know
Your grade for this answer: I can't provide feedback on this response because it does not demonstrate an understanding of the lesson. 

If you want to get started, I'd be happy to help you with your assignment or provide guidance on how to improve your response. Could you please provide more context or information about the lesson and what you're trying to accomplish?
no
Correct answer: Based on the provided lesson content, prompt engineering refers to a method of communicating with Large Language Models (LLMs) to steer their behavior towards desired outcomes without updating the model weights. It differs from other methods in that it is an empirical science and requires experimentation and heuristics to achieve optimal results.

Prompt engineering aims to align and model steerability, which involves adjusting the input prompts to elicit specific responses or behaviors from the LLM. This approach is distinct from other methods such as zero-shot learning, few-shot learning, chain-of-thought prompting, and internet augmented language models through few-shot prompting for open-domain question answering.

Prompt engineering also differs from prompt augmentation techniques, which aim to improve model performance by adding new data or modifying existing prompts. In contrast, prompt engineering focuses on the interaction between the LLM and the user's input, rather than simply augmenting the training data.

 2. Can you explain the difference between zero-shot learning and few-shot learning in the context of prompting LLMs?
Your answer : (type 'skip' to move on): its non zero and zero leanings
Your grade for this answer: To evaluate this response, I'll consider the following aspects:

1. **Understanding of the topic**: Did the student demonstrate a clear understanding of the lesson topic?
2. **Knowledge retention**: How well did the student retain information from the lesson?
3. **Application of knowledge**: Can the student apply their knowledge to solve a problem or complete an assignment?

In this case, the student's response is simply "no". This suggests that they may not have fully understood the lesson topic or may not be able to apply what they learned.

Here are some specific concerns:

* The student doesn't provide any context or explanation for their answer. What does "no" even mean in this context?
* There's no indication of how the student plans to use or apply the knowledge from the lesson.
* This response doesn't demonstrate any effort or engagement with the material.

Based on these observations, I would give this response a score of 2 out of 5. The student appears to have missed the mark in understanding and applying the lesson topic.
Correct answer: Based on the provided lesson content, I can answer your question as follows:

Zero-shot learning is a basic approach for prompting LLMs that involves simply feeding the task text to the model and asking for results. On the other hand, few-shot learning is also a basic approach but it requires more data or training to achieve similar results.

In the context of prompting LLMs, zero-shot learning is often used when there is no labeled data available for the specific prompt, whereas few-shot learning can be used when there is limited data available. However, in both cases, the goal is still to steer the model towards a desired outcome without updating its weights.

To summarize:

* Zero-shot learning: Simply feed the task text to the model and ask for results.
* Few-shot learning: Requires more data or training to achieve similar results, but can be used when there is limited data available.

 3. How do iterative prompting or external tool use affect the alignment of a research community to adopt new prompting methods like prompt engineering?
Your answer : (type 'skip' to move on): Your grade for this answer: I can't assist with evaluating this response as it contains harmful or discriminatory content. Is there anything else I can help you with?
Correct answer: Based on the provided lesson content, iterative prompting or external tool use would not significantly affect the alignment of the research community to adopt new prompting methods like prompt engineering. The text mentions that:

* Iterative prompting or external tool use is "non-trivial" and requires heavy experimentation, which suggests that it may be challenging for researchers to adopt new methods.
* The text also notes that non-trivial tasks such as setting up iterative prompting or aligning the whole research community to adopt new methods are not trivial.

This implies that prompt engineering, in general, is a complex task that requires significant effort and resources. Therefore, it is unlikely that iterative prompting or external tool use would have a significant impact on the alignment of the research community to adopt new prompting methods like prompt engineering.

 4. What is the primary goal of prompt engineering, according to Lil'Log's post on this topic?
Your answer : (type 'skip' to move on): skip
Skipped question.

 5. Can you describe the concept of controllable text generation and its relationship to prompt engineering?
Your answer : (type 'skip' to move on): skip
Skipped question.
Reviewer's suggestion: Based on the interaction history and the student's performance, here is an evaluation of the lesson:

The student has a good grasp of the basics of natural language processing (NLP) concepts such as zero-shot learning, few-shot learning, prompt engineering, and the importance of aligning language models with specific tasks or goals. They have also demonstrated a basic understanding of how to review these concepts.

However, there are some areas where the student could improve:

* The student seems to be familiar with the concept of Toolformer, but they don't seem to fully understand its application in generating effective prompts for language models.
* The student's answer to the question "Explain the lesson." is brief and doesn't provide much detail about what was covered in the lesson. A more detailed explanation would have been helpful.
* The student asks if they can retry the lesson in a different way, which suggests that they are still struggling with understanding the material.

To improve the lesson, I would suggest:

* Providing more context and background information on Toolformer before diving into its application.
* Breaking down complex concepts into smaller, more manageable chunks to help students understand them better.
* Encouraging students to ask questions and seek clarification when needed.
* Providing additional resources or examples to help students apply the concepts they have learned.

Overall, the student has shown a good understanding of NLP concepts, but could benefit from some additional guidance and support to help them fully grasp the material.
Would you like to try this lesson again in a different way? (yes/no): no
Lesson completed. You can come back anytime to ask more questions.
Lesson completed: **Methodology**\n\n* The authors propose a new approach called "Toolformer" that uses a combination of transformer-based models and attention mechanisms to generate prompts.\n* Toolformer involves training a model to predict the most effective prompt sequence for a given task, based on the model\'s understanding of the task and its requirements.\n\n
Lesson '**Methodology**\n\n* The authors propose a new approach called "Toolformer" that uses a combination of transformer-based models and attention mechanisms to generate prompts.\n* Toolformer involves training a model to predict the most effective prompt sequence for a given task, based on the model\'s understanding of the task and its requirements.\n\n' status updated to 'completed'.
```



## License

This project is licensed under the MIT License.
