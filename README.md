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

```sh
(myenv) sireeshan4@penguin:~/vscode$ python rag_app.py 
Attempting to load vector store from SQLite...
Vector store loaded from ./DataStore/vectorstore.db.
Initializing RAG application...
Fetching last completed lesson...
Last completed lesson: None
Fetching next lesson...
Next lesson: **Introduction**\n\n* The paper discusses the importance of prompt engineering in natural language processing (NLP) tasks.\n* Prompt engineering involves designing input sequences that elicit specific responses from language models.\n\n
Running tutor to generate lesson contents...
Lesson Explanation: Hello student,

    Welcome to our lesson on **Introduction** to Natural Language Processing (NLP) and prompt engineering. Today, we're going to explore the importance of prompt engineering in NLP tasks.

    **What is Prompt Engineering?**

    Prompt engineering is a crucial aspect of natural language processing that involves designing input sequences to elicit specific responses from language models. In other words, it's about creating prompts that guide language models towards generating desired outputs.

    **Why is Prompt Engineering Important?**

    Prompt engineering plays a vital role in various NLP tasks, including:

    1. **Language Translation**: Prompt engineering helps translate text from one language to another.
    2. **Question Answering**: It enables machines to answer questions by providing relevant responses.
    3. **Text Summarization**: Prompt engineering facilitates the process of summarizing long pieces of text into concise summaries.

    **Key Concepts**

    To understand prompt engineering, let's break down some key concepts:

    1. **In-Context Prompting**: This is a type of prompt engineering where the input sequence is designed to be context-dependent.
    2. **Controllable Text Generation**: This refers to generating text that can be controlled or manipulated by the user.
    3. **Model Steerability**: Prompt engineering aims to steer language models towards desired outcomes without updating their weights.

    **Real-World Applications**

    Prompt engineering has numerous applications in various industries, including:

    1. **Virtual Assistants**: Prompt engineering is used to create voice assistants that can understand and respond to user queries.
    2. **Chatbots**: It's essential for chatbots to generate responses that are relevant and engaging.
    3. **Content Generation**: Prompt engineering helps generate high-quality content, such as articles, blog posts, or social media posts.

    **Resources**

    To learn more about prompt engineering, I've included some useful resources:

    1. **OpenAI Cookbook**: A comprehensive collection of examples for efficient language model usage.
    2. **LangChain**: A library that combines language models with other components to build applications.
    3. **PromptPerfect**: A guide repository containing education materials on prompt engineering.

    **Conclusion**

    In conclusion, prompt engineering is a critical aspect of NLP tasks that involves designing input sequences to elicit specific responses from language models. By understanding the importance and key concepts of prompt engineering, you'll be better equipped to tackle various NLP challenges and applications in the future.

Do you have any follow-up questions about this lesson? (yes/no): no

If you have any more questions later, feel free to ask!
Generating test questions for you...

Question: 1. What is prompt engineering, and what is its primary goal?
Your answer to: 1. What is prompt engineering, and what is its primary goal? (type 'skip' to move on): prompt engineering is nothing
Your answer: prompt engineering is nothing
Correct answer: Prompt engineering refers to methods for communicating with large language models (LLMs) to steer their behavior for desired outcomes without updating the model weights. Its primary goal is to align and make the model more steerable, allowing it to generate responses that meet specific objectives or requirements.
Your grade for this answer: I'd evaluate this student's answer as follows:

Clarity: 2/10
The statement lacks clarity and conveys a negative attitude towards the subject of prompt engineering. The sentence structure is simple and straightforward, but it doesn't provide any meaningful insight or explanation.

Accuracy: 4/10
While the student mentions that "prompt engineering is nothing," they don't provide any evidence to support this claim. In fact, prompt engineering is a real field with applications in various industries, such as graphic design, video production, and web development.

Completeness: 3/10
The answer doesn't cover all aspects of prompt engineering. It doesn't mention the different types of prompts, their characteristics, or how they are used in practice. The student also fails to provide any examples or explanations to support their claim.

Overall, this answer demonstrates a poor understanding of the subject matter and lacks clarity, accuracy, and completeness. A more effective response would demonstrate a thorough understanding of prompt engineering and provide meaningful insights into its applications and importance.
```

## Excluding `src` Folder

The application is configured to exclude the `src` folder when creating the vector store.

## License

This project is licensed under the MIT License.
