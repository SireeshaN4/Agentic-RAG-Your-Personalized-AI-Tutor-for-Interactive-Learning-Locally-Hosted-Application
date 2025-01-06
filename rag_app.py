import os
import pickle
import sqlite3
import requests
import numpy as np
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain.embeddings.base import Embeddings
import json
import re


# source ~/myenv/bin/activate

# Set environment variable
os.environ["USER_AGENT"] = "MyApp/1.0"

# Custom embedding class
class LocalHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts)
    
    def embed_query(self, text):
        return self.model.encode(text)

# SQLite database functions
def create_database():
    conn = sqlite3.connect('./DataStore/training_plan.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS training_plan (
                    id INTEGER PRIMARY KEY,
                    lesson TEXT,
                    date TEXT,
                    status TEXT
                )''')
    
    
    conn.commit()
    conn.close()



def get_last_completed_lesson():
    conn = sqlite3.connect('./DataStore/training_plan.db')
    c = conn.cursor()
    
    # Fetch the last completed lesson (based on the highest lesson_id and completed status)
    c.execute('''SELECT training_plan.lesson 
                FROM training_plan
                WHERE training_plan.id IN (
                    SELECT MAX(id) 
                    FROM training_plan
                    GROUP BY lesson
                )
                AND status <> 'N/A'
                ORDER BY training_plan.id DESC
                    ''')
    
    last_lesson = c.fetchone()
    conn.close()

    
    if last_lesson:
        return last_lesson[0]
    else:
        # If no lesson has been completed, return None
        return None

def get_next_lesson(last_lesson):
    conn = sqlite3.connect('./DataStore/training_plan.db')
    c = conn.cursor()
    
    if last_lesson is None:
        # If no lesson has been completed, start from the first lesson
        c.execute('SELECT lesson FROM training_plan ORDER BY id ASC LIMIT 1')
    else:
        # Fetch the next lesson based on the current last completed lesson
        c.execute('''SELECT lesson FROM training_plan
                     WHERE id > (SELECT id FROM training_plan WHERE lesson = ?)
                     ORDER BY id ASC
                     LIMIT 1''', (last_lesson,))
    
    next_lesson = c.fetchone()
    conn.close()
    
    # Return the first lesson if no next lesson is found
    return next_lesson[0] if next_lesson else None

def get_training_plan():
    conn = sqlite3.connect('./DataStore/training_plan.db')
    c = conn.cursor()
    c.execute("SELECT * FROM training_plan")
    plan = c.fetchall()
    conn.close()
    return plan

def update_progress(lesson_id, completed=True, grade=None):
    conn = sqlite3.connect('./DataStore/training_plan.db')
    c = conn.cursor()
    if grade is not None:
        c.execute("INSERT OR REPLACE INTO progress (lesson_id, completed, grade) VALUES (?, ?, ?)",
                  (lesson_id, 1 if completed else 0, grade))
    else:
        c.execute("INSERT OR REPLACE INTO progress (lesson_id, completed) VALUES (?, ?)",
                  (lesson_id, 1 if completed else 0))
    conn.commit()
    conn.close()

def adjust_training_plan(lesson, grade):
    llm = ChatOllama(model="llama3.2:1b", temperature=0)
    prompt = PromptTemplate(
        template="""You are a teaching assistant for a 10-year-old student.
        The student is currently learning the following lesson: {lesson}.
        The student received a grade of {grade}. Please adjust the training plan accordingly:
        - If the student is doing well, continue with the next lesson.
        - If the student is struggling, suggest revisiting this lesson or provide extra practice.
        Output the updated plan as a dictionary with lesson names as keys and dates as values.
        """,
        input_variables=["lesson", "grade"],
    )
    
    # Format the prompt string
    plan_message = llm.invoke(prompt.format(lesson=lesson, grade=grade))
    
    # Debugging: print the raw response to check its content
    print("Debugging - plan_message.content:", plan_message.content)

    try:
        # Extract the plan content from the message
        plan = eval(plan_message.content)  # Assuming the response is a string representation of a dictionary
    except Exception as e:
        print(f"Error parsing the plan: {e}")
        plan = {}

    return plan

def create_vectorstore():
    folder_path = "./PDFs"
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            docs.extend(loader.load())

    if not docs:
        print("No documents found. Ensure that your PDFs are located in the './PDFs' folder.")
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs)
    
    embeddings = LocalHuggingFaceEmbeddings()
    vectorstore = SKLearnVectorStore.from_texts(
        texts=[doc.page_content for doc in doc_splits],
        embedding=embeddings,
        metadatas=[doc.metadata for doc in doc_splits]
    )
    
    print(f"Vectorstore created with {len(doc_splits)} document chunks.")
    return vectorstore.as_retriever(k=4)

def save_vectorstore_to_file(retriever, filename="./DataStore/vectorstore.pkl"):
    try:
        # Extract documents, embeddings, and metadata from the retriever object
        documents = retriever.index  # These are the document contents
        embeddings = retriever.embeddings  # These are the embeddings
        metadatas = retriever.metadatas  # These are the metadata
        
        # Save to a pickle file
        with open(filename, 'wb') as f:
            pickle.dump((documents, embeddings, metadatas), f)
        
        print(f"Vector store saved to {filename}.")
    except Exception as e:
        print(f"Error saving vector store: {e}")



def load_vectorstore_from_file(filename="./DataStore/vectorstore.pkl"):
    if os.path.exists(filename):
        try:
            # Load the saved components
            with open(filename, 'rb') as f:
                documents, embeddings, metadatas = pickle.load(f)
            
            # Rebuild the vector store with the saved components
            retriever = SKLearnVectorStore.from_texts(
                texts=[doc.page_content for doc in documents],
                embedding=embeddings,
                metadatas=metadatas
            )
            
            print(f"Vector store loaded from {filename}.")
            return retriever
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None
    else:
        print(f"No saved vector store found. Creating a new one...", filename)
        return None


def create_and_save_vectorstore():
    # Create the vector store
    retriever = create_vectorstore()

    # Save the vector store to SQLite
    save_vectorstore_to_sqlite(retriever)

    return retriever
class RAGApplication:
    def __init__(self, retriever):
        self.retriever = retriever
        self.tutor_llm = ChatOllama(model="llama3.2:1b", temperature=0)
        self.reviewer_llm = ChatOllama(model="llama3.2:1b", temperature=0)
        self.plan_llm = ChatOllama(model="llama3.2:1b", temperature=0)
        
        # Define the prompt templates
        self.tutor_prompt = PromptTemplate(
            template="""You are a teaching assistant for a 10-year-old student.
            Use the following documents to answer the question.
            If you don't know the answer, just say that you don't know.
            Keep the answer concise:
            Question: {question}
            Documents: {documents}
            Answer:""",
            input_variables=["question", "documents"],
        )

        self.reviewer_prompt = PromptTemplate(
            template="""You are an assistant grading a student's answer. 
            Evaluate the student's answer based on how well they understood the lesson. 
            The student's answer is: 
            "{student_answer}".
            """,
            input_variables=["student_answer"],
        )

        # History of interactions between student and tutor
        self.interaction_history = []

    def run_tutor(self, lesson_title):
        plan = get_training_plan()
        if not plan:
            print("Training plan is empty. Creating a new training plan...")
            new_plan = self.create_training_plan()
            print("New training plan created and stored.")

        # Get the lesson content
        documents = self.get_lesson_content(lesson_title)
        if not documents:
            return "Sorry, I couldn't find any relevant documents for this lesson. Please try again later."

        # Generate lesson explanation
        tutor_response = self.get_lesson_explanation(lesson_title, documents)
        
        # Manage follow-up questions after explanation
        self.handle_follow_up_questions()

        # Test user's understanding if no follow-up questions
        self.test_user_understanding(documents)

        # After lesson completion, ask for suggestions on how to improve the lesson
        self.ask_for_retry_or_improvement()

        return tutor_response

    def get_lesson_content(self, lesson_title):
        """Fetch the lesson content from the retriever."""
        try:
            search_type = 'similarity'
            docs = self.retriever.search(lesson_title, search_type)
            if not docs:
                return None
            return "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return None

    def get_lesson_explanation(self, lesson_title, documents):
        """Generate an explanation of the lesson using the tutor model."""
        prompt_message = f"""
        Given the following lesson content, explain it to the student:

        Lesson Title: {lesson_title}

        Context:
        {documents}
        """
        try:
            tutor_answer = self.tutor_llm.invoke(prompt_message)
            tutor_response = tutor_answer.content
            self.interaction_history.append({
                "question": "Explain the lesson.",
                "answer": tutor_response
            })
            print(f"Lesson Explanation: {tutor_response}")
            return tutor_response
        except Exception as e:
            print(f"Error generating tutor explanation: {e}")
            return None

    def handle_follow_up_questions(self):
        """Handle the follow-up questions from the user."""
        follow_up_question = input("Do you have any follow-up questions about this lesson? (yes/no): ").strip().lower()

        if follow_up_question == "yes":
            while True:
                user_question = input("Ask your question or type exit: ").strip()
                if user_question.lower() == "exit":
                    print("Goodbye!")
                    break

                # Include the previous context and new user question
                follow_up_prompt = f"""
                Previous context:
                {self.interaction_history[-1]["answer"]}

                New question: {user_question}

                Answer the new question based on the previous context and lesson.
                """

                # Call the tutor model with the updated prompt
                try:
                    tutor_follow_up_answer = self.tutor_llm.invoke(follow_up_prompt)
                    tutor_follow_up_response = tutor_follow_up_answer.content
                    print(f"Tutor's response: {tutor_follow_up_response}")
                    self.interaction_history.append({
                        "question": user_question,
                        "answer": tutor_follow_up_response
                    })
                except Exception as e:
                    print(f"Error generating follow-up answer: {e}")

    def ask_for_retry_or_improvement(self):
        """Ask the student if they want to retry the lesson in a different way."""
        feedback_prompt = f"""
        Based on the interaction history and the student's performance, evaluate the lesson.
        Ask the student if they would like to retry the lesson in a different way.
        Use the following history:
        {self.interaction_history}
        """
        try:
            reviewer_response = self.reviewer_llm.invoke(feedback_prompt)
            print(f"Reviewer's suggestion: {reviewer_response.content}")
            
            # If the student agrees to retry, provide feedback to Tutor LLM
            retry_decision = input("Would you like to try this lesson again in a different way? (yes/no): ").strip().lower()
            if retry_decision == "yes":
                self.modify_lesson_for_retry(reviewer_response.content)
                print("Retrying the lesson with modifications...")
                self.run_tutor(self.interaction_history[-1]["question"])  # Re-run tutor with suggestions
            else:
                print("Lesson completed. You can come back anytime to ask more questions.")

        except Exception as e:
            print(f"Error asking for retry: {e}")

    def modify_lesson_for_retry(self, suggestion):
        """Modify the lesson based on the reviewer’s suggestion."""
        # Send feedback to the Tutor LLM on how to modify the lesson
        print(f"Modifying lesson based on suggestion: {suggestion}")
        # You can pass the suggestion to the Tutor LLM to adjust lesson presentation

    def generate_dynamic_questions(self, lesson_title, lesson_content):
        """
        Generate dynamic questions based on the lesson content using the LLM.
        """
        prompt = f"""
        Based on the following lesson content, generate 3-5 single line questions that can be used to test the student's understanding of the material. If the question is multiple choice, provide options like a), b), c), etc. If the question is open-ended, just provide the question text.

        Lesson Title: {lesson_title}

        Context: 
        {lesson_content}
        """
        try:
            # Get the dynamic questions from the LLM
            question_response = self.tutor_llm.invoke(prompt)
            
            # Split the response based on new lines, and filter out empty lines or unwanted prefixes
            questions = question_response.content.split("\n")  # Assuming each question is separated by a new line
            questions = [q.strip() for q in questions if q.strip()]
            
            # If no valid questions are returned, set a fallback message
            if not questions:
                questions = ["Sorry, no questions generated. Please try again later."]
            
            return questions
        except Exception as e:
            print(f"Error generating questions: {e}")
            return ["Sorry, there was an issue generating questions."]

    def display_test_questions(self, questions):
        """
        Display the generated test questions to the user.
        """
        print("Generating test questions for you...\n")

        for idx, question in enumerate(questions, start=1):
            print(f"Question {idx}: {question}")
            
            # if 'a)' in question or 'b)' in question or 'c)' in question:
            #     # If the question has multiple-choice options
            #     print("Select one of the following options:")
            #     options = [line for line in question.split("\n") if line.strip().startswith(('a)', 'b)', 'c)'))]
            #     for option in options:
            #         print(option)
                
            #     while True:
            #         user_answer = input("Your answer (e.g., a, b, c, or type 'skip' to move on): ").strip().lower()

            #         if user_answer in ['a', 'b', 'c']:  # Valid options
            #             print(f"You selected option: {user_answer}")
            #             break  # Exit the loop after a valid choice
            #         elif user_answer == 'skip':  # Skip option
            #             print("Skipped question.\n")
            #             break
            #         else:  # Invalid answer, prompt again
            #             print("Invalid selection. Please choose a valid option (a, b, or c).")
            # else:
            #     # If it's a regular question (not multiple choice)
            #     user_answer = input(f"Your answer to: {question} (type 'skip' to move on): ").strip()

            #     if user_answer.lower() == "skip":
            #         print("Skipped question.\n")
            #     else:
            #         # Process the user's answer for grading or feedback
            #         print(f"Your answer: {user_answer}")
            
            print("\n")

    def generate_answer_for_question(self, lesson_title, lesson_content, question):
        """
        Generate the answer to a dynamic question based on the lesson content using the LLM.
        """
        prompt = f"""
        Based on the following lesson content, answer the following question:

        Lesson Title: {lesson_title}

        Context:
        {lesson_content}

        Question: {question}
        """
        try:
            # Get the answer from the LLM
            answer_response = self.tutor_llm.invoke(prompt)
            return answer_response.content.strip()
        except Exception as e:
            print(f"Error generating answer for the question: {e}")
            return "Sorry, I couldn't generate an answer at this time."

    def grade_student_answer(self, student_answer):
        """
        Grade the student's answer based on the reviewer LLM.
        """
        prompt_message = self.reviewer_prompt.format(student_answer=student_answer)
        try:
            
            grade = self.reviewer_llm.invoke(prompt_message)
            
            return grade.content
        except Exception as e:
            print(f"Error grading the student's answer: {e}")
            return "Error in grading."

    def handle_follow_up_questions(self):
        """Handle the follow-up questions from the user."""
        follow_up_question = input("Do you have any follow-up questions about this lesson? (yes/no): ").strip().lower()

        if follow_up_question == "yes":
            while True:
                user_question = input("Ask your question or type exit: ").strip()
                if user_question.lower() == "exit":
                    print("Goodbye!")
                    break

                # Include the previous context and new user question
                follow_up_prompt = f"""
                Previous context:
                {self.context_history}

                New question: {user_question}

                Answer the new question based on the previous context and lesson.
                """

                # Call the tutor model with the updated prompt
                try:
                    tutor_follow_up_answer = self.tutor_llm.invoke(follow_up_prompt)
                    tutor_follow_up_response = tutor_follow_up_answer.content
                    print(f"Tutor's response: {tutor_follow_up_response}")
                    
                    # Update the context for the next question
                    self.context_history += f"\nFollow-up Question: {user_question}\nAnswer: {tutor_follow_up_response}"

                except Exception as e:
                    print(f"Error generating follow-up answer: {e}")
        else:
            print("If you have any more questions later, feel free to ask!")

    def test_user_understanding(self, documents):
        """Test the user's understanding by generating dynamic questions and grading the answers."""
        print("Generating test questions for you...")
        dynamic_questions = self.generate_dynamic_questions("Lesson Title", documents)

        if dynamic_questions:
            for idx, question in enumerate(dynamic_questions, 1):
                print(f"\n {question}")
                user_answer = input(f"Your answer : (type 'skip' to move on): ").strip()
                if user_answer.lower() != "skip":
                    # print(f"Your answer: {user_answer}")
                    # Generate the correct answer for the question
                    grade = self.grade_student_answer(user_answer)
                    print(f"Your grade for this answer: {grade}")
                    correct_answer = self.generate_answer_for_question("Lesson Title", documents, question)
                    print(f"Correct answer: {correct_answer}")                  
                else:
                    print("Skipped question.")
                    # Grade the student's answer
                

        else:
            print("Sorry, I couldn't generate any test questions. Please try again later.")



def update_lesson_status_to_completed(lesson_name):
    conn = sqlite3.connect('./DataStore/training_plan.db')
    c = conn.cursor()
    
    # Update the status of the specified lesson to 'completed'
    c.execute('''UPDATE training_plan
                 SET status = 'completed'
                 WHERE lesson = ?''', (lesson_name,))
    
    conn.commit()  # Commit the changes to the database
    conn.close()
    
    print(f"Lesson '{lesson_name}' status updated to 'completed'.")

def save_vectorstore_to_sqlite(retriever, db_name="./DataStore/vectorstore.db"):
    try:
        conn = sqlite3.connect(db_name)
        c = conn.cursor()

        # Create the vectorstore table if it doesn't exist
        c.execute('''CREATE TABLE IF NOT EXISTS vectorstore (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document TEXT,
                        embedding BLOB,
                        metadata TEXT
                    )''')

        # Access the internal vectorstore (retriever's vectorstore)
        vectorstore = retriever.vectorstore

        # Extract documents, embeddings, and metadata from the vectorstore
        documents = vectorstore._texts  # Directly access the 'texts' attribute, which contains document content
        embeddings = vectorstore._embeddings  # Access the embeddings attribute
        metadatas = vectorstore._metadatas  # Access the metadata attribute

        # Insert documents, embeddings, and metadata into the SQLite table
        for doc, embedding, metadata in zip(documents, embeddings, metadatas):
            c.execute("INSERT INTO vectorstore (document, embedding, metadata) VALUES (?, ?, ?)",
                    (doc, sqlite3.Binary(embedding.tobytes()), str(metadata)))
        
        conn.commit()
        conn.close()
        print(f"Vector store saved to {db_name}.")
    except Exception as e:
        print(f"Error saving vector store: {e}")

# Embedding class from your previous code
# Embedding class from your previous code
class LocalHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts)
    
    def embed_query(self, text):
        return self.model.encode(text)

def load_vectorstore_from_sqlite(db_name="./DataStore/vectorstore.db"):
    try:
        conn = sqlite3.connect(db_name)
        c = conn.cursor()

        # Retrieve the stored documents, embeddings, and metadata
        c.execute("SELECT * FROM vectorstore")
        rows = c.fetchall()

        documents = [row[1] for row in rows]  # Extract document content from rows
        embeddings = [np.frombuffer(row[2], dtype=np.float32) for row in rows]  # Convert embedding back from bytes
        metadatas = [eval(row[3]) for row in rows]  # Convert metadata back from string

        conn.close()

        # Now we need to create the vectorstore with the correct format
        # First, let's convert the embeddings into the appropriate format for SKLearnVectorStore
        embeddings_matrix = np.vstack(embeddings)  # Stack embeddings into a 2D matrix

        # Create the embedding instance
        embedding_function = LocalHuggingFaceEmbeddings()

        # Create the vector store from the loaded documents, embeddings, and metadata
        retriever = SKLearnVectorStore.from_texts(
            texts=documents,  # The documents
            embedding=embedding_function,  # The embedding model
            metadatas=metadatas  # The metadata
        )

        print(f"Vector store loaded from {db_name}.")
        return retriever
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

def main():
    # Initialize database for training plan (already done in your code)
    create_database()

    # Try loading the vector store from SQLite
    print("Attempting to load vector store from SQLite...")
    retriever = load_vectorstore_from_sqlite("./DataStore/vectorstore.db")
    
    if retriever is None:
        # If no vector store was found, create and save a new one
        print("No vector store found. Creating and saving a new one...")
        retriever = create_and_save_vectorstore()

    # Initialize RAG application
    print("Initializing RAG application...")
    app = RAGApplication(retriever)

    # Fetch last completed lesson
    print("Fetching last completed lesson...")
    last_lesson = get_last_completed_lesson()
    print(f"Last completed lesson: {last_lesson}")

    # if last_lesson is None:
        # print("No completed lessons found. Initializing the training plan...")
        # If no lessons have been completed, create and store a new training plan
        # plan = app.create_training_plan()  # Generate a new plan using the reviewer LLM

    print("Fetching next lesson...")
    next_lesson = get_next_lesson(last_lesson)

    if next_lesson:
        print(f"Next lesson: {next_lesson}")
        # Assume student's answer is retrieved or generated
        print("Running tutor to generate lesson contents...")
        student_answer = app.run_tutor(next_lesson)
        print(f"Lesson completed: {next_lesson}")


        # You could update progress and adjust plan based on grade here
        update_lesson_status_to_completed(next_lesson)
    else:
        print("No more lessons to display.")

if __name__ == "__main__":
    main()

