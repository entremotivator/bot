import os
import streamlit as st
from trulens_eval import TruCustomApp, Feedback, Select
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI
from trulens_eval.tru_custom_app import instrument
import numpy as np
from openai import OpenAI  # Import OpenAI directly

# Set up TruLens feedback functions
fopenai = fOpenAI()
grounded = Groundedness(groundedness_provider=fopenai)  # Use the correct variable name

f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

f_qa_relevance = (
    Feedback(fopenai.relevance_with_cot_reasons, name="Answer Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on_output()
)

f_context_relevance = (
    Feedback(fopenai.qs_relevance_with_cot_reasons, name="Context Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on(Select.RecordCalls.retrieve.rets.collect())
    .aggregate(np.mean)
)

# Create RAG model and set up TruCustomApp
os.environ["OPENAI_API_KEY"] = "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"
oai_client = OpenAI()

university_info = """The University of Washington, founded in 1861 in Seattle, is a public research university
with over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.
As the flagship institution of the six public universities in Washington state,
UW encompasses over 500 buildings and 20 million square feet of space,
including one of the largest library systems in the world."""

oai_client.embeddings.create(
    model="text-embedding-ada-002",
    input=university_info
)

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

embedding_function = OpenAIEmbeddingFunction(api_key=os.environ["OPENAI_API_KEY"],
                                             model_name="text-embedding-ada-002")

chroma_client = chromadb.PersistentClient(path="./chromadb")
vector_store = chroma_client.get_or_create_collection(name="Universities",
                                                      embedding_function=embedding_function)

vector_store.add("uni_info", documents=university_info)

class RAG_from_scratch:
    @instrument
    def retrieve(self, query: str) -> list:
        results = vector_store.query(query_texts=query, n_results=2)
        return results['documents'][0]

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        completion = oai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content":
                    f"We have provided context information below. \n"
                    f"---------------------\n"
                    f"{context_str}"
                    f"\n---------------------\n"
                    f"Given this information, please answer the question: {query}"
                }
            ]
        ).choices[0].message.content
        return completion

    @instrument
    def query(self, query: str) -> str:
        context_str = self.retrieve(query)
        completion = self.generate_completion(query, context_str)
        return completion

rag = RAG_from_scratch()

# Wrap the custom RAG with TruCustomApp
tru_rag = TruCustomApp(rag, app_id='RAG v1', feedbacks=[f_groundedness, f_qa_relevance, f_context_relevance])

# Create Streamlit app
st.title("RAG with TruLens and Palm 2 Chatbot")

# Define the Streamlit app function
def main():
    # User input query
    query = st.text_input("Ask a question:")

    # Check if a question is asked
    if query:
        # Query the RAG model
        with tru_rag as recording:
            response = rag.query(query)

        # Display the response
        st.subheader("Answer:")
        st.write(response)

# Run the Streamlit app
if __name__ == "__main__":
    main()
