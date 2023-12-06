import os
import streamlit as st
import google.generativeai as palm
import openai
import sqlite3
from trulens_eval import Tru
from trulens_eval.tru_custom_app import instrument
from openai import OpenAI
from trulens_eval import TruCustomApp, Feedback, Select
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.endpoint.openai import OpenAIEndpoint
from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI
from trulens_eval.tru_custom_app import instrument
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import numpy as np

# Sample data for GPT report card
GPT_reportcard = """The GPT Report Card is a comprehensive evaluation tool designed to assess the performance 
of various language models and artificial intelligence agents. It provides valuable insights into the 
capabilities of these models across different domains and tasks. This report card aims to facilitate 
testing, analysis, and comparison of AI language models, enabling users to make informed decisions 
about their applications and implementations."""

# Print the GPT report card information
print(GPT_reportcard)

# Set up TruLens
tru = Tru()

# Define the RAG class
class RAG_from_scratch:
    @instrument
    def retrieve(self, query: str) -> list:
        """
        Retrieve relevant text from vector store.
        """
        results = vector_store.query(
            query_texts=query,
            n_results=2
        )
        return results['documents'][0]

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        """
        Generate answer from context.
        """
        completion = oai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "user",
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


# Set up TruLens feedback functions
fopenai = fOpenAI()
grounded = Groundedness(groundedness_provider=fopenai)

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
# Pass the API key directly
# Assuming the rest of your code remains the same...

# Question/answer relevance between overall question and answer.
f_qa_relevance = (
    Feedback(fopenai.relevance_with_cot_reasons, name="Answer Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on_output()
)

# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(fopenai.qs_relevance_with_cot_reasons, name="Context Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on(Select.RecordCalls.retrieve.rets.collect())
    .aggregate(np.mean)
)


# Wrap the custom RAG with TruCustomApp
# Construct the app
tru_rag = TruCustomApp(rag,
                      app_id='RAG v1',
                      feedbacks=[f_groundedness, f_qa_relevance, f_context_relevance])

# Run the app
with tru_rag as recording:
    rag.query("When was the University of Washington founded?")

# Display leaderboard
st.write(tru.get_leaderboard(app_ids=["RAG v1"]))


# Define the Streamlit app function
def main():
    # User input query
    query = st.text_input("Ask a question:")

    # Check if a question is asked
    if query:
        # Query the RAG model
        with tru_rag as recording:
            response = rag.query(query)

        # Use VertexAI Text Generation Model
        vertexai_parameters = {
            "candidate_count": 1,
            "max_output_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40
        }
        vertexai_model = TextGenerationModel.from_pretrained("text-bison")
        vertexai_response = vertexai_model.predict(response, **vertexai_parameters)

        # Display the response
        st.subheader("Answer from VertexAI Text Generation:")
        st.write(vertexai_response.text)


tru.run_dashboard()

# Run the Streamlit app
    if __name__ == "__main__":
    main()

