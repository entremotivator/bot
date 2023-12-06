import os
import streamlit as st
import google.generativeai as palm
import openai
from openai import OpenAI
from trulens_eval import TruCustomApp, Feedback, Select
from trulens_eval.feedback import Groundedness
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

# Create RAG model and set up TruCustomApp
os.environ["OPENAI_API_KEY"] = "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"
oai_client = OpenAI(api_key="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")  # Pass the API key directly


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

# Assuming the rest of your code remains the same...

# Wrap the custom RAG with TruCustomApp
rag = RAG_from_scratch()
tru_rag = TruCustomApp(rag, app_id='RAG v1', feedbacks=[f_groundedness, f_qa_relevance, f_context_relevance])

# Create Streamlit app
st.title("RAG with TruLens and VertexAI Text Generation")

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

# Run the Streamlit app
if __name__ == "__main__":
    main()

