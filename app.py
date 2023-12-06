import os
import streamlit as st
from trulens_eval import TruCustomApp, Feedback, Select
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI
from trulens_eval.tru_custom_app import instrument
import numpy as np
from openai import OpenAI

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
oai_client = OpenAI()

# Assuming the rest of your code remains the same...

# Wrap the custom RAG with TruCustomApp
rag = RAG_from_scratch()
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

