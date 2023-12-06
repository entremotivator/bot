import streamlit as st
from trulens_eval import TruCustomApp

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

# Wrap the custom RAG with TruCustomApp
tru_rag = TruCustomApp(rag, app_id='RAG v1', feedbacks=[f_groundedness, f_qa_relevance, f_context_relevance])

# Run the Streamlit app
if __name__ == "__main__":
    main()
