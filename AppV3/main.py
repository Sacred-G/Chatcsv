import streamlit as st
import pandas as pd
from pdf2image import convert_from_path
from dotenv import load_dotenv
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType
from html_templates import css, user_template, bot_template
from tempfile import NamedTemporaryFile
import os
def main():
    st.set_page_config(page_title="Pandas Agent")
    logo_url = "https://i.imgur.com/9DLn81j.png"
    st.image(logo_url, width=400)
    st.subheader("MTI Pandas Agent")
    st.write("Upload a CSV, XLSX, or PDF file and query answers from your data.")

    # Apply CSS
    st.write(css, unsafe_allow_html=True)

    # Define chat history session state variable
    st.session_state.setdefault('chat_history', [])

    # Temperature slider
    with st.sidebar:
        with st.expander("Settings",  expanded=True):
            TEMP = st.slider(label="LLM Temperature", min_value=0.0, max_value=1.0, value=0.3)
            st.markdown("Adjust the LLM Temperature: A higher value makes the output more random, while a lower value makes it more deterministic.")
            st.markdown("NOTE: Anything above 0.7 may produce hallucinations")
            st.markdown("### If unable to upload document or getting an error. You will need to set an openai environment variable")

    # Upload File
    file = st.file_uploader("Upload CSV, XLSX, or PDF file", type=["csv", "xlsx", "pdf"])
    
    temp_pdf_path = None  # Define the variable outside the conditional block            

    if file:
        if file.name.endswith('.pdf'):
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(file.getvalue())
            temp_pdf_path = temp_pdf.name
    text = ""
    if file:
        if file.name.endswith('.csv'):
            data = pd.read_csv(file)
            st.write("Data Preview:")
            st.dataframe(data.head(100))
        elif file.name.endswith('.xlsx'):
            data = pd.read_excel(file)
            st.write("Data Preview:")
            st.dataframe(data.head(100))
        elif file.name.endswith('.pdf'):
            # Save uploaded PDF to a temporary file
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(file.getvalue())
                temp_pdf_path = temp_pdf.name

            # Convert PDF pages to images
            images = convert_from_path(temp_pdf_path)

            # Extract text from all pages of the PDF
            with pdfplumber.open(temp_pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text()
            # Add pagination controls and display the selected page
            page_number = st.slider('Select Page:', min_value=1, max_value=len(images), value=1) - 1
            st.image(images[page_number])
            st.write("PDF processed. Ask your questions below.")

    # Define large language model (LLM)
    llm = OpenAI(temperature=TEMP)

    # Define pandas df agent
    if 'data' in locals():
        agent = create_pandas_dataframe_agent(llm, data, verbose=True)

    # Accept input from user
    query = st.text_input("Enter a query:")

    # Execute Button Logic
    if st.button("Execute") and query:
        with st.spinner('Generating response...'):
            try:
                # Define prompt for agent
                prompt = f'''
                    Consider the uploaded document, respond intelligently to user input.
                    \nDOCUMENT CONTENT: {text if file.name.endswith('.pdf') else 'CSV/XLSX data'}
                    \nCHAT HISTORY: {st.session_state.chat_history}
                    \nUSER INPUT: {query}
                    \nAI RESPONSE HERE:
                '''

                # Get answer from agent
                answer = agent.run(prompt)

                # Store conversation
                st.session_state.chat_history.append(f"USER: {query}")
                st.session_state.chat_history.append(f"AI: {answer}")

                # Display conversation in reverse order
                for i, message in enumerate(reversed(st.session_state.chat_history)):
                    if i % 2 == 0: st.markdown(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)
                    else: st.markdown(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)

            # Error Handling
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                
            if temp_pdf_path:  # Check if the variable has been defined
                os.remove(temp_pdf_path)
            
    st.markdown("Created By: Steven Bouldin")
    

if __name__ == "__main__":
    load_dotenv() # Import environmental variables
    main()
