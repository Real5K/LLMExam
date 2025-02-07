# LLM Exam AI

This project consists of two main parts: a Streamlit web application for generating exam questions from PDF documents and a set of scripts for extracting questions from PDFs and storing them in a Weaviate database.

## Table of Contents
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Conda Environment Setup](#conda-environment-setup)
- [Web Application](#web-application)
  - [Running the Web Application](#running-the-web-application)
  - [Using the Web Application](#using-the-web-application)
- [PDF to Database](#pdf-to-database)
  - [Running the PDF to Database Script](#running-the-pdf-to-database-script)

## Setup

### Prerequisites
- Python 3.10 or higher
- Conda
- Weaviate Cloud account (for PDF to Database)

### Conda Environment Setup

1.  Create a new Conda environment based on the dependencies:
    ```
    conda env create -f environment.yml
    ```

## Web Application

The web application allows users to upload PDF files and generate exam questions based on the content of the documents.

### Running the Web Application
1.  Navigate to the directory containing `webApp.py`.
2.  Run the Streamlit application:
    ```
    streamlit run webApp.py
    ```
3.  The application will open in your web browser.

### Using the Web Application
1.  Upload one or more PDF files using the file uploader in the left panel.
2.  Click the "Generate Questions" button in the right panel to generate exam questions.
3.  The generated questions will be displayed in the text area in the right panel.

## PDF to Database

The `pdf_to_db.ipynb` script extracts questions from PDF documents and stores them in a Weaviate database.

### Running the PDF to Database Script

1.  Open the `pdf_to_db.ipynb` notebook in Jupyter.
2.  Ensure that the required environment variables (`WEAVIATE_URL`, `WEAVIATE_API_KEY`, and `Mistral API key`) are set in the notebook or the environment.
3.  Run all cells in the notebook.