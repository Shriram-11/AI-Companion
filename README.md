# AI-Companion

## Introduction

AI-Companion is an intelligent assistant designed to seamlessly integrate external documents and text with the vast knowledge of a large language model (LLM). This project allows users to input their own knowledge base—whether through documents, texts, or other forms of data—and combine it with the assistant’s existing knowledge to provide accurate, insightful, and context-aware guidance. The assistant can help users analyze information, find solutions, clarify concepts, and offer suggestions based on both the uploaded knowledge and the LLM’s understanding. Essentially, it acts as a powerful, hybrid tool for decision-making, problem-solving, and information discovery, capable of drawing on a personalized knowledge base while also tapping into the vast general knowledge of an AI system.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- FastAPI
- PyPDF2
- LangChain
- Google Generative AI
- ChromaDB
- Uvicorn
- dotenv

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/AI-Companion.git
    cd AI-Companion
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Create a `.env` file in the root directory.
    - Add your Google API key:
        ```
        GOOGLE_API_KEY=your_google_api_key
        ```

### Running the Application

1. Start the FastAPI server:
    ```sh
    uvicorn app:app --host 0.0.0.0 --port 5000 --reload
    ```

2. Access the API documentation at `http://localhost:5000/docs`.

### Running Tests

1. Run the tests using pytest:
    ```sh
    pytest
    ```

## Usage

- **Health Check**: Check if the API is running successfully by sending a GET request to `/status`.
- **Upload Files**: Upload files (PDF or TXT) to be processed and stored by sending a POST request to `/upload-files`.
- **Ask a Question**: Send a user prompt to the LLM by sending a POST request to `/ask`.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
