# Natural Language Commanding via Program Synthesis

This repository contains a Python implementation of the framework described in the article "Natural Language Commanding via Program Synthesis". The package allows you to implement your own natural language commanding application leveraging LLM and DSL.

## Usage

### Docker Setup

To use this package, you can follow these steps to set up the required environment using Docker:

1. Pull the `qdrant/qdrant` Docker image:
   ```
   docker pull qdrant/qdrant
   ```

2. Run the Docker container with port mapping and volume mounting:
   ```bash
   docker run -p 6333:6333 \
       -v $(pwd)/qdrant_storage:/qdrant/storage \
       qdrant/qdrant
   ```

### Python Setup

If you want to work with the Python package directly, follow these steps:

1. Set up your Python environment to use version 3.10.2 using `pyenv`:
   ```bash
   pyenv local 3.10.2
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

### Test and Usage

You can find usage examples and tests in the `tests/test_nlcps.py` file.

To run the tests, make sure you have the necessary environment variables set:

```bash
OPENAI_API_KEY='your-api-key' OPENAI_BASE='https://api.openai.com/v1' pytest -s tests/
```

Replace `'your-api-key'` with your actual OpenAI API key.