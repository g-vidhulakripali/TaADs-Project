# Retrieval-Augmented Generation (RAG) Project

This README provides a step-by-step guide to set up and run the RAG project, including installing the Ollama library, downloading the Mistral model, and running the application.

---

## Prerequisites

### System Requirements
- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.8 or higher
- **Memory**: At least 8GB RAM
- **Disk Space**: Sufficient space to store models and project dependencies

### Tools and Dependencies
- [Ollama](https://ollama.com/) library for running LLMs.
- [Mistral model](https://ollama.com/library) for Retrieval-Augmented Generation.
- Python libraries listed in `requirements.txt`.

---

## Installation Steps

### 1. Clone the Repository
```bash
# Clone this repository
$ git clone <repository-url>
$ cd <repository-folder>
```

### 2. Install Dependencies

Install Python dependencies listed in `requirements.txt`:
```bash
# Create and activate a virtual environment (optional but recommended)
$ python3 -m venv venv
$ source venv/bin/activate   # On Windows, use `venv\Scripts\activate`

# Install dependencies
$ pip install -r requirements.txt
```

### 3. Install Ollama

Follow the instructions below to install the Ollama CLI tool:

#### macOS / Linux
1. Download the Ollama CLI tool from [https://ollama.com/](https://ollama.com/).
2. Run the installer and add the binary to your PATH (if not done automatically).

#### Windows
1. Download the appropriate installer from [https://ollama.com/](https://ollama.com/).
2. Follow the setup wizard and ensure the tool is accessible in your command prompt.

Verify the installation:
```bash
$ ollama version
```

### 4. Download the Mistral Model

Download and install the Mistral model using the Ollama CLI:
```bash
$ ollama pull mistral
```

This command will download and set up the Mistral model for local usage.

---

## Project Structure

- **`src/`**: Contains the main source code.
- **`data/`**: Stores any dataset files required for retrieval.
- **`requirements.txt`**: Python dependencies.
- **`README.md`**: Documentation for the project.

---

## Running the Project

### 1. Start the Quart Server

Navigate to the project directory and run the server:
```bash
$ python src/server.py
```

This will start the Quart server, typically at `http://0.0.0.0:3000`.

### 2. Test the Retrieval-Augmented Generation

1. Open your web browser and go to `http://0.0.0.0:3000`.
2. Interact with the chatbot interface to see the retrieval-augmented responses powered by the Mistral model.

---

## Troubleshooting

### Common Issues

1. **Ollama CLI Not Found**:
   Ensure the Ollama binary is added to your system PATH.
   ```bash
   $ export PATH=$PATH:/path/to/ollama
   ```

2. **Model Download Fails**:
   Verify your internet connection and retry the `ollama pull mistral` command.

3. **Server Errors**:
   Check the logs for errors when running `server.py`. Ensure all dependencies are installed.

---

## Additional Resources

- [Ollama Documentation](https://ollama.com/docs)
- [Mistral Model Info](https://ollama.com/library)
- [Quart Framework Documentation](https://pgjones.gitlab.io/quart/)

---

## Contributing

Feel free to fork the repository, make improvements, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.


