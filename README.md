# PolicyPal

**PolicyPal** is a personal side project designed to explore the fundamentals of **Retrieval-Augmented Generation (RAG)**. This project demonstrates how an AI agent can intelligently answer queries regarding company policies using a custom RAG framework.

The AI agent in PolicyPal is capable of reading PDF-based company policies and FAQs, processing them with **RAG**, and returning contextually relevant answers. This project was primarily implemented as a learning exercise to gain hands-on experience with RAG frameworks and AI-driven knowledge retrieval.

---

## Features

* **RAG-Powered AI Agent**: Answers questions about company policies using the Retrieval-Augmented Generation approach.
* **Custom RAG Framework**: Built entirely from scratch, published as **[RetrievalMind](https://github.com/Himanshu7921/RetrievalMind)** on PyPI.

  ```bash
  pip install RetrievalMind==0.1.1
  ```

  Anyone can use RetrievalMind locally to build their own RAG-powered applications.
* **Supports PDFs as Knowledge Source**: Easily handles company FAQs and policy documents.
* **Lightweight Web Frontend**: Interact with the AI agent via a simple HTML interface.

---

## Project Structure

```
PolicyPal/
├── __pycache__/
│   ├── main.cpython-313.pyc
│   └── server.cpython-313.pyc
├── data/
│   ├── faqs/
│   │   ├── Finance_faqs.pdf
│   │   ├── HR_faqs.pdf
│   │   └── IT_faqs.pdf
│   ├── policies/
│   │   ├── Finance_policies.pdf
│   │   ├── HR_policies.pdf
│   │   └── IT_policies.pdf
│   └── policy_pal_vector_store/
│       ├── 75ec9019-9e83-4659-9202-9ef33f547c73/
│       │   ├── data_level0.bin
│       │   ├── header.bin
│       │   ├── length.bin
│       │   └── link_lists.bin
│       └── chroma.sqlite3
├── index.html
├── main.py
├── serve_frontend.py
└── server.py
```

**Folder Details:**

* **data/faqs**: Contains PDF files with frequently asked questions.
* **data/policies**: Contains PDF files with company policies.
* **data/policy_pal_vector_store**: Stores vectorized embeddings of documents for RAG retrieval.
* **main.py**: Entry point for the RAG-based AI agent.
* **server.py**: Backend server handling AI queries.
* **serve_frontend.py**: Serves the HTML frontend for interaction.
* **index.html**: Web interface to chat with PolicyPal.

---

## Usage

1. **Install RetrievalMind (custom RAG framework):**

   ```bash
   pip install RetrievalMind==0.1.1
   ```

2. **Clone PolicyPal repository:**

   ```bash
   git clone https://github.com/Himanshu7921/PolicyPal-RAG-Agent
   cd PolicyPal
   ```

3. **Run the backend server:**

   ```bash
   python server.py
   ```

4. **Serve the frontend (optional if using index.html locally):**

   ```bash
   python serve_frontend.py
   ```

5. **Open `index.html`** in your browser and start querying company policies.

---

## About RetrievalMind

**RetrievalMind** is my custom RAG framework designed to simplify the process of creating RAG-powered AI agents. It allows you to:

* Ingest documents (PDFs, text files)
* Generate embeddings
* Store embeddings in a vector store
* Retrieve context and generate AI responses

GitHub: [https://github.com/Himanshu7921/RetrievalMind](https://github.com/Himanshu7921/RetrievalMind)

PyPI:

```bash
pip install RetrievalMind==0.1.1
```

This framework powers **PolicyPal**, enabling it to retrieve and answer queries from company policy documents efficiently.

---

## Future Enhancements

* Multi-user support via web interface
* Integration with additional file formats like DOCX
* Advanced ranking and filtering for more precise answers
* Deploy as a cloud-based API for enterprise usage

---

## License

This project is **for personal learning and demonstration purposes**. For details about the RetrievalMind framework license, check its [GitHub repository](https://github.com/Himanshu7921/RetrievalMind).

---