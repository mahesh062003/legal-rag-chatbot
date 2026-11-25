# Legal Document RAG Chatbot

A retrieval-augmented chatbot that answers questions about legal documents,
combining dense and sparse retrieval and streaming its responses over
websockets.

Legal research is a retrieval problem before it is a generation problem. An
answer citing the wrong provision is worse than no answer, so this favours
retrieval precision and always surfaces the source documents behind a response.

---

## Retrieval

`hybrid_search` queries a FAISS vector index and a BM25 index, fuses the
results, and returns the top-ranked chunks.

Legal text needs both halves. Dense retrieval catches paraphrase — a question
about "tenant eviction" should reach text about "lessee dispossession" — while
BM25 catches the exact section numbers and defined terms that embeddings blur.
Either alone leaves a class of queries unanswerable.

Documents are chunked, embedded with a sentence-transformers model, and indexed
into both stores. Indexing is **incremental and hash-based**: re-running over a
document set that has grown processes only what is new. Files that fail to parse
are recorded with their reason rather than silently skipped, so they can be
revisited without rescanning everything.

### Query classification

Not every message deserves a retrieval pass. `QueryClassifier` routes on
`is_casual_query` and `has_legal_context`, so greetings and small talk get a
direct reply while substantive questions go to the index. This keeps
conversational latency low and avoids answering "hello" with a statute extract.

When retrieval surfaces nothing relevant, the response says so rather than
generating an unsupported answer.

---

## API

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Chat interface |
| `/api/chat` | POST | Submit a query, stream the response |
| `/api/chats/<user_id>` | GET | List a user's conversations |
| `/api/chat/<user_id>/<chat_id>` | GET | Fetch one conversation |
| `/api/new-chat/<user_id>` | POST | Start a conversation |
| `/api/delete-chat/<user_id>/<chat_id>` | DELETE | Delete a conversation |
| `/api/clear-all/<user_id>` | DELETE | Clear a user's history |
| `/api/documents` | GET | Index statistics |
| `/api/feedback` | POST | Record response feedback |
| `/api/health` | GET | Health check |

Websocket `connect`/`disconnect` events drive response streaming, so answers
render progressively instead of after a long pause. Requests are rate-limited
per IP address.

---

## Usage

```bash
pip install -r requirements.txt

# 1. Build the index over a document directory (GPU recommended)
python build_index.py

# 2. Serve
python app.py
```

`build_index.py` handles PDF, HTML, XML and plain text. It writes an index
directory containing the FAISS index, chunk metadata, the BM25 index, keyword
mappings, a record of the build parameters, and a log of any files that failed
to parse.

Neither the index nor any document set is included in this repository — a built
index for a corpus of real size runs to several gigabytes, well beyond what
belongs in Git. Point `build_index.py` at your own documents to generate one.

## Stack

Python · Flask · Flask-SocketIO · LangChain · sentence-transformers · FAISS ·
rank_bm25 · PyMuPDF · BeautifulSoup · lxml · spaCy · PyTorch

---

## Use

© 2026 Mahesh Veeraiah R. All rights reserved.

Published for review and assessment. Not licensed for reuse, redistribution or
modification — please get in touch about any other use.
