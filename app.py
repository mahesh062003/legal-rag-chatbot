import os
import json
import pickle
import logging
import time
import socket
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from threading import Lock

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from rank_bm25 import BM25Okapi
import uuid
import re
from collections import defaultdict, deque
import torch
import hashlib
import warnings

# Suppress unnecessary warnings and logs
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Suppress transformers logging
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set logging levels for third-party libraries
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)

# --- Configuration and Logging ---

class ColoredFormatter(logging.Formatter):
    """Colored formatter for terminal output"""
    
    COLORS = {
        'DEBUG': '\033[36m', # Cyan
        'INFO': '\033[32m', # Green
        'WARNING': '\033[33m', # Yellow
        'ERROR': '\033[31m', # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m' # Reset
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

logger = logging.getLogger('LegalChatbot')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = ColoredFormatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler('legal_chatbot.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "127.0.0.1"

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', str(uuid.uuid4()))
CORS(app)

socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='threading',
    logger=False, 
    engineio_logger=False 
)

INDEX_DIR = "legal_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_HISTORY_LENGTH = 10
RELEVANCE_THRESHOLD = 0.5 # Increased threshold

RATE_LIMIT_DURATION = 3600 
RATE_LIMIT_MAX_REQUESTS = 100
rate_limits = {}
rate_limit_lock = Lock()

# Enhanced legal keywords for better classification
LEGAL_KEYWORDS = {
    'certificate', 'tax', 'reserve', 'legal', 'court', 'statute', 'regulation',
    'provision', 'section', 'clause', 'article', 'amendment', 'act', 'law',
    'jurisdiction', 'procedure', 'filing', 'notice', 'compliance', 'liability',
    'judgment', 'appeal', 'petition', 'defendant', 'plaintiff', 'contract',
    'constitution', 'bill', 'ordinance', 'decree', 'rules', 'code', 'penal',
    'civil', 'criminal', 'evidence', 'witness', 'testimony', 'hearing',
    'tribunal', 'magistrate', 'judge', 'lawyer', 'attorney', 'advocate',
    'barrister', 'solicitor', 'legal aid', 'injunction', 'writ', 'summons',
    'subpoena', 'affidavit', 'deposition', 'discovery', 'litigation',
    'settlement', 'damages', 'compensation', 'fine', 'penalty', 'sentence',
    'probation', 'parole', 'custody', 'bail', 'bond', 'surety', 'guarantee',
    'indemnity', 'warranty', 'breach', 'violation', 'infringement',
    'intellectual property', 'copyright', 'patent', 'trademark', 'license',
    'permit', 'authorization', 'approval', 'consent', 'agreement',
    'memorandum', 'protocol', 'treaty', 'convention', 'charter', 'bylaw',
    'policy', 'directive', 'circular', 'notification', 'gazette',
    'property', 'land', 'title', 'deed', 'mortgage', 'lease', 'rent',
    'tenant', 'landlord', 'eviction', 'foreclosure', 'easement',
    'inheritance', 'will', 'testament', 'probate', 'estate', 'trust',
    'executor', 'administrator', 'beneficiary', 'heir', 'legacy',
    'marriage', 'divorce', 'separation', 'custody', 'alimony', 'maintenance',
    'adoption', 'guardianship', 'minor', 'juvenile', 'family court',
    'employment', 'labor', 'worker', 'employee', 'employer', 'wages',
    'salary', 'benefits', 'termination', 'dismissal', 'redundancy',
    'discrimination', 'harassment', 'safety', 'compensation',
    'company', 'corporation', 'partnership', 'sole proprietorship',
    'business', 'commercial', 'trade', 'commerce', 'finance', 'banking',
    'insurance', 'securities', 'investment', 'fraud', 'embezzlement',
    'tax evasion', 'money laundering', 'corruption', 'bribery'
}

# Casual conversation patterns
CASUAL_PATTERNS = {
    'greetings': ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening'],
    'wellbeing': ['how are you', 'how do you do', 'how is it going', 'what\'s up', 'how have you been'],
    'thanks': ['thank you', 'thanks', 'thank u', 'thx', 'much appreciated', 'grateful'],
    'goodbye': ['bye', 'goodbye', 'see you', 'take care', 'farewell', 'catch you later'],
    'casual_questions': ['who are you', 'what are you', 'what can you do', 'help me', 'what is this'],
    'affirmations': ['ok', 'okay', 'yes', 'no', 'sure', 'alright', 'fine', 'cool']
}

def check_rate_limit(ip_address):
    with rate_limit_lock:
        now = time.time()
        if ip_address not in rate_limits:
            rate_limits[ip_address] = {'count': 1, 'reset_time': now + RATE_LIMIT_DURATION}
            return True, RATE_LIMIT_MAX_REQUESTS - 1
        
        if now > rate_limits[ip_address]['reset_time']:
            rate_limits[ip_address] = {'count': 1, 'reset_time': now + RATE_LIMIT_DURATION}
            return True, RATE_LIMIT_MAX_REQUESTS - 1
        
        if rate_limits[ip_address]['count'] < RATE_LIMIT_MAX_REQUESTS:
            rate_limits[ip_address]['count'] += 1
            return True, RATE_LIMIT_MAX_REQUESTS - rate_limits[ip_address]['count']
        
        return False, 0

class QueryClassifier:
    """Classifies user queries as legal or casual conversation"""
    
    @staticmethod
    def is_casual_query(query: str) -> bool:
        """Check if query is casual conversation"""
        query_lower = query.lower().strip()
        
        # Check for exact casual patterns
        for category, patterns in CASUAL_PATTERNS.items():
            for pattern in patterns:
                if pattern in query_lower or query_lower == pattern:
                    return True
        
        # Check query length - very short queries are likely casual
        words = query_lower.split()
        if len(words) <= 2 and not any(word in LEGAL_KEYWORDS for word in words):
            return True
            
        return False
    
    @staticmethod
    def has_legal_context(query: str) -> bool:
        """Check if query contains legal keywords or context"""
        query_lower = query.lower()
        
        # Direct legal keyword check
        query_words = set(query_lower.split())
        if query_words.intersection(LEGAL_KEYWORDS):
            return True
        
        # Check for legal phrases
        legal_phrases = [
            'legal advice', 'court case', 'file a case', 'legal rights',
            'what does the law say', 'according to law', 'legally',
            'legal document', 'legal provision'
        ]
        
        for phrase in legal_phrases:
            if phrase in query_lower:
                return True
        
        return False
    
    @staticmethod
    def get_query_type(query: str) -> str:
        """Classify query type"""
        if QueryClassifier.is_casual_query(query):
            return 'casual'
        elif QueryClassifier.has_legal_context(query):
            return 'legal'
        elif len(query.strip().split()) >= 4: # Complex queries assumed legal
            return 'legal'
        else:
            return 'unclear'

class LegalDocumentChatbot:
    """Professional Legal Document Chatbot with Hybrid Search."""

    def __init__(self):
        self.faiss_db = None
        self.bm25_index = None
        self.chunks_metadata = []
        self.embeddings = None
        self.is_loaded = False
        self.config = {}
        self.user_chat_histories = {} # Store multiple chat sessions per user
        self.llm_lock = Lock()

    def validate_chunk_structure(self):
        """Validate that chunk metadata has the expected structure"""
        if not self.chunks_metadata:
            return False
        
        sample_chunk = self.chunks_metadata[0]
        if isinstance(sample_chunk, dict) and 'content' in sample_chunk and 'metadata' in sample_chunk:
            logger.info("✅ Chunk data structure validated - new format")
            return True
        else:
            logger.warning("⚠️ Unexpected chunk data structure - may need migration")
            return False

    def load_index(self) -> bool:
        """Load the pre-built search index and embeddings."""
        try:
            logger.info("🔄 Loading legal document index...")
            
            if not os.path.exists(INDEX_DIR):
                logger.error(f"❌ Index directory '{INDEX_DIR}' not found. Please run train.py first.")
                return False
            
            config_path = os.path.join(INDEX_DIR, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': DEVICE},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("✅ Embeddings model loaded")

            self.faiss_db = FAISS.load_local(INDEX_DIR, self.embeddings, allow_dangerous_deserialization=True)
            logger.info("✅ FAISS database loaded")
            
            bm25_path = os.path.join(INDEX_DIR, "bm25_index.pkl")
            chunks_path = os.path.join(INDEX_DIR, "chunks_metadata.pkl")

            if os.path.exists(bm25_path) and os.path.exists(chunks_path):
                with open(bm25_path, "rb") as f:
                    self.bm25_index = pickle.load(f)
                with open(chunks_path, "rb") as f:
                    chunk_data = pickle.load(f)
                    # Convert the chunk data format to match what we expect
                    self.chunks_metadata = []
                    for data in chunk_data:
                        # If it's the new format from train.py
                        if isinstance(data, dict) and 'content' in data:
                            self.chunks_metadata.append(data)
                        else:
                            # If it's old format, adapt accordingly
                            self.chunks_metadata.append(data)
                
                # Validate chunk structure
                self.validate_chunk_structure()
                logger.info("✅ BM25 index and metadata loaded")
            else:
                logger.error("❌ BM25 index or chunk metadata not found. Please run train.py.")
                return False
            
            self.is_loaded = True
            logger.info("🎯 Legal document index loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading index: {e}")
            self.is_loaded = False
            return False

    def hybrid_search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Perform hybrid search using FAISS and BM25 with score weighting."""
        if not self.is_loaded:
            return []
        
        try:
            results = {}
            
            # FAISS semantic search
            if self.faiss_db:
                semantic_results = self.faiss_db.similarity_search_with_score(query, k=top_k * 2)
                for doc, score in semantic_results:
                    chunk_id = doc.metadata.get('chunk_id')
                    if chunk_id is not None:
                        semantic_score = float(1 - score) if score <= 1.0 else 0.0
                        if chunk_id not in results:
                            results[chunk_id] = {
                                'content': doc.page_content, 
                                'metadata': doc.metadata, 
                                'semantic_score': semantic_score, 
                                'bm25_score': 0.0
                            }
                        else:
                            results[chunk_id]['semantic_score'] = max(results[chunk_id]['semantic_score'], semantic_score)
            
            # BM25 keyword search (FIXED VERSION)
            if self.bm25_index and self.chunks_metadata:
                query_tokens = query.lower().split()
                bm25_scores = self.bm25_index.get_scores(query_tokens)
                top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
                
                for idx in top_bm25_indices:
                    if idx < len(self.chunks_metadata) and bm25_scores[idx] > 0:
                        bm25_score = float(bm25_scores[idx])
                        normalized_bm25_score = min(bm25_score / 10.0, 1.0)
                        
                        # Get chunk data - handle both formats
                        chunk_data = self.chunks_metadata[idx]
                        if isinstance(chunk_data, dict):
                            content = chunk_data.get('content', '')
                            metadata = chunk_data.get('metadata', {})
                        else:
                            # Handle legacy format if needed
                            content = str(chunk_data)
                            metadata = {}
                        
                        chunk_id = metadata.get('chunk_id', idx)
                        
                        if chunk_id not in results:
                            results[chunk_id] = {
                                'content': content, 
                                'metadata': metadata, 
                                'semantic_score': 0.0, 
                                'bm25_score': normalized_bm25_score
                            }
                        else:
                            results[chunk_id]['bm25_score'] = max(results[chunk_id]['bm25_score'], normalized_bm25_score)
            
            # Combine scores and return results
            final_results = []
            for result in results.values():
                combined_score = (0.6 * result['semantic_score']) + (0.4 * result['bm25_score'])
                result['combined_score'] = combined_score
                final_results.append(result)
            
            final_results = [r for r in final_results if r['combined_score'] >= RELEVANCE_THRESHOLD]
            final_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Hybrid search error: {e}")
            return []

    def create_casual_response(self, query: str, query_type: str) -> str:
        """Create casual conversation responses"""
        query_lower = query.lower().strip()
        
        # Greeting responses
        if any(greeting in query_lower for greeting in CASUAL_PATTERNS['greetings']):
            return """Hello! 👋 Welcome to the Legal Assistant.

I'm a legal document assistant. I can help you find information about:
• Legal provisions and acts
• Court procedures and regulations  
• Rights and obligations under the law
• Legal definitions and interpretations

Please feel free to ask me any legal questions you might have!"""

        # Wellbeing responses
        if any(wellbeing in query_lower for wellbeing in CASUAL_PATTERNS['wellbeing']):
            return """Thank you for asking! I'm functioning well and ready to assist you with your legal queries. 😊

As your legal AI assistant, I'm here to help you navigate legal documents and provide information on various legal matters.

What legal topic would you like to explore today?"""

        # Thanks responses
        if any(thanks in query_lower for thanks in CASUAL_PATTERNS['thanks']):
            return """You're very welcome! 🙏 

I'm always here to help with your legal questions. If you need assistance with any legal matters, documentation, or have questions about laws and regulations, please don't hesitate to ask.

Is there anything specific you'd like to know about the law?"""

        # Goodbye responses
        if any(goodbye in query_lower for goodbye in CASUAL_PATTERNS['goodbye']):
            return """Goodbye! 👋 Thank you for using the Legal Assistant.

Feel free to return anytime you need assistance with legal matters. Take care and have a great day! ⚖️"""

        # Identity/capability questions
        if any(q in query_lower for q in ['who are you', 'what are you', 'what can you do']):
            return """I'm a legal document assistant! ⚖️

**What I can help you with:**
• Finding specific legal provisions in acts
• Explaining court procedures and legal processes
• Providing information about rights and obligations
• Interpreting legal definitions and terminology
• Navigating complex legal documents

**How to get the best results:**
• Ask specific legal questions
• Mention relevant legal areas (civil, criminal, commercial, etc.)
• Use legal keywords when possible

What legal topic can I assist you with today?"""

        # Unclear queries
        if query_type == 'unclear':
            return """I'd be happy to help you with legal matters! 📚

To provide you with the most accurate legal information, could you please:
• Be more specific about the legal topic you're interested in
• Mention any relevant acts or legal areas
• Provide more context about your legal question

For example, you could ask about:
• "What does the Penal Code say about..."
• "How do I file a case..."
• "What are the requirements for..."

What specific legal information are you looking for?"""

        # Default casual response
        return """Hello! I'm Legal Assistant, specialized in legal documents and law. ⚖️

I'm designed to help with legal queries, document interpretation, and providing information about legal provisions.

For the best assistance, please ask me specific legal questions or mention the legal topic you're interested in.

How can I help you with legal matters today?"""

    def stream_response_from_llm(self, query: str, search_results: List[Dict[str, Any]], user_id: str, chat_id: str, query_type: str = 'legal'):
        """Simulates streaming a response from an LLM."""
        if query_type == 'casual':
            full_response = self.create_casual_response(query, query_type)
        elif not search_results:
            full_response = self._create_no_results_response()
        else:
            full_response = self._create_professional_response(query, search_results)
        
        lines = full_response.split('\n')
        
        for line in lines:
            words = line.split()
            current_chunk = ""
            if not words:
                time.sleep(0.01) 
                socketio.emit('response_chunk', {'chunk': '\n', 'user_id': user_id, 'chat_id': chat_id})
                continue
            
            for word in words:
                current_chunk += word + " "
                if len(current_chunk) > 30: 
                    time.sleep(0.03) 
                    socketio.emit('response_chunk', {'chunk': current_chunk, 'user_id': user_id, 'chat_id': chat_id})
                    current_chunk = ""
            
            if current_chunk:
                time.sleep(0.03) 
                socketio.emit('response_chunk', {'chunk': current_chunk, 'user_id': user_id, 'chat_id': chat_id})
            
            socketio.emit('response_chunk', {'chunk': '\n', 'user_id': user_id, 'chat_id': chat_id})
        
        self.add_to_chat_history(user_id, chat_id, query, full_response)
        socketio.emit('response_complete', {'user_id': user_id, 'chat_id': chat_id})

    def _to_sentence_case(self, text: str) -> str:
        """Converts a string to sentence case, handling acronyms and proper nouns."""
        if not text:
            return ""
        
        # A quick fix for known legal acronyms/phrases to be case-sensitive
        text = text.replace("ACT NO", "Act No")
        
        sentences = re.split(r'(?<=[.!?])\s+', text.strip().lower())
        
        capitalized_sentences = []
        for sentence in sentences:
            if not sentence:
                continue
            capitalized = sentence.capitalize()
            capitalized_sentences.append(capitalized)
            
        return " ".join(capitalized_sentences)
    
    def _clean_and_extract_content(self, raw_content: str) -> str:
        """Clean and normalize extracted text, preserving legal citations."""
        if not raw_content:
            return ""

        # Protect "Act No." and its associated numbers
        protected_content = re.sub(r'(act\s+no\.?\s*\d+(?:\s+of\s*\d+)?)', r'__ACTNO__\1__', raw_content, flags=re.IGNORECASE)
        
        # Remove problematic patterns from the user's output
        cleaned_content = re.sub(r'Sectioncontent:\s*', '', protected_content, flags=re.IGNORECASE)
        cleaned_content = re.sub(r'Description-head:\s*', '', cleaned_content, flags=re.IGNORECASE)
        cleaned_content = re.sub(r'\[\s*\d+\s*,\s*\d+\s*of\s*\d+\s*]\s*', '', cleaned_content, flags=re.IGNORECASE)

        # General noise cleaning patterns
        noise_patterns = [
            r'File:\s*\d+[A-Z]\d+[A-Z]\d+[A-Z]\d+\.xml', 
            r'chapter:\s*act-name:', 
            r'sectiongroup:\s*section:\s*sectionshorttitle:', 
            r'sectionshorttitle:', 
            r'section:\s*\d+', 
            r'^\d+\.\s*File:', 
            r'^\d+\.\s*Telecommunication\s*description', 
            r'^\d+\.\s*Chapterdescription', 
            r'Limited\s*\(hereafter\s*in\s*this', 
            r'^\d+\.\s*Validation\s*of\s*the\s*purchase', 
        ]
        
        for pattern in noise_patterns:
            cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.IGNORECASE | re.MULTILINE)
        
        # Normalize whitespace and fix newlines
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
        cleaned_content = cleaned_content.strip()

        # Restore the protected "Act No." phrases
        final_content = re.sub(r'__ACTNO__(.*?)__', lambda m: m.group(1), cleaned_content)

        return final_content

    def _extract_meaningful_sentences(self, content: str) -> List[str]:
        """Extract meaningful sentences from cleaned content."""
        if not content:
            return []
        
        sentences = re.split(r'(?<!Act No\.)(?<!No\.)(?<=[.!?])\s+', content)
        meaningful_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            if len(sentence.split()) < 5:
                continue
                
            if len(re.findall(r'\d', sentence)) > len(sentence) * 0.3 and 'Act No.' not in sentence:
                continue
                
            if sentence.isupper() and len(sentence) > 10 and 'Act No.' not in sentence:
                continue
            
            noise_words = ['file:', 'chapter:', 'section:', 'description:', 'hereafter', 'xml', 'pdf']
            if any(noise.lower() in sentence.lower() for noise in noise_words):
                continue
            
            meaningful_sentences.append(sentence)
        
        return meaningful_sentences

    def _get_cleaned_source_name(self, source_name: str) -> str:
        """Cleans up the source file name for display."""
        if not source_name:
            return "Unknown Document"
        
        name_without_ext = os.path.splitext(source_name)[0]
        cleaned_name = re.sub(r'^\d{4}Y\d+V\d+C\d+A', '', name_without_ext)
        cleaned_name = re.sub(r'[_-]', ' ', cleaned_name).strip()
        cleaned_name = " ".join(word.capitalize() for word in cleaned_name.split())
        
        # Special case for "Act No." and similar terms
        cleaned_name = re.sub(r'Act No', 'Act No.', cleaned_name, flags=re.IGNORECASE)
        
        if not cleaned_name:
            return source_name
            
        return cleaned_name

    def _create_professional_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Generate a clean, well-formatted professional response."""
        
        if not search_results:
            return self._create_no_results_response()

        all_sentences = []
        for result in search_results:
            cleaned_content = self._clean_and_extract_content(result['content'])
            sentences = self._extract_meaningful_sentences(cleaned_content)
            all_sentences.extend(sentences)
        
        unique_sentences = []
        seen = set()
        for sentence in all_sentences:
            sentence_lower = sentence.lower()
            if sentence_lower not in seen and len(sentence) > 25:
                seen.add(sentence_lower)
                unique_sentences.append(self._to_sentence_case(sentence))

        if not unique_sentences:
            return self._create_no_results_response()
            
        response_parts = []
        response_parts.append("## 📋 Legal Information Summary")
        response_parts.append("")
        
        response_parts.append("Based on the legal documents, here is a detailed summary of the key provisions relevant to your query:")
        response_parts.append("")
        
        # Create a coherent paragraph from the key sentences
        summary_paragraph = " ".join(unique_sentences[:5])
        response_parts.append(summary_paragraph)

        if len(unique_sentences) > 5:
            response_parts.append("")
            response_parts.append("Furthermore, the documents contain the following additional provisions:")
            response_parts.append("")
            for sentence in unique_sentences[5:10]:
                response_parts.append(f"• {sentence}")

        response_parts.append("")
        response_parts.append("---")
        response_parts.append("")
        
        response_parts.append("**📚 Document Sources:**")
        response_parts.append("")
        
        sources_used = set()
        source_list = []
        for result in search_results:
            source_name = result['metadata'].get('source_file', 'Unknown Document')
            if source_name not in sources_used:
                clean_source_name = self._get_cleaned_source_name(source_name)
                page_info = f" (Page {result['metadata'].get('page_number')})" if result['metadata'].get('page_number') else ""
                
                source_list.append(f"{clean_source_name}{page_info}")
                sources_used.add(source_name)
        
        for i, source in enumerate(source_list[:5], 1):
             response_parts.append(f"{i}. {source}")

        response_parts.append("")
        response_parts.append("---")
        response_parts.append("")
        response_parts.append("**⚖️ Legal Disclaimer:** This information is provided for informational purposes only and does not constitute legal advice. Always consult with a qualified attorney for specific legal guidance regarding your situation.")

        return "\n".join(response_parts)
    
    def _create_no_results_response(self):
        """Create response when no relevant results are found."""
        return """I couldn't find specific information related to your legal query in the available documents.

**💡 Suggestions to improve your search:**
• Try rephrasing your question using different legal terms
• Be more specific about the legal area (e.g., civil law, criminal law, commercial law)
• Include relevant act names or legal provisions
• Break down complex questions into simpler parts

**📚 Legal areas I can help with:**
• Constitutional Law • Civil Procedure • Criminal Law
• Commercial Law • Family Law • Property Law
• Employment Law • Administrative Law

**⚖️ Legal Disclaimer:** This information is for informational purposes only and does not constitute legal advice. Please consult with a qualified attorney for specific legal guidance.

Could you please rephrase your question with more specific legal terms or context?"""
    
    def create_new_chat(self, user_id: str) -> str:
        """Create a new chat session for a user."""
        if user_id not in self.user_chat_histories:
            self.user_chat_histories[user_id] = {}
        
        chat_id = str(uuid.uuid4())
        self.user_chat_histories[user_id][chat_id] = {
            'messages': [],
            'created_at': datetime.now().isoformat(),
            'title': 'New Chat',
            'last_updated': datetime.now().isoformat()
        }
        return chat_id

    def get_chat_history(self, user_id: str, chat_id: str) -> List[Dict[str, str]]:
        """Get messages for a specific chat."""
        if user_id in self.user_chat_histories and chat_id in self.user_chat_histories[user_id]:
            return self.user_chat_histories[user_id][chat_id]['messages']
        return []

    def get_user_chats(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all chats for a user."""
        if user_id not in self.user_chat_histories:
            return []
        
        chats = []
        for chat_id, chat_data in self.user_chat_histories[user_id].items():
            chats.append({
                'chat_id': chat_id,
                'title': chat_data.get('title', 'New Chat'),
                'created_at': chat_data.get('created_at'),
                'last_updated': chat_data.get('last_updated'),
                'message_count': len(chat_data.get('messages', []))
            })
        
        # Sort by last updated, most recent first
        chats.sort(key=lambda x: x['last_updated'], reverse=True)
        return chats

    def add_to_chat_history(self, user_id: str, chat_id: str, query: str, response: str):
        """Add a message to a specific chat."""
        if user_id not in self.user_chat_histories:
            self.user_chat_histories[user_id] = {}
        
        if chat_id not in self.user_chat_histories[user_id]:
            self.user_chat_histories[user_id][chat_id] = {
                'messages': [],
                'created_at': datetime.now().isoformat(),
                'title': 'New Chat',
                'last_updated': datetime.now().isoformat()
            }
        
        # Update title with first query if it's still "New Chat"
        if self.user_chat_histories[user_id][chat_id]['title'] == 'New Chat':
            title = query[:50] + "..." if len(query) > 50 else query
            self.user_chat_histories[user_id][chat_id]['title'] = title
        
        self.user_chat_histories[user_id][chat_id]['messages'].append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response
        })
        
        self.user_chat_histories[user_id][chat_id]['last_updated'] = datetime.now().isoformat()

    def clear_all_chats(self, user_id: str):
        """Clear all chats for a user."""
        if user_id in self.user_chat_histories:
            del self.user_chat_histories[user_id]
    
    def delete_chat(self, user_id: str, chat_id: str):
        """Delete a specific chat."""
        if user_id in self.user_chat_histories and chat_id in self.user_chat_histories[user_id]:
            del self.user_chat_histories[user_id][chat_id]
    
    def get_document_stats(self) -> Dict[str, Any]:
        if not self.is_loaded:
            return {}
        
        total_chunks = len(self.chunks_metadata)
        documents = set()
        file_types = defaultdict(int)
        
        for chunk in self.chunks_metadata:
            if isinstance(chunk, dict):
                metadata = chunk.get('metadata', {})
            else:
                # Handle legacy format
                metadata = {}
            
            documents.add(metadata.get('source_file', 'Unknown'))
            file_type = metadata.get('file_type', 'unknown')
            file_types[file_type] += 1
        
        return {
            'total_documents': len(documents),
            'total_chunks': total_chunks,
            'file_types': dict(file_types),
            'index_version': self.config.get('version', 'Unknown'),
            'last_updated': self.config.get('timestamp', 'Unknown')
        }

# --- Flask Routes and SocketIO ---

chatbot = LegalDocumentChatbot()

@app.route('/')
def index():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    ip_address = request.remote_addr
    is_limited, remaining = check_rate_limit(ip_address)
    if not is_limited:
        return jsonify({'error': f'Rate limit exceeded. Try again in {RATE_LIMIT_DURATION/60} minutes.'}), 429

    try:
        data = request.get_json()
        query = data.get('message', '').strip()
        user_id = data.get('user_id')
        chat_id = data.get('chat_id')
        
        if not query:
            return jsonify({'error': 'Empty query.'}), 400
        
        if not chatbot.is_loaded:
            return jsonify({'error': 'Document index not loaded. Please check server logs.'}), 500
        
        # Create new chat if not provided
        if not chat_id:
            chat_id = chatbot.create_new_chat(user_id)
        
        # Classify query type
        query_type = QueryClassifier.get_query_type(query)
        
        # Only perform search for legal queries
        search_results = []
        if query_type == 'legal':
            search_results = chatbot.hybrid_search(query, top_k=8)
        
        socketio.start_background_task(chatbot.stream_response_from_llm, query, search_results, user_id, chat_id, query_type)
        
        return jsonify({'status': 'streaming_started', 'chat_id': chat_id})
        
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/chats/<string:user_id>', methods=['GET'])
def get_user_chats(user_id):
    """Get all chats for a user."""
    chats = chatbot.get_user_chats(user_id)
    return jsonify({'chats': chats})

@app.route('/api/chat/<string:user_id>/<string:chat_id>', methods=['GET'])
def get_chat_history(user_id, chat_id):
    """Get specific chat history."""
    history = chatbot.get_chat_history(user_id, chat_id)
    return jsonify({'history': history})

@app.route('/api/new-chat/<string:user_id>', methods=['POST'])
def create_new_chat(user_id):
    """Create a new chat."""
    chat_id = chatbot.create_new_chat(user_id)
    return jsonify({'chat_id': chat_id})

@app.route('/api/clear-all/<string:user_id>', methods=['DELETE'])
def clear_all_chats(user_id):
    """Clear all chats for a user."""
    chatbot.clear_all_chats(user_id)
    return jsonify({'success': True})

@app.route('/api/delete-chat/<string:user_id>/<string:chat_id>', methods=['DELETE'])
def delete_chat(user_id, chat_id):
    """Delete a specific chat."""
    chatbot.delete_chat(user_id, chat_id)
    return jsonify({'success': True})

@socketio.on('connect')
def handle_connect():
    pass 

@socketio.on('disconnect')
def handle_disconnect():
    pass 

@app.route('/api/documents', methods=['GET'])
def get_documents():
    stats = chatbot.get_document_stats()
    return jsonify(stats)

@app.route('/api/health', methods=['GET'])
def get_health():
    return jsonify({
        'status': 'healthy' if chatbot.is_loaded else 'unhealthy',
        'index_loaded': chatbot.is_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/feedback', methods=['POST'])
def save_feedback():
    try:
        feedback_data = request.get_json()
        logger.info(f"📝 User feedback received")
        return jsonify({'status': 'Feedback received'}), 200
    except Exception as e:
        logger.error(f"❌ Feedback endpoint error: {e}")
        return jsonify({'error': 'Failed to save feedback'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found.'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"❌ Internal error: {error}")
    return jsonify({'error': 'Internal server error.'}), 500

def print_startup_banner():
    """Print a clean startup banner with network information"""
    local_ip = get_local_ip()
    port = 5000
    
    print("\n" + "="*60)
    print("🔥 LEGAL CHATBOT SERVER")
    print("="*60)
    print(f"📍 Local Access: http://localhost:{port}")
    print(f"🌐 Network Access: http://{local_ip}:{port}")
    print(f"⚡ Device: {DEVICE.upper()}")
    print(f"📊 Status: {'✅ Ready' if chatbot.is_loaded else '❌ Index Not Loaded'}")
    print("="*60)
    print("🚀 Server is running! Press Ctrl+C to stop")
    print("="*60 + "\n")

# --- Main Execution ---

if __name__ == '__main__':
    with app.app_context():
        success = chatbot.load_index()
        if not success:
            logger.error("❌ Failed to load document index during application startup!")
    
    print_startup_banner()
    
    socketio.run(
        app, 
        debug=True,
        host='0.0.0.0',
        port=5000,
        allow_unsafe_werkzeug=True,
        use_reloader=True
    )