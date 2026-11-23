import os
import pickle
import re
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

# Auto-install required packages
def install_requirements():
    import platform
    import subprocess
    import sys
    
    # Check if we're on Windows and handle FAISS appropriately
    is_windows = platform.system() == "Windows"
    
    # Try to detect CUDA availability before choosing FAISS version
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False
        # Install torch first
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
            import torch
            cuda_available = torch.cuda.is_available()
        except:
            cuda_available = False
    
    # Choose FAISS version based on CUDA availability and platform
    if cuda_available and not is_windows:
        faiss_package = "faiss-gpu"
        print("🚀 CUDA detected on non-Windows system - using faiss-gpu")
    else:
        faiss_package = "faiss-cpu"
        if is_windows:
            print("💻 Windows detected - using faiss-cpu")
        else:
            print("⚡ CUDA not available - using faiss-cpu")
    
    packages = [
        "rank-bm25", "nltk", "langchain", "langchain-community", 
        "sentence-transformers", faiss_package, "PyPDF2", "beautifulsoup4",
        "lxml", "transformers", "torch"
    ]
    
    for package in packages:
        try:
            # Handle special cases for import names
            import_name = package.replace('-', '_')
            if package == "faiss-gpu" or package == "faiss-cpu":
                import_name = "faiss"
            elif package == "PyPDF2":
                import_name = "PyPDF2"
            
            __import__(import_name)
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package}: {e}")
                if "faiss" in package and cuda_available:
                    print("Falling back to faiss-cpu...")
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
                    except:
                        print("❌ Failed to install FAISS. Please install manually.")

install_requirements()

from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import torch
import warnings
warnings.filterwarnings("ignore")

# Download NLTK data
nltk_downloads = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
for item in nltk_downloads:
    try:
        nltk.data.find(f'tokenizers/{item}')
    except:
        try:
            nltk.data.find(f'corpora/{item}')
        except:
            try:
                nltk.download(item, quiet=True)
            except:
                print(f"Failed to download {item}")

# Configuration
SOURCE_DIR = "files"
INDEX_DIR = "legal_index"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Auto-detect optimal batch size based on device and available memory
def get_optimal_batch_size():
    """Automatically determine optimal batch size based on device capabilities"""
    if torch.cuda.is_available():
        try:
            # Get GPU memory info
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            if gpu_memory >= 8:
                return 64  # High-end GPU
            elif gpu_memory >= 4:
                return 32  # Mid-range GPU
            else:
                return 16  # Lower-end GPU
        except:
            return 32  # Default for GPU
    else:
        # CPU mode - smaller batches to avoid memory issues
        return 8

BATCH_SIZE = get_optimal_batch_size()

LEGAL_KEYWORDS = {
    'certificate', 'tax', 'reserve', 'legal', 'court', 'statute', 'regulation',
    'provision', 'section', 'clause', 'article', 'amendment', 'act', 'law',
    'jurisdiction', 'procedure', 'filing', 'notice', 'compliance', 'liability',
    'judgment', 'appeal', 'petition', 'defendant', 'plaintiff', 'contract'
}

class FileRegistryManager:
    """Manages processed and failed file registries with thread safety"""
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.processed_registry_path = os.path.join(index_dir, "processed_files.json")
        self.failed_registry_path = os.path.join(index_dir, "failed_files.json")
        self.training_log_path = os.path.join(index_dir, "training_log.json")
        
        # Create index directory if it doesn't exist
        os.makedirs(index_dir, exist_ok=True)
        
        # Load existing registries
        self.processed_files = self.load_processed_registry()
        self.failed_files = self.load_failed_registry()
        self.training_sessions = self.load_training_log()
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate SHA-256 file hash for reliable tracking"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256()
                chunk = f.read(8192)
                while chunk:
                    file_hash.update(chunk)
                    chunk = f.read(8192)
            
            # Include file size and modification time for better tracking
            stat = os.stat(file_path)
            combined_hash = hashlib.sha256()
            combined_hash.update(file_hash.hexdigest().encode())
            combined_hash.update(str(stat.st_size).encode())
            combined_hash.update(str(stat.st_mtime).encode())
            
            return combined_hash.hexdigest()
        except Exception as e:
            print(f"Error generating hash for {file_path}: {e}")
            return None
    
    def load_processed_registry(self) -> Dict:
        """Load registry of processed files"""
        try:
            if os.path.exists(self.processed_registry_path):
                with open(self.processed_registry_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load processed files registry: {e}")
        return {}
    
    def load_failed_registry(self) -> Dict:
        """Load registry of failed files"""
        try:
            if os.path.exists(self.failed_registry_path):
                with open(self.failed_registry_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load failed files registry: {e}")
        return {}
    
    def load_training_log(self) -> List:
        """Load training session log"""
        try:
            if os.path.exists(self.training_log_path):
                with open(self.training_log_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load training log: {e}")
        return []
    
    def is_file_processed(self, file_path: str) -> bool:
        """Check if file has been processed"""
        relative_path = os.path.relpath(file_path)
        if relative_path not in self.processed_files:
            return False
        
        # Check if file has been modified since last processing
        current_hash = self.get_file_hash(file_path)
        if current_hash is None:
            return False
        
        stored_hash = self.processed_files[relative_path].get('file_hash')
        return current_hash == stored_hash
    
    def mark_file_processed(self, file_path: str, chunks_created: int, processing_time: float):
        """Mark file as processed"""
        relative_path = os.path.relpath(file_path)
        file_hash = self.get_file_hash(file_path)
        
        self.processed_files[relative_path] = {
            'file_hash': file_hash,
            'processed_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'chunks_created': chunks_created,
            'processing_time_seconds': processing_time,
            'file_size_bytes': os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }
    
    def mark_file_failed(self, file_path: str, error_reason: str):
        """Mark file as failed"""
        relative_path = os.path.relpath(file_path)
        file_hash = self.get_file_hash(file_path)
        
        self.failed_files[relative_path] = {
            'file_hash': file_hash,
            'failed_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'error_reason': error_reason,
            'file_size_bytes': os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }
    
    def log_training_session(self, session_stats: Dict):
        """Log training session details"""
        session_log = {
            'session_id': len(self.training_sessions) + 1,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'stats': session_stats
        }
        self.training_sessions.append(session_log)
    
    def save_registries(self):
        """Save all registries to disk with error handling"""
        try:
            # Create backup of existing files
            for registry_path in [self.processed_registry_path, self.failed_registry_path, self.training_log_path]:
                if os.path.exists(registry_path):
                    backup_path = registry_path + '.backup'
                    try:
                        os.rename(registry_path, backup_path)
                    except:
                        pass
            
            # Save processed files registry
            with open(self.processed_registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.processed_files, f, indent=2, ensure_ascii=False)
            
            # Save failed files registry
            with open(self.failed_registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.failed_files, f, indent=2, ensure_ascii=False)
            
            # Save training log
            with open(self.training_log_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_sessions, f, indent=2, ensure_ascii=False)
            
            print("✓ File registries saved successfully")
            
            # Remove backup files on success
            for registry_path in [self.processed_registry_path, self.failed_registry_path, self.training_log_path]:
                backup_path = registry_path + '.backup'
                if os.path.exists(backup_path):
                    try:
                        os.remove(backup_path)
                    except:
                        pass
            
        except Exception as e:
            print(f"❌ Error saving registries: {e}")
            # Restore from backup if save failed
            for registry_path in [self.processed_registry_path, self.failed_registry_path, self.training_log_path]:
                backup_path = registry_path + '.backup'
                if os.path.exists(backup_path):
                    try:
                        os.rename(backup_path, registry_path)
                        print(f"✓ Restored {registry_path} from backup")
                    except:
                        pass
    
    def get_unprocessed_files(self, all_files: List[str], max_files: int = None) -> List[str]:
        """Get list of unprocessed files up to max_files limit"""
        unprocessed = []
        
        for file_path in all_files:
            if not self.is_file_processed(file_path):
                unprocessed.append(file_path)
                if max_files and len(unprocessed) >= max_files:
                    break
        
        return unprocessed
    
    def get_registry_stats(self) -> Dict:
        """Get registry statistics"""
        return {
            'total_processed_files': len(self.processed_files),
            'total_failed_files': len(self.failed_files),
            'total_training_sessions': len(self.training_sessions),
            'last_training_session': self.training_sessions[-1]['timestamp'] if self.training_sessions else "Never"
        }

class UniversalDocumentLoader:
    """Universal document loader for multiple formats"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english')) - LEGAL_KEYWORDS
        except:
            self.stop_words = set()
            print("⚠️ Could not load stopwords, using empty set")
        
        try:
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.lemmatizer = None
            print("⚠️ Could not load lemmatizer")
    
    def extract_text_from_html(self, file_path: str) -> str:
        """Enhanced HTML text extraction"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()
            
            # Get text from specific content areas first
            text_content = ""
            content_selectors = ['main', '.content', '#content', '.main-content', 'article', 'body']
            
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    text_content = ' '.join([elem.get_text(separator=' ', strip=True) for elem in elements])
                    break
            
            # Fallback to body text if no specific content found
            if not text_content or len(text_content) < 100:
                text_content = soup.get_text(separator=' ', strip=True)
            
            return self.clean_extracted_text(text_content)
            
        except Exception as e:
            print(f"HTML extraction error: {e}")
            return ""
    
    def extract_text_from_xml(self, file_path: str) -> str:
        """Extract text from XML files"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            text_content = ""
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    text_content += elem.text.strip() + " "
                if elem.tail and elem.tail.strip():
                    text_content += elem.tail.strip() + " "
            
            return self.clean_extracted_text(text_content)
            
        except Exception as e:
            print(f"XML extraction error: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return self.clean_extracted_text(content)
        except Exception as e:
            print(f"TXT extraction error: {e}")
            return ""
    
    def clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Limit excessive newlines
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/\%\$\&\@\#]', ' ', text)
        text = text.strip()
        
        return text
    
    def is_content_valid(self, text: str) -> bool:
        """Check if extracted content is valid"""
        if not text or len(text) < 50:
            return False
        
        # Check if content has enough meaningful words
        words = text.split()
        if len(words) < 10:
            return False
        
        # Check for repetitive content
        unique_words = set(words)
        if len(unique_words) < len(words) * 0.1:
            return False
        
        return True
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load document based on file extension with enhanced error handling"""
        file_ext = Path(file_path).suffix.lower()
        file_name = Path(file_path).name
        
        try:
            if file_ext == ".pdf":
                return self.load_pdf_document(file_path)
            elif file_ext in [".html", ".htm"]:
                content = self.extract_text_from_html(file_path)
            elif file_ext == ".xml":
                content = self.extract_text_from_xml(file_path)
            elif file_ext == ".txt":
                content = self.extract_text_from_txt(file_path)
            else:
                print(f"Unsupported format: {file_name}")
                return []
            
            if not self.is_content_valid(content):
                print(f"⚠ {file_name}: Invalid or insufficient content")
                return []
            
            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    "source_file": file_name,
                    "full_path": str(file_path),
                    "file_type": file_ext[1:],
                    "word_count": len(content.split()),
                    "content_length": len(content),
                }
            )
            
            print(f"✓ {file_name}: {len(content)} chars extracted")
            return [doc]
            
        except Exception as e:
            print(f"✗ {file_name}: {str(e)}")
            return []
    
    def load_pdf_document(self, file_path: str) -> List[Document]:
        """Load PDF document with enhanced handling"""
        try:
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()
            
            valid_docs = []
            for page_num, doc in enumerate(raw_docs):
                cleaned_content = self.clean_extracted_text(doc.page_content)
                
                if self.is_content_valid(cleaned_content):
                    doc.page_content = cleaned_content
                    doc.metadata.update({
                        "source_file": Path(file_path).name,
                        "full_path": str(file_path),
                        "page_number": page_num + 1,
                        "file_type": "pdf",
                        "word_count": len(cleaned_content.split()),
                        "content_length": len(cleaned_content),
                    })
                    valid_docs.append(doc)
            
            return valid_docs
            
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return []

class IncrementalLegalProcessor:
    """Incremental legal document processor with file tracking and memory management"""
    
    def __init__(self, index_dir: str):
        self.loader = UniversalDocumentLoader()
        self.registry = FileRegistryManager(index_dir)
        self.processed_files = 0
        self.total_files = 0
        self.failed_files = []
        self.stats = {'total_pages': 0, 'chunks_created': 0}
        self.next_chunk_id = 0  # For incremental chunk IDs
    
    def cleanup_memory(self):
        """Clean up GPU/CPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        try:
            tokens = word_tokenize(text.lower())
        except:
            # Fallback to simple split if NLTK fails
            tokens = text.lower().split()
        
        keywords = []
        
        for token in tokens:
            if (token.isalpha() and len(token) > 2 and 
                token not in self.loader.stop_words):
                try:
                    if self.loader.lemmatizer:
                        lemmatized = self.loader.lemmatizer.lemmatize(token)
                    else:
                        lemmatized = token
                except:
                    lemmatized = token
                keywords.append(lemmatized)
        
        return list(set(keywords))[:30]
    
    def load_existing_chunk_metadata(self) -> List[Document]:
        """Load existing chunks from previous training sessions"""
        metadata_path = os.path.join(self.registry.index_dir, "chunks_metadata.pkl")
        existing_chunks = []
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'rb') as f:
                    chunk_data = pickle.load(f)
                
                for data in chunk_data:
                    doc = Document(
                        page_content=data['content'],
                        metadata=data['metadata']
                    )
                    existing_chunks.append(doc)
                
                # Update next chunk ID
                if existing_chunks:
                    max_chunk_id = max([chunk.metadata.get('chunk_id', 0) for chunk in existing_chunks])
                    self.next_chunk_id = max_chunk_id + 1
                
                print(f"✓ Loaded {len(existing_chunks)} existing chunks")
                
            except Exception as e:
                print(f"⚠️ Could not load existing chunks: {e}")
        
        return existing_chunks
    
    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """Create optimized chunks from documents"""
        print("\n✂️ Creating document chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        all_chunks = []
        
        for doc in documents:
            try:
                chunks = text_splitter.split_documents([doc])
                
                for chunk_idx, chunk in enumerate(chunks):
                    if len(chunk.page_content.strip()) < 50:
                        continue
                    
                    keywords = self.extract_keywords(chunk.page_content)
                    
                    chunk.metadata.update({
                        "chunk_id": self.next_chunk_id,
                        "chunk_index": chunk_idx,
                        "chunk_keywords": keywords,
                        "chunk_size": len(chunk.page_content),
                        "chunk_words": len(chunk.page_content.split()),
                        "has_legal_terms": bool(set(keywords) & LEGAL_KEYWORDS)
                    })
                    
                    all_chunks.append(chunk)
                    self.next_chunk_id += 1
                    
            except Exception as e:
                print(f"Chunking error: {e}")
                continue
        
        print(f"✓ Created {len(all_chunks)} new chunks")
        return all_chunks
    
    def load_batch_documents(self, max_files: int = None) -> Tuple[List[Document], List[str]]:
        """Load documents in batches with file tracking and periodic cleanup"""
        print("🔍 Scanning for unprocessed documents...")
        
        if not os.path.exists(SOURCE_DIR):
            print(f"❌ Source directory '{SOURCE_DIR}' not found!")
            return [], []
        
        # Find all supported files
        supported_extensions = {".pdf", ".html", ".htm", ".xml", ".txt"}
        all_files = []
        
        for root, _, files in os.walk(SOURCE_DIR):
            for file in files:
                if Path(file).suffix.lower() in supported_extensions:
                    all_files.append(os.path.join(root, file))
        
        print(f"Found {len(all_files)} total supported files")
        
        # Get unprocessed files
        unprocessed_files = self.registry.get_unprocessed_files(all_files, max_files)
        
        if not unprocessed_files:
            print("✅ All files have been processed already!")
            return [], []
        
        self.total_files = len(unprocessed_files)
        print(f"Processing {self.total_files} unprocessed files" + 
              (f" (limited to {max_files})" if max_files else ""))
        
        # Process files with periodic cleanup
        all_documents = []
        processed_file_list = []
        
        for i, file_path in enumerate(unprocessed_files, 1):
            file_name = Path(file_path).name
            start_time = time.time()
            
            try:
                docs = self.loader.load_document(file_path)
                processing_time = time.time() - start_time
                
                if docs:
                    all_documents.extend(docs)
                    chunks_created = len(docs)  # For non-PDF files, this is typically 1
                    
                    # Mark as processed
                    self.registry.mark_file_processed(file_path, chunks_created, processing_time)
                    processed_file_list.append(file_path)
                    self.processed_files += 1
                else:
                    # Mark as failed
                    self.registry.mark_file_failed(file_path, "No valid content extracted")
                    self.failed_files.append((file_name, "No valid content extracted"))
                    
            except Exception as e:
                processing_time = time.time() - start_time
                error_msg = str(e)
                
                # Mark as failed
                self.registry.mark_file_failed(file_path, error_msg)
                self.failed_files.append((file_name, error_msg))
                print(f"✗ {file_name}: {error_msg}")
            
            # Periodic cleanup and checkpoint saving
            if i % 100 == 0:
                self.cleanup_memory()
                # Save progress checkpoint
                try:
                    self.registry.save_registries()
                    print(f"💾 Checkpoint saved at {i}/{len(unprocessed_files)} files")
                except:
                    pass
            
            # Progress update
            if i % 50 == 0 or i == len(unprocessed_files):
                progress = (i / len(unprocessed_files)) * 100
                print(f"Progress: {progress:.1f}% ({i}/{len(unprocessed_files)})")
        
        self.stats['total_pages'] = len(all_documents)
        print(f"\n📊 Batch Summary: {len(all_documents)} documents loaded from {self.processed_files} files")
        
        return all_documents, processed_file_list

class IncrementalHybridSearchIndex:
    """Incremental hybrid search index with BM25 and FAISS"""
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.chunks = []
        self.chunk_texts = []
        self.bm25 = None
        self.keyword_to_chunks = defaultdict(list)
        
        # Load existing index if available
        self.load_existing_index()
    
    def load_existing_index(self):
        """Load existing search index components"""
        try:
            # Load existing chunks
            chunks_path = os.path.join(self.index_dir, "chunks_metadata.pkl")
            if os.path.exists(chunks_path):
                with open(chunks_path, 'rb') as f:
                    chunk_data = pickle.load(f)
                
                for data in chunk_data:
                    doc = Document(
                        page_content=data['content'],
                        metadata=data['metadata']
                    )
                    self.chunks.append(doc)
                
                self.chunk_texts = [chunk.page_content for chunk in self.chunks]
                
                # Rebuild BM25 index
                tokenized_chunks = [chunk.page_content.lower().split() for chunk in self.chunks]
                self.bm25 = BM25Okapi(tokenized_chunks)
                
                # Rebuild keyword mappings
                for idx, chunk in enumerate(self.chunks):
                    for keyword in chunk.metadata.get('chunk_keywords', []):
                        self.keyword_to_chunks[keyword].append(idx)
                
                print(f"✓ Loaded existing index with {len(self.chunks)} chunks")
            else:
                print("ℹ️ No existing index found, starting fresh")
                
        except Exception as e:
            print(f"⚠️ Error loading existing index: {e}")
            self.chunks = []
            self.chunk_texts = []
    
    def add_chunks(self, new_chunks: List[Document]):
        """Add new chunks to existing index efficiently"""
        if not new_chunks:
            print("No new chunks to add")
            return
        
        print(f"📈 Adding {len(new_chunks)} new chunks to existing index...")
        
        # Add chunks
        self.chunks.extend(new_chunks)
        self.chunk_texts = [chunk.page_content for chunk in self.chunks]
        
        # Rebuild BM25 index with all chunks (more efficient than incremental)
        tokenized_chunks = [chunk.page_content.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
        # Update keyword mappings incrementally
        start_idx = len(self.chunks) - len(new_chunks)
        for i, chunk in enumerate(new_chunks):
            chunk_idx = start_idx + i
            for keyword in chunk.metadata.get('chunk_keywords', []):
                self.keyword_to_chunks[keyword].append(chunk_idx)
        
        print(f"✓ Index now contains {len(self.chunks)} total chunks")
    
    def save_index(self, faiss_db):
        """Save complete search index with atomic operations"""
        print(f"\n💾 Saving incremental index to {self.index_dir}...")
        os.makedirs(self.index_dir, exist_ok=True)
        
        try:
            # Create temporary files for atomic operations
            temp_dir = os.path.join(self.index_dir, "temp_save")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save FAISS database
            faiss_db.save_local(temp_dir)
            
            # Save BM25 index
            if self.bm25:
                with open(os.path.join(temp_dir, "bm25_index.pkl"), "wb") as f:
                    pickle.dump(self.bm25, f)
            
            # Save keyword mappings
            with open(os.path.join(temp_dir, "keyword_mappings.pkl"), "wb") as f:
                pickle.dump(dict(self.keyword_to_chunks), f)
            
            # Save chunk metadata
            with open(os.path.join(temp_dir, "chunks_metadata.pkl"), "wb") as f:
                chunk_data = [{
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                } for chunk in self.chunks]
                pickle.dump(chunk_data, f)
            
            # Update configuration
            config = {
                "version": "Legal_AI_Incremental_v2.1",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": MODEL_NAME,
                "total_chunks": len(self.chunks),
                "chunk_size": CHUNK_SIZE,
                "batch_size": BATCH_SIZE,
                "device_used": "cuda" if torch.cuda.is_available() else "cpu",
                "supported_formats": ["PDF", "HTML", "XML", "TXT"],
                "incremental_mode": True
            }
            
            with open(os.path.join(temp_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
            
            # Atomic move - replace old files with new ones
            import shutil
            for filename in os.listdir(temp_dir):
                src = os.path.join(temp_dir, filename)
                dst = os.path.join(self.index_dir, filename)
                # Create backup of existing file
                if os.path.exists(dst):
                    backup_dst = dst + '.old'
                    if os.path.exists(backup_dst):
                        os.remove(backup_dst)
                    os.rename(dst, backup_dst)
                shutil.move(src, dst)
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
            
            # Remove old backup files
            for filename in os.listdir(self.index_dir):
                if filename.endswith('.old'):
                    os.remove(os.path.join(self.index_dir, filename))
            
            print("✓ Incremental index saved successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error saving index: {e}")
            # Clean up temp directory on failure
            try:
                import shutil
                temp_dir = os.path.join(self.index_dir, "temp_save")
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except:
                pass
            return False

def check_system_capabilities():
    """Check system capabilities and return optimal configuration"""
    import platform
    import psutil
    
    system_info = {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': 0,
        'cuda_memory_gb': 0,
        'faiss_gpu_available': False
    }
    
    if torch.cuda.is_available():
        system_info['cuda_device_count'] = torch.cuda.device_count()
        try:
            system_info['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            pass
        
        # Check FAISS GPU support
        try:
            import faiss
            system_info['faiss_gpu_available'] = hasattr(faiss, 'StandardGpuResources')
        except ImportError:
            pass
    
    return system_info

def print_system_info(system_info):
    """Print detailed system information"""
    print(f"\n🔧 SYSTEM CONFIGURATION:")
    print(f"   Platform: {system_info['platform']}")
    print(f"   Python: {system_info['python_version']}")
    print(f"   CPU Cores: {system_info['cpu_count']}")
    print(f"   RAM: {system_info['memory_gb']:.1f} GB")
    print(f"   CUDA Available: {system_info['cuda_available']}")
    
    if system_info['cuda_available']:
        print(f"   GPU Count: {system_info['cuda_device_count']}")
        print(f"   GPU Memory: {system_info['cuda_memory_gb']:.1f} GB")
        print(f"   FAISS GPU: {'Available' if system_info['faiss_gpu_available'] else 'Not Available'}")
        
        if not system_info['faiss_gpu_available'] and system_info['platform'] == 'Windows':
            print(f"   💡 For FAISS GPU on Windows, use: conda install -c conda-forge faiss-gpu")
    
    print(f"   Optimal Batch Size: {BATCH_SIZE}")

def get_user_batch_size():
    """Get batch size from user input with smart defaults"""
    while True:
        try:
            print("\n" + "="*50)
            print("📊 BATCH PROCESSING CONFIGURATION")
            print("="*50)
            
            # Show system-recommended batch size
            print(f"💡 System-recommended batch size: {BATCH_SIZE} files")
            print("   This is optimized for your hardware configuration.")
            print()
            
            batch_size = input("Enter number of files to process in this batch (or 'all' for all remaining, 'auto' for recommended): ").strip()
            
            if batch_size.lower() == 'all':
                return None  # Process all files
            elif batch_size.lower() == 'auto':
                return BATCH_SIZE  # Use system-recommended size
            
            batch_size = int(batch_size)
            if batch_size <= 0:
                print("❌ Please enter a positive number")
                continue
                
            return batch_size
            
        except ValueError:
            print("❌ Please enter a valid number, 'all', or 'auto'")
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user")
            return None

def main():
    """Main incremental indexing pipeline with enhanced system detection"""
    print("🏛️ ENHANCED LEGAL DOCUMENT AI INDEXING SYSTEM")
    print("=" * 60)
    print("✨ Features: Auto CPU/GPU Detection | Smart Batch Processing | File Tracking")
    print("📁 Supported formats: PDF, HTML, XML, TXT")
    print("=" * 60)
    
    # Check system capabilities
    system_info = check_system_capabilities()
    print_system_info(system_info)
    
    start_time = time.time()
    
    try:
        # Initialize processor with file tracking
        processor = IncrementalLegalProcessor(INDEX_DIR)
        
        # Show registry statistics
        registry_stats = processor.registry.get_registry_stats()
        print(f"\n📊 CURRENT STATUS:")
        print(f"   Previously processed files: {registry_stats['total_processed_files']}")
        print(f"   Previously failed files: {registry_stats['total_failed_files']}")
        print(f"   Training sessions completed: {registry_stats['total_training_sessions']}")
        print(f"   Last training: {registry_stats['last_training_session']}")
        
        # Get batch size from user
        max_files = get_user_batch_size()
        if max_files is not None and max_files == 0:
            print("❌ No files to process")
            return False
        
        # Load documents in batch
        print(f"\n🔄 Starting batch processing...")
        documents, processed_files = processor.load_batch_documents(max_files)
        
        if not documents:
            print("✅ No new documents to process!")
            processor.registry.save_registries()
            return True
        
        # Load existing chunks for incremental processing
        existing_chunks = processor.load_existing_chunk_metadata()
        
        # Create new chunks
        new_chunks = processor.create_chunks(documents)
        if not new_chunks:
            print("❌ No chunks created from new documents")
            return False
        
        # Combine existing and new chunks
        all_chunks = existing_chunks + new_chunks
        print(f"\n📈 Total chunks: {len(all_chunks)} (existing: {len(existing_chunks)}, new: {len(new_chunks)})")
        
        # Initialize embeddings model with optimal device
        print(f"\n🔧 Building incremental search index...")
        device = 'cuda' if system_info['cuda_available'] else 'cpu'
        print(f"🤖 Initializing embeddings model on {device.upper()}...")
        
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=MODEL_NAME,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True, 'batch_size': BATCH_SIZE}
            )
            print(f"✓ Embeddings model loaded successfully on {device.upper()}")
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            # Fallback to CPU if GPU fails
            if device == 'cuda':
                print("🔄 Falling back to CPU...")
                try:
                    embeddings = HuggingFaceEmbeddings(
                        model_name=MODEL_NAME,
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True, 'batch_size': 8}
                    )
                    device = 'cpu'
                    print("✓ Embeddings model loaded successfully on CPU")
                except Exception as e2:
                    print(f"❌ CPU fallback also failed: {e2}")
                    return False
            else:
                return False
        
        # Build or update FAISS index with proper error handling
        print("🏗️ Building/updating FAISS vector database...")
        
        # Check if existing FAISS index exists
        existing_faiss_path = os.path.join(INDEX_DIR, "index.faiss")
        
        try:
            if existing_chunks and os.path.exists(existing_faiss_path):
                print("📚 Loading existing FAISS index...")
                try:
                    # Load existing FAISS index
                    existing_faiss_db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
                    
                    if new_chunks:
                        print("➕ Adding new documents to existing FAISS index...")
                        # Add new documents to existing index
                        new_faiss_db = FAISS.from_documents(new_chunks, embeddings)
                        
                        # Merge indices
                        existing_faiss_db.merge_from(new_faiss_db)
                        faiss_db = existing_faiss_db
                    else:
                        faiss_db = existing_faiss_db
                    
                    print("✓ FAISS index updated successfully")
                except Exception as e:
                    print(f"⚠️ Could not load existing FAISS index: {e}")
                    print("🔄 Rebuilding entire FAISS index...")
                    faiss_db = FAISS.from_documents(all_chunks, embeddings)
            else:
                # Create new FAISS index
                print("🆕 Creating new FAISS index...")
                faiss_db = FAISS.from_documents(all_chunks, embeddings)
            
            print("✓ FAISS index ready")
            
        except Exception as e:
            print(f"❌ FAISS index creation failed: {e}")
            processor.cleanup_memory()
            return False
        
        # Clean up memory after FAISS operations
        processor.cleanup_memory()
        
        # Build incremental hybrid search index
        print("🔗 Building/updating hybrid search components...")
        hybrid_index = IncrementalHybridSearchIndex(INDEX_DIR)
        
        if new_chunks:
            hybrid_index.add_chunks(new_chunks)
        
        print("✓ Hybrid index updated successfully")
        
        # Save updated index with error handling
        success = hybrid_index.save_index(faiss_db)
        if not success:
            print("❌ Failed to save index")
            return False
        
        # Log training session
        session_stats = {
            'files_in_batch': max_files if max_files else "all_remaining",
            'files_processed': processor.processed_files,
            'files_failed': len(processor.failed_files),
            'new_documents': len(documents),
            'new_chunks': len(new_chunks),
            'total_chunks': len(all_chunks),
            'processing_time_seconds': time.time() - start_time,
            'processed_files_list': [os.path.relpath(f) for f in processed_files],
            'device_used': device,
            'batch_size_used': BATCH_SIZE,
            'system_info': system_info
        }
        
        processor.registry.log_training_session(session_stats)
        
        # Save all registries
        processor.registry.save_registries()
        
        # Final memory cleanup
        processor.cleanup_memory()
        
        # Success report
        total_time = time.time() - start_time
        print(f"\n🎯 INCREMENTAL TRAINING COMPLETED!")
        print("=" * 50)
        print(f"⏱️  Total time: {total_time:.1f} seconds")
        print(f"📊 Session Stats:")
        print(f"   Files processed this batch: {processor.processed_files}")
        print(f"   Files failed this batch: {len(processor.failed_files)}")
        print(f"   New documents added: {len(documents)}")
        print(f"   New chunks created: {len(new_chunks)}")
        print(f"   Total chunks in index: {len(all_chunks)}")
        print(f"   Device used: {device.upper()}")
        print(f"   Batch size: {BATCH_SIZE}")
        print(f"   Index location: {INDEX_DIR}")
        
        # Updated registry stats
        updated_stats = processor.registry.get_registry_stats()
        print(f"\n📈 Updated Registry Stats:")
        print(f"   Total processed files: {updated_stats['total_processed_files']}")
        print(f"   Total failed files: {updated_stats['total_failed_files']}")
        print(f"   Total training sessions: {updated_stats['total_training_sessions']}")
        
        # Print failed files report for this batch
        if processor.failed_files:
            print(f"\n❌ Failed Files This Batch ({len(processor.failed_files)}):")
            for i, (filename, reason) in enumerate(processor.failed_files, 1):
                print(f"   {i}. {filename} - {reason}")
        
        # Show files that can be retrained
        failed_registry_path = os.path.join(INDEX_DIR, "failed_files.json")
        if os.path.exists(failed_registry_path):
            try:
                with open(failed_registry_path, 'r') as f:
                    all_failed_files = json.load(f)
                if all_failed_files:
                    print(f"\n📝 Total Failed Files Log: {len(all_failed_files)} files")
                    print(f"   Log location: {failed_registry_path}")
                    print("   💡 Tip: Fix issues and re-run to retry failed files")
            except:
                pass
        
        print(f"\n🚀 Next Steps:")
        print(f"   • Run this script again to process more files in batches")
        print(f"   • Run 'python app.py' to start querying your documents")
        print(f"   • Check failed_files.json for files that need attention")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
        # Save what we have so far
        try:
            processor.registry.save_registries()
            processor.cleanup_memory()
            print("✓ Progress saved - you can resume later")
        except:
            print("⚠️ Could not save progress")
        return False
        
    except Exception as e:
        print(f"\n❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save registries and cleanup even if there's an error
        try:
            processor.registry.save_registries()
            processor.cleanup_memory()
            print("✓ Registries saved and memory cleaned despite error")
        except:
            print("❌ Could not save registries or cleanup memory")
        
        return False

def show_registry_info():
    """Show detailed registry information"""
    print("\n" + "="*60)
    print("📊 DETAILED REGISTRY INFORMATION")
    print("="*60)
    
    registry = FileRegistryManager(INDEX_DIR)
    stats = registry.get_registry_stats()
    
    print(f"📁 Index Directory: {INDEX_DIR}")
    print(f"📈 Total Processed Files: {stats['total_processed_files']}")
    print(f"❌ Total Failed Files: {stats['total_failed_files']}")
    print(f"🎯 Training Sessions: {stats['total_training_sessions']}")
    print(f"🕐 Last Training: {stats['last_training_session']}")
    
    # Show recent processed files
    if registry.processed_files:
        recent_processed = sorted(
            registry.processed_files.items(), 
            key=lambda x: x[1]['processed_timestamp'], 
            reverse=True
        )[:5]
        
        print(f"\n✅ Recent Processed Files (last 5):")
        for file_path, info in recent_processed:
            print(f"   • {Path(file_path).name}")
            print(f"     └ Processed: {info['processed_timestamp']}")
            print(f"     └ Chunks: {info['chunks_created']}, Time: {info['processing_time_seconds']:.2f}s")
    
    # Show recent failed files
    if registry.failed_files:
        recent_failed = sorted(
            registry.failed_files.items(), 
            key=lambda x: x[1]['failed_timestamp'], 
            reverse=True
        )[:5]
        
        print(f"\n❌ Recent Failed Files (last 5):")
        for file_path, info in recent_failed:
            print(f"   • {Path(file_path).name}")
            print(f"     └ Failed: {info['failed_timestamp']}")
            print(f"     └ Reason: {info['error_reason']}")
    
    # Show training sessions with system info
    if registry.training_sessions:
        print(f"\n🎯 Recent Training Sessions:")
        for session in registry.training_sessions[-3:]:  # Last 3 sessions
            stats = session['stats']
            print(f"   Session {session['session_id']} - {session['timestamp']}")
            print(f"     └ Files processed: {stats.get('files_processed', 0)}")
            print(f"     └ New chunks: {stats.get('new_chunks', 0)}")
            print(f"     └ Total chunks: {stats.get('total_chunks', 0)}")
            print(f"     └ Device: {stats.get('device_used', 'unknown')}")
            print(f"     └ Batch size: {stats.get('batch_size_used', 'unknown')}")

if __name__ == "__main__":
    import sys
    
    # Check for info argument
    if len(sys.argv) > 1 and sys.argv[1] == '--info':
        show_registry_info()
    else:
        main()