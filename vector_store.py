"""
Vector Store Management for GL Account Documentation
Persistent ChromaDB for semantic search over GL policies and historical patterns
"""

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pathlib import Path
from typing import List, Optional
import logging
import os
import pandas as pd
import time

from config import Config

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage persistent vector store for GL documentation"""
    
    def __init__(self, cost_tracker=None):
        self.persist_dir = Config.CHROMA_PERSIST_DIR
        self.collection_name = Config.CHROMA_COLLECTION_NAME
        self.embedding_model = Config.EMBEDDING_MODEL
        self.cost_tracker = cost_tracker
        
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        # Load or create vector store
        if Path(self.persist_dir).exists():
            logger.info(f"📂 Loading existing vector store from {self.persist_dir}")
            self.vectorstore = self._load_existing()
        else:
            logger.info(f"🆕 Creating new vector store at {self.persist_dir}")
            self.vectorstore = self._create_new()
    
    def _load_existing(self):
        """Load existing persistent vector store"""
        return Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
    
    def _create_new(self):
        """Create new vector store and index GL documentation"""
        # Check if GL documentation directory exists
        if not Config.GL_DOCS_DIR.exists():
            logger.warning(f"⚠️  GL documentation directory not found: {Config.GL_DOCS_DIR}")
            logger.info("Creating empty vector store. Add GL docs and run rebuild_index()")
            
            # Create empty store
            return Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
        
        # Load and index GL documentation
        return self.rebuild_index()
    
    def rebuild_index(self):
        """Rebuild vector store from GL documentation directory"""
        logger.info("🔨 Rebuilding vector store index...")
        
        documents = []
        
        # Load all documentation files
        for filepath in Config.GL_DOCS_DIR.glob("**/*"):
            if filepath.suffix in ['.md', '.txt']:
                # Extract account ID from filename (e.g., "5000_rent_expense.md")
                account_id = filepath.stem.split('_')[0]
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                documents.append(Document(
                    page_content=content,
                    metadata={
                        'account_id': account_id,
                        'source_file': filepath.name
                    }
                ))
        
        if not documents:
            logger.warning("⚠️  No GL documentation files found")
            return Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        
        splits = text_splitter.split_documents(documents)
        
        logger.info(f"📄 Processing {len(documents)} documents into {len(splits)} chunks")
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir),
            collection_name=self.collection_name
        )
        
        logger.info(f"✅ Vector store built: {len(splits)} chunks indexed")
        
        return vectorstore
    
    def similarity_search(self, query: str, account_id: str = None, k: int = 3) -> List[Document]:
        """
        Semantic search over GL documentation
        
        Args:
            query: Search query
            account_id: Filter by specific GL account (optional)
            k: Number of results to return
        
        Returns:
            List of relevant Document objects
        """
        try:
            # Track embedding cost
            start_time = time.time()
            
            if account_id:
                # Filter by account ID
                docs = self.vectorstore.similarity_search(
                    query,
                    k=k,
                    filter={"account_id": account_id}
                )
            else:
                docs = self.vectorstore.similarity_search(query, k=k)
            
            call_duration = time.time() - start_time
            
            # Track cost if tracker is available
            if self.cost_tracker:
                # Estimate tokens for embedding (query length / 4)
                input_tokens = len(query) // 4
                self.cost_tracker.track_embedding_call(
                    model=self.embedding_model,
                    agent="retrieval",
                    input_tokens=input_tokens,
                    call_duration=call_duration,
                    query_preview=query[:100]
                )
            
            logger.info(f"🔍 Found {len(docs)} relevant documents for query")
            return docs
            
        except Exception as e:
            logger.error(f"❌ Vector search failed: {e}")
            return []
    
    def add_document(self, content: str, account_id: str, source_file: str):
        """Add new documentation to vector store"""
        doc = Document(
            page_content=content,
            metadata={
                'account_id': account_id,
                'source_file': source_file
            }
        )
        
        # Split if large
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents([doc])
        
        # Add to existing store
        self.vectorstore.add_documents(splits)
        
        logger.info(f"✅ Added {len(splits)} chunks for account {account_id}")
    
    def search_similar_anomalies(self, anomaly_description: str, k: int = 5) -> List[str]:
        """Search for similar past anomaly patterns"""
        query = f"Historical anomaly: {anomaly_description}"
        docs = self.similarity_search(query, k=k)
        
        return [d.page_content for d in docs]


def build_initial_gl_knowledge_base(gl_accounts_df: pd.DataFrame):
    """
    Build initial GL documentation from GL master CSV
    Creates basic documentation files if none exist
    """
    gl_docs_dir = Config.GL_DOCS_DIR
    gl_docs_dir.mkdir(parents=True, exist_ok=True)
    
    for _, account in gl_accounts_df.iterrows():
        filename = f"{account['account_id']}_{account['account_name'].lower().replace(' ', '_')}.md"
        filepath = gl_docs_dir / filename
        
        if not filepath.exists():
            # Generate basic documentation
            content = f"""# {account['account_name']} (GL {account['account_id']})

## Account Details
- **Category**: {account['category']}
- **Typical Monthly Range**: ${account.get('typical_min', 'N/A'):,.2f} - ${account.get('typical_max', 'N/A'):,.2f}

## Description
{account.get('description', 'No description available')}

## Typical Causes for Variance
- Normal business fluctuations
- Seasonal patterns
- One-time expenses or adjustments

## Past Anomalies
(Will be updated as anomalies are detected and investigated)
"""
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            logger.info(f"📝 Created GL documentation: {filename}")
    
    logger.info(f"✅ GL knowledge base initialized in {gl_docs_dir}")


if __name__ == "__main__":
    # Test vector store
    logging.basicConfig(level=logging.INFO)
    
    manager = VectorStoreManager()
    
    # Test search
    results = manager.similarity_search("Why would rent expense increase?", k=3)
    print(f"Found {len(results)} results")
    for doc in results:
        print(f"- {doc.metadata}: {doc.page_content[:100]}...")

