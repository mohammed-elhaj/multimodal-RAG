"""Data processing components for the Ashraq RAG Agent."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles JSON data loading and preprocessing for the RAG system."""
    
    def __init__(self):
        """Initialize the DataProcessor."""
        pass
    
    def load_source_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load and validate data from JSON file.
        
        Args:
            file_path: Path to the JSON data file
            
        Returns:
            List of document dictionaries
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
            ValueError: If the data structure is invalid
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate data structure
            if not isinstance(data, list):
                raise ValueError("Data must be a list of documents")
            
            # Validate each document has required fields
            required_fields = {'page_number', 'slide_title', 'content_type', 'content', 'image_path'}
            for i, doc in enumerate(data):
                if not isinstance(doc, dict):
                    raise ValueError(f"Document {i} must be a dictionary")
                
                missing_fields = required_fields - set(doc.keys())
                if missing_fields:
                    raise ValueError(f"Document {i} missing required fields: {missing_fields}")
                
                # Validate page_number is an integer
                if not isinstance(doc['page_number'], int):
                    raise ValueError(f"Document {i}: page_number must be an integer")
            
            logger.info(f"Successfully loaded {len(data)} documents from {file_path}")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def prepare_documents(self, data: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """
        Prepare documents for embedding and storage.
        
        Args:
            data: List of document dictionaries from load_source_data
            
        Returns:
            Tuple of (documents, metadatas, ids) where:
            - documents: List of content strings for embedding
            - metadatas: List of complete document dictionaries
            - ids: List of string-based page IDs (e.g., "page_1")
        """
        if not data:
            logger.warning("No data provided to prepare_documents")
            return [], [], []
        
        documents = []
        metadatas = []
        ids = []
        
        for doc in data:
            try:
                # Extract content for embedding
                content = doc['content']
                documents.append(content)
                
                # Store complete document as metadata
                metadatas.append(doc.copy())
                
                # Generate string-based ID
                page_id = f"page_{doc['page_number']}"
                ids.append(page_id)
                
            except KeyError as e:
                logger.error(f"Missing required field in document: {e}")
                raise ValueError(f"Document missing required field: {e}")
        
        logger.info(f"Prepared {len(documents)} documents for indexing")
        return documents, metadatas, ids
    
    def validate_document_structure(self, doc: Dict[str, Any]) -> bool:
        """
        Validate that a document has the expected structure.
        
        Args:
            doc: Document dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = {'page_number', 'slide_title', 'content_type', 'content', 'image_path'}
        
        if not isinstance(doc, dict):
            return False
        
        # Check required fields exist
        if not all(field in doc for field in required_fields):
            return False
        
        # Check field types
        if not isinstance(doc['page_number'], int):
            return False
        
        if not all(isinstance(doc[field], str) for field in ['slide_title', 'content_type', 'content', 'image_path']):
            return False
        
        return True
    
    def enhance_pdf_generated_data(self, data: List[Dict[str, Any]], 
                                  content_enhancements: Optional[Dict[int, str]] = None) -> List[Dict[str, Any]]:
        """
        Enhance PDF-generated data with better content descriptions.
        
        Args:
            data: List of document dictionaries from PDF processing
            content_enhancements: Optional dictionary mapping page numbers to enhanced descriptions
            
        Returns:
            Enhanced document data
        """
        enhanced_data = []
        
        for doc in data:
            enhanced_doc = doc.copy()
            
            page_num = doc.get('page_number', 0)
            
            # Apply content enhancements if provided
            if content_enhancements and page_num in content_enhancements:
                enhanced_doc['content'] = content_enhancements[page_num]
            
            # Improve slide titles for PDF-generated content
            if doc.get('content_type') == 'extracted_page':
                # Try to extract better title from content if possible
                content = doc.get('content', '')
                if 'title' in content.lower() or 'strategy' in content.lower():
                    enhanced_doc['content_type'] = 'title_page'
                elif 'diagram' in content.lower() or 'chart' in content.lower():
                    enhanced_doc['content_type'] = 'diagram'
                elif 'table' in content.lower() or 'entities' in content.lower():
                    enhanced_doc['content_type'] = 'table'
                elif 'list' in content.lower() or 'findings' in content.lower():
                    enhanced_doc['content_type'] = 'list'
            
            enhanced_data.append(enhanced_doc)
        
        logger.info(f"Enhanced {len(enhanced_data)} PDF-generated documents")
        return enhanced_data
    
    def get_document_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the loaded documents.
        
        Args:
            data: List of document dictionaries
            
        Returns:
            Dictionary with statistics
        """
        if not data:
            return {
                'total_documents': 0,
                'content_types': {},
                'avg_content_length': 0,
                'page_range': None
            }
        
        content_types = {}
        content_lengths = []
        page_numbers = []
        
        for doc in data:
            # Count content types
            content_type = doc.get('content_type', 'unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
            
            # Track content lengths
            content_lengths.append(len(doc.get('content', '')))
            
            # Track page numbers
            page_numbers.append(doc.get('page_number', 0))
        
        return {
            'total_documents': len(data),
            'content_types': content_types,
            'avg_content_length': sum(content_lengths) / len(content_lengths) if content_lengths else 0,
            'page_range': (min(page_numbers), max(page_numbers)) if page_numbers else None
        }