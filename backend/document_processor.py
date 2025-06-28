# document_processor.py
import PyPDF2
import re
import os
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from pathlib import Path

class DocumentProcessor:
    """Handle document processing tasks including text extraction and chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.logger = logging.getLogger(__name__)
        
        # Configure logging if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from PDF file with better error handling"""
        try:
            # Validate file path
            if not self._validate_file_path(file_path):
                raise FileNotFoundError(f"File not found or not accessible: {file_path}")
            
            text = ""
            with open(file_path, 'rb') as file:
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    # Check if PDF is encrypted
                    if pdf_reader.is_encrypted:
                        self.logger.warning(f"PDF {file_path} is encrypted. Attempting to decrypt...")
                        if not pdf_reader.decrypt(''):
                            raise ValueError("PDF is password-protected and cannot be decrypted")
                    
                    # Extract text from all pages
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                            else:
                                self.logger.warning(f"No text found on page {page_num + 1}")
                        except Exception as e:
                            self.logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                            continue
                    
                except PyPDF2.errors.PdfReadError as e:
                    raise ValueError(f"Invalid or corrupted PDF file: {str(e)}")
                except Exception as e:
                    raise ValueError(f"Error reading PDF: {str(e)}")
            
            # Clean the extracted text
            text = self._clean_text(text)
            
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF. The file might be image-based or corrupted.")
            
            self.logger.info(f"Successfully extracted {len(text)} characters from {file_path}")
            return text
        
        except Exception as e:
            self.logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise
    
    def _validate_file_path(self, file_path: str) -> bool:
        """Validate file path and check if file exists and is readable"""
        try:
            path = Path(file_path)
            if not path.exists():
                self.logger.error(f"File does not exist: {file_path}")
                return False
            
            if not path.is_file():
                self.logger.error(f"Path is not a file: {file_path}")
                return False
            
            if not path.suffix.lower() == '.pdf':
                self.logger.error(f"File is not a PDF: {file_path}")
                return False
            
            if not os.access(file_path, os.R_OK):
                self.logger.error(f"File is not readable: {file_path}")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error validating file path {file_path}: {str(e)}")
            return False
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text with improved handling"""
        if not text:
            return ""
        
        # Remove null characters and other problematic characters
        text = text.replace('\x00', '')
        text = text.replace('\ufeff', '')  # Remove BOM
        
        # Normalize whitespace but preserve paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)  # Limit consecutive newlines
        
        # Fix common PDF extraction issues
        # Handle hyphenated words at line breaks
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        
        # Fix spacing issues between words
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Remove excessive whitespace
        text = re.sub(r' +', ' ', text)
        
        # Clean up line breaks
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:  # Only add non-empty lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def create_chunks(self, text: str, min_chunk_length: int = 50) -> List[str]:
        """Split text into chunks for vector storage with better validation"""
        try:
            if not text or not text.strip():
                raise ValueError("Cannot create chunks from empty text")
            
            # Log text statistics
            self.logger.info(f"Creating chunks from text with {len(text)} characters")
            
            chunks = self.text_splitter.split_text(text)
            
            # Filter out very short chunks and empty chunks
            valid_chunks = []
            for chunk in chunks:
                cleaned_chunk = chunk.strip()
                if len(cleaned_chunk) >= min_chunk_length:
                    valid_chunks.append(cleaned_chunk)
            
            if not valid_chunks:
                # If no valid chunks, try with smaller minimum length
                self.logger.warning(f"No chunks found with minimum length {min_chunk_length}, trying with length 10")
                valid_chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) >= 10]
                
                if not valid_chunks:
                    raise ValueError("No valid chunks could be created from the text")
            
            self.logger.info(f"Created {len(valid_chunks)} valid chunks")
            return valid_chunks
        
        except Exception as e:
            self.logger.error(f"Error creating chunks: {str(e)}")
            raise
    
    def get_document_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF document with better error handling"""
        try:
            if not self._validate_file_path(file_path):
                return {'file_path': file_path, 'error': 'Invalid file path'}
            
            metadata = {
                'file_path': file_path,
                'file_size_bytes': os.path.getsize(file_path),
                'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2)
            }
            
            with open(file_path, 'rb') as file:
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    # Basic metadata
                    metadata['num_pages'] = len(pdf_reader.pages)
                    metadata['is_encrypted'] = pdf_reader.is_encrypted
                    
                    # Document info if available
                    if pdf_reader.metadata:
                        for key, value in pdf_reader.metadata.items():
                            clean_key = key.replace('/', '').lower()
                            metadata[clean_key] = str(value) if value else 'Unknown'
                    
                    # Try to get page size info from first page
                    if pdf_reader.pages:
                        first_page = pdf_reader.pages[0]
                        if hasattr(first_page, 'mediabox'):
                            metadata['page_width'] = float(first_page.mediabox.width)
                            metadata['page_height'] = float(first_page.mediabox.height)
                
                except Exception as e:
                    metadata['extraction_error'] = str(e)
            
            return metadata
        
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
            return {'file_path': file_path, 'error': str(e)}
    
    def extract_text_by_pages(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text page by page for more granular processing"""
        try:
            if not self._validate_file_path(file_path):
                raise FileNotFoundError(f"Invalid file path: {file_path}")
            
            pages_data = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if pdf_reader.is_encrypted and not pdf_reader.decrypt(''):
                    raise ValueError("PDF is password-protected")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        cleaned_text = self._clean_text(text)
                        
                        page_data = {
                            'page_number': page_num + 1,
                            'text': cleaned_text,
                            'char_count': len(cleaned_text),
                            'word_count': len(cleaned_text.split()) if cleaned_text else 0,
                            'has_text': bool(cleaned_text.strip())
                        }
                        
                        # Add page dimensions if available
                        if hasattr(page, 'mediabox'):
                            page_data['width'] = float(page.mediabox.width)
                            page_data['height'] = float(page.mediabox.height)
                        
                        pages_data.append(page_data)
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing page {page_num + 1}: {str(e)}")
                        pages_data.append({
                            'page_number': page_num + 1,
                            'text': '',
                            'char_count': 0,
                            'word_count': 0,
                            'has_text': False,
                            'error': str(e)
                        })
            
            return pages_data
        
        except Exception as e:
            self.logger.error(f"Error extracting text by pages from {file_path}: {str(e)}")
            raise
    
    def validate_document(self, file_path: str) -> Dict[str, Any]:
        """Validate document and return processing recommendations"""
        validation_result = {
            'is_valid': False,
            'file_size_mb': 0,
            'estimated_processing_time': 0,
            'warnings': [],
            'recommendations': [],
            'num_pages': 0
        }
        
        try:
            # Basic file validation
            if not self._validate_file_path(file_path):
                validation_result['warnings'].append("File path validation failed")
                return validation_result
            
            # Check file size
            file_size = os.path.getsize(file_path)
            validation_result['file_size_mb'] = round(file_size / (1024 * 1024), 2)
            
            # Estimate processing time (rough calculation)
            validation_result['estimated_processing_time'] = max(1, int(file_size / 500000))
            
            # Check if file can be opened and read
            with open(file_path, 'rb') as file:
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                    num_pages = len(pdf_reader.pages)
                    validation_result['num_pages'] = num_pages
                    
                    if num_pages == 0:
                        validation_result['warnings'].append("Document has no pages")
                        return validation_result
                    
                    # Check for encryption
                    if pdf_reader.is_encrypted:
                        validation_result['warnings'].append("Document is encrypted")
                        if not pdf_reader.decrypt(''):
                            validation_result['warnings'].append("Document is password-protected")
                            return validation_result
                    
                    # Sample first few pages to check text extraction
                    sample_pages = min(3, num_pages)
                    total_sample_text = ""
                    
                    for i in range(sample_pages):
                        try:
                            page_text = pdf_reader.pages[i].extract_text()
                            total_sample_text += page_text
                        except Exception as e:
                            validation_result['warnings'].append(f"Error reading page {i+1}: {str(e)}")
                    
                    if len(total_sample_text.strip()) < 50:
                        validation_result['warnings'].append("Very little extractable text found in sample pages")
                    
                    # Add recommendations based on document characteristics
                    if num_pages > 100:
                        validation_result['recommendations'].append("Large document - consider increasing chunk size")
                    
                    if validation_result['file_size_mb'] > 10:
                        validation_result['recommendations'].append("Large file - processing may take several minutes")
                    
                    if validation_result['file_size_mb'] > 50:
                        validation_result['recommendations'].append("Very large file - consider processing in batches")
                    
                    validation_result['is_valid'] = True
                    
                except PyPDF2.errors.PdfReadError as e:
                    validation_result['warnings'].append(f"PDF read error: {str(e)}")
                except Exception as e:
                    validation_result['warnings'].append(f"Unexpected error: {str(e)}")
            
            return validation_result
            
        except Exception as e:
            validation_result['warnings'].append(f"Validation error: {str(e)}")
            return validation_result
    
    def get_processing_status(self, file_path: str) -> Dict[str, Any]:
        """Get processing status and recommendations for a document"""
        validation = self.validate_document(file_path)
        
        status = {
            'can_process': validation['is_valid'],
            'file_size_mb': validation['file_size_mb'],
            'num_pages': validation.get('num_pages', 0),
            'estimated_time': validation['estimated_processing_time'],
            'warnings': validation['warnings'],
            'recommendations': validation['recommendations']
        }
        
        return status