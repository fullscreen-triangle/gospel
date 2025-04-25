"""
Text extraction from PDF files for knowledge base creation.
"""

import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import fitz  # PyMuPDF
import numpy as np
from tqdm import tqdm

from gospel.utils.gene_utils import find_gene_mentions


@dataclass
class TextChunk:
    """A chunk of text extracted from a document."""
    id: str
    text: str
    page: int
    position: Dict = field(default_factory=dict)
    genes: List[str] = field(default_factory=list)


@dataclass
class Document:
    """A document containing text chunks."""
    source: str
    chunks: List[TextChunk] = field(default_factory=list)


class PdfTextExtractor:
    """Extract text from PDF files for knowledge base creation."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize a PDF text extractor.

        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_from_file(self, pdf_path: str) -> Document:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Document containing extracted text
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        document = Document(source=pdf_path)
        
        try:
            # Open PDF
            with fitz.open(pdf_path) as pdf:
                # Process each page
                for page_idx, page in enumerate(pdf):
                    text = page.get_text()
                    
                    # Skip empty pages
                    if not text.strip():
                        continue
                    
                    # Create chunks
                    chunks = self._create_chunks(text, page_idx)
                    
                    # Add chunks to document
                    document.chunks.extend(chunks)
                    
            # Identify genes in chunks
            self._identify_genes(document)
            
            return document
        
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return document

    def extract_from_directory(self, directory: str) -> List[Document]:
        """
        Extract text from all PDF files in a directory.

        Args:
            directory: Directory containing PDF files

        Returns:
            List of documents
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(".pdf")]
        
        if not pdf_files:
            print(f"No PDF files found in {directory}")
            return []
        
        documents = []
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDF files"):
            try:
                doc = self.extract_from_file(pdf_file)
                documents.append(doc)
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
        
        return documents

    def _create_chunks(self, text: str, page_idx: int) -> List[TextChunk]:
        """
        Create chunks from text.

        Args:
            text: Text to chunk
            page_idx: Page index

        Returns:
            List of text chunks
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        
        # Initialize chunks
        chunks = []
        
        # Split text into chunks with overlap
        start = 0
        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))
            
            # If not at the end, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence boundary within the last 20% of the chunk
                boundary_search_start = max(end - int(0.2 * self.chunk_size), start)
                
                # Find the last sentence boundary in this region
                boundary_matches = list(re.finditer(r'[.!?]\s+', text[boundary_search_start:end]))
                if boundary_matches:
                    # Adjust end to the last sentence boundary
                    end = boundary_search_start + boundary_matches[-1].end()
            
            # Create chunk
            chunk_text = text[start:end].strip()
            if chunk_text:  # Skip empty chunks
                chunk = TextChunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    page=page_idx,
                    position={"start": start, "end": end}
                )
                chunks.append(chunk)
            
            # Move start position for next chunk
            start = end - self.chunk_overlap
            
            # Ensure we make progress
            if start >= end:
                start = end
        
        return chunks

    def _identify_genes(self, document: Document) -> None:
        """
        Identify genes in document chunks.

        Args:
            document: Document to process
        """
        for chunk in document.chunks:
            genes = find_gene_mentions(chunk.text)
            chunk.genes = genes
