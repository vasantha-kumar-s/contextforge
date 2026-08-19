import os
import fitz  # PyMuPDF
from typing import List, Dict, Any

class DocumentExtractor:
    """Extracts clean text and metadata from PDF, TXT, and Markdown files using PyMuPDF block parsing."""

    @staticmethod
    def extract_from_file(file_path: str, filename: str) -> List[Dict[str, Any]]:
        """
        Extracts pages from a file maintaining clean reading order.
        Returns: [{'page_number': int, 'text': str, 'filename': str}]
        """
        ext = os.path.splitext(filename)[1].lower()
        pages = []

        if ext == ".pdf":
            try:
                doc = fitz.open(file_path)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    # Block extraction sorts text in reading order (handles 2-column papers clean)
                    blocks = page.get_text("blocks")
                    block_texts = []
                    for b in blocks:
                        # b[4] is the text content of block
                        txt = b[4].strip()
                        if txt and len(txt) > 10:  # Skip tiny header/footer noise
                            block_texts.append(txt)

                    text = "\n\n".join(block_texts)
                    if text:
                        pages.append({
                            "filename": filename,
                            "page_number": page_num + 1,
                            "text": text
                        })
                doc.close()
            except Exception as e:
                # Fallback if text-based spec file was saved with .pdf extension
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                    if text:
                        pages.append({
                            "filename": filename,
                            "page_number": 1,
                            "text": text
                        })
        elif ext in [".txt", ".md", ".json"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
                if text:
                    pages.append({
                        "filename": filename,
                        "page_number": 1,
                        "text": text
                    })
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        return pages
