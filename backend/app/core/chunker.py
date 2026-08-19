import re
from typing import List, Dict, Any

class RecursiveTextChunker:
    """
    Splits text recursively using paragraphs, sentences, and spaces.
    Optimized for academic papers and technical specs.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split_text_with_separators(self, text: str, separators: List[str]) -> List[str]:
        if not text:
            return []
        
        if len(text) <= self.chunk_size or not separators:
            return [text]

        separator = separators[0]
        new_separators = separators[1:]

        if separator == "":
            splits = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
            return splits

        if separator in ["\n\n", "\n"]:
            parts = text.split(separator)
        else:
            parts = [p + separator.strip() for p in text.split(separator) if p]

        chunks = []
        current_chunk = ""

        for part in parts:
            if len(part) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                sub_parts = self._split_text_with_separators(part, new_separators)
                chunks.extend(sub_parts)
            elif len(current_chunk) + len(part) + len(separator) <= self.chunk_size:
                current_chunk = current_chunk + (separator if current_chunk else "") + part
            else:
                chunks.append(current_chunk.strip())
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + (separator if overlap_text else "") + part

        if current_chunk:
            chunks.append(current_chunk.strip())

        return [c for c in chunks if c]

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []
        global_chunk_idx = 0

        for page in pages:
            raw_chunks = self._split_text_with_separators(page["text"], self.separators)
            for idx, chunk_text in enumerate(raw_chunks):
                if not chunk_text.strip():
                    continue
                global_chunk_idx += 1
                all_chunks.append({
                    "chunk_id": f"{page['filename']}_p{page['page_number']}_c{idx + 1}_{global_chunk_idx}",
                    "filename": page["filename"],
                    "page_number": page["page_number"],
                    "chunk_index": idx + 1,
                    "global_index": global_chunk_idx,
                    "text": chunk_text.strip(),
                    "char_count": len(chunk_text.strip())
                })

        return all_chunks
