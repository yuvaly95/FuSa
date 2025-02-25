import re
from functools import lru_cache
import pandas as pd
from pathlib import Path

#from sentence_transformers import SentenceTransformer

def parse_markdown_to_dataframe(file_path):
    """
    Parses a markdown file to extract clauses with their numbers, headings, and text.
    
    Parameters:
        file_path (str): Path to the markdown file.
    
    Returns:
        pd.DataFrame: A DataFrame containing Clause Number, Heading, and Clause Text.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        
    file_name = Path(file_path).stem
    clauses = []
    current_clause_number = None
    current_heading = None
    current_text = []

    # Regex pattern for clause detection
    clause_pattern = re.compile(r"^#[ \t]+([A-Z\d]{1}[.\d ]+)[ \t]*(.*)")

    for line in lines:
        match = clause_pattern.match(line.strip())

        if match:
            # Store the previous clause
            if current_clause_number:
                clauses.append((file_name, current_clause_number, current_heading, "\n".join(current_text).strip()))

            # Start a new clause
            current_clause_number = match.group(1).strip()
            current_heading = match.group(2).strip() if match.group(2) else None
            current_text = []
        else:
            current_text.append(line.strip())

    # Add the last clause if any
    if current_clause_number:
        clauses.append((current_clause_number, current_heading, "\n".join(current_text).strip()))

    # Convert to DataFrame
    df = pd.DataFrame(clauses, columns=["Document", "Clause Number", "Clause Heading", "Clause Text"])
    return df

def parse_document_structure(file_path):

    # Load the structural markdown file
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.readlines()

    # Extracting relevant parts from the file
    toc_entries = []
    structure_section = False

    for line in content:
        if line.startswith("TOC"):
            parts = line.strip().split(";")
            if len(parts) >= 4:
                document = ''.join(parts[2].split()[:2]).split(':')[0]
                clause_number = parts[2].split()[2].strip()
                heading = parts[3].strip()
                tag = parts[4].strip() if len(parts) > 4 else None
                toc_entries.append((document, clause_number, heading, tag))

    # Create a dataframe from the extracted TOC entries
    toc_df = pd.DataFrame(toc_entries, columns=["Document", "Clause Number", "Human Heading", "Tag"])
    
    return toc_df


# Cache function to store embeddings
@lru_cache(maxsize=100)  # Cache up to 100 entries, adjust as needed
def get_embedding(text, model_name:str ='all-MiniLM-L6-v2', show_progress=True):
    """
    Compute and cache the embedding for a given text.
    """
    model = SentenceTransformer(model_name, trust_remote_code=True)
    embedding = model.encode(text, show_progress_bar=show_progress)
    return tuple(embedding)  # Tuples are hashable and compatible with lru_cache

def get_embeddings_batch(texts, model_name:str ='all-MiniLM-L6-v2', batch_size=16):
    model = SentenceTransformer(model_name, trust_remote_code=True)
    #model = model.to('cuda')
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, device='cuda')