import re
from functools import lru_cache
import pandas as pd
from sentence_transformers import SentenceTransformer


def parse_markdown_to_dataframe(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data = []
    current_number, current_heading, current_clause = None, None, []

    for line in map(str.strip, lines):  # Strip whitespace as lines are read
        if line.startswith("#"):  # Check if the line is a heading
            if current_number:  # Save the current clause if there's an active heading
                data.append((current_number, current_heading or "", " ".join(current_clause)))
            # Parse clause number and heading
            match = re.match(r'#\s*([\d\.]+)\s*(.*)', line)
            if match:
                current_number, current_heading = match.group(1), (match.group(2) or "").strip() if match else (None, None)
                current_clause = []  # Reset for new clause
        elif line:  # Add non-heading, non-empty lines to the current clause
            current_clause.append(line)

    # Add the last clause if any
    if current_number:
        data.append((current_number, current_heading or "", " ".join(current_clause)))

    # Create and return the DataFrame
    return pd.DataFrame(data, columns=["Number", "Heading", "Clause"])


# Cache function to store embeddings
@lru_cache(maxsize=100)  # Cache up to 100 entries, adjust as needed
def get_embedding(text, model_name:str ='all-MiniLM-L6-v2', show_progress=True):
    """
    Compute and cache the embedding for a given text.
    """
    model = SentenceTransformer(model_name, trust_remote_code=True)
    embedding = model.encode(text, show_progress_bar=show_progress)
    return tuple(embedding)  # Tuples are hashable and compatible with lru_cache