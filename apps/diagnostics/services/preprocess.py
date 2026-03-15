VALID_NUCLEOTIDES = set('ACGTN')


def extract_sequence_from_text(raw_text: str) -> str:
    """Extract sequence text from plain DNA or FASTA/FNA content."""
    lines = raw_text.splitlines()
    sequence_lines = [line.strip() for line in lines if line and not line.startswith('>')]
    return ''.join(sequence_lines)


def clean_sequence(raw_sequence: str) -> str:
    """Normalize to uppercase and remove non-DNA characters."""
    return ''.join(char for char in raw_sequence.upper() if char in VALID_NUCLEOTIDES)
