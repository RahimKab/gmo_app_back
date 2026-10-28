VALID_NUCLEOTIDES = set('ACGTN')
SUPPORTED_SEQUENCE_EXTENSIONS = ('.fasta', '.fna', '.fa', '.txt')


def extract_sequences_from_text(raw_text: str) -> list[str]:
    """Extract one cleaned sequence per FASTA entry or pasted sequence block."""
    normalized_text = raw_text.replace('\r\n', '\n').replace('\r', '\n').replace('>', '\n>')
    sequences = []
    sequence_lines = []

    for line in normalized_text.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            continue

        if stripped_line.startswith('>'):
            if sequence_lines:
                cleaned_sequence = clean_sequence(''.join(sequence_lines))
                if cleaned_sequence:
                    sequences.append(cleaned_sequence)
                sequence_lines = []
            continue

        sequence_lines.append(stripped_line)

    if sequence_lines:
        cleaned_sequence = clean_sequence(''.join(sequence_lines))
        if cleaned_sequence:
            sequences.append(cleaned_sequence)

    return sequences


def is_supported_sequence_filename(filename: str) -> bool:
    """Return whether a file has a supported FASTA or text extension."""
    return filename.lower().endswith(SUPPORTED_SEQUENCE_EXTENSIONS)


def clean_sequence(raw_sequence: str) -> str:
    """Normalize to uppercase and remove non-DNA characters."""
    return ''.join(char for char in raw_sequence.upper() if char in VALID_NUCLEOTIDES)
