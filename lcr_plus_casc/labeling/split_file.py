"""Read/write helpers for the TSV artifacts produced by the labeling pipeline."""


def read_rows(path, detect_header=True):
    """Read a TSV file and return a list of rows (each a list of cell strings).

    A leading header row (first cell 'sentence' or any '*_score' cell) is
    skipped when ``detect_header`` is true.
    """
    with open(path, encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f if line.strip()]
    if detect_header and lines and _is_header(lines[0]):
        lines = lines[1:]
    return [line.split('\t') for line in lines]


def _is_header(line):
    cells = line.split('\t')
    return cells[0] == 'sentence' or any(cell.endswith(('_score', '_word')) for cell in cells)


def read_labeled(path):
    """Read a labeled-data file, returning a list of (idx, cat, pol, sentence).

    Auto-detects the format: if the first non-empty line contains a tab the
    file is TSV (``idx<TAB>cat<TAB>pol<TAB>sentence``), otherwise the legacy
    alternating layout (one sentence per even line, one "cat pol" per odd line).
    """
    with open(path, encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f if line.strip()]

    if not lines or '\t' not in lines[0]:
        labeled = []
        for i in range(0, len(lines) - 1, 2):
            sentence = lines[i]
            cat, pol = lines[i + 1].split()
            labeled.append((i // 2, cat, pol, sentence))
        return labeled

    return [
        (row[0], row[1], row[2], row[3])
        for row in (line.split('\t') for line in lines)
    ]


def write_labeled(path, labeled):
    """Write labels as TSV rows of ``idx<TAB>cat<TAB>pol<TAB>sentence``."""
    with open(path, 'w', encoding='utf-8') as f:
        for idx, cat, pol, sentence in labeled:
            f.write(f'{idx}\t{cat}\t{pol}\t{sentence}\n')
