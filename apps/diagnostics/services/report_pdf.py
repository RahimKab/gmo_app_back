import datetime
import io
from pathlib import Path
from typing import Any, Dict, List

from django.conf import settings


def _escape_pdf_text(value: str) -> str:
    return value.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _pdf_text(
    commands: List[str],
    x: float,
    y: float,
    text: str,
    font: str = 'F1',
    size: int = 11,
    color: tuple = (0.08, 0.13, 0.24),
) -> None:
    safe = _escape_pdf_text(text)
    r, g, b = color
    commands.append('BT')
    commands.append(f'/{font} {size} Tf')
    commands.append(f'{r:.3f} {g:.3f} {b:.3f} rg')
    commands.append(f'1 0 0 1 {x:.2f} {y:.2f} Tm')
    commands.append(f'({safe}) Tj')
    commands.append('ET')


def _pdf_rect(
    commands: List[str],
    x: float,
    y: float,
    width: float,
    height: float,
    fill: tuple,
    stroke: tuple | None = None,
    line_width: float = 1.0,
) -> None:
    fr, fg, fb = fill
    commands.append(f'{fr:.3f} {fg:.3f} {fb:.3f} rg')
    if stroke is not None:
        sr, sg, sb = stroke
        commands.append(f'{sr:.3f} {sg:.3f} {sb:.3f} RG')
        commands.append(f'{line_width:.2f} w')
        commands.append(f'{x:.2f} {y:.2f} {width:.2f} {height:.2f} re B')
    else:
        commands.append(f'{x:.2f} {y:.2f} {width:.2f} {height:.2f} re f')


def build_report_pdf_bytes(payload: Dict[str, Any]) -> bytes:
    sequence_name = str(payload.get('sequence_name', 'Séquence inconnue'))
    seq_id = str(payload.get('seq_id', 'N/A'))
    predicted_label = str(payload.get('predicted_label', 'unknown'))
    score = float(payload.get('score', 0.0))
    gmo = float(payload.get('gmo_probability', 0.0))
    no_gmo = float(payload.get('no_gmo_probability', 0.0))
    chunks_processed = int(payload.get('chunks_processed', 1))
    attention: List[Dict[str, Any]] = payload.get('attention', []) or []
    sequence_cleaned = str(payload.get('sequence_cleaned', ''))

    gmo = _clamp(gmo, 0.0, 1.0)
    no_gmo = _clamp(no_gmo, 0.0, 1.0)
    score = _clamp(score, 0.0, 1.0)

    now = datetime.datetime.utcnow().strftime('%d-%m-%Y %H:%M')

    parsed_attention: List[Dict[str, Any]] = []
    for item in attention[:6]:
        start = int(item.get('start', 0))
        end = int(item.get('end', 0))
        att_score = _clamp(float(item.get('score', 0.0)), 0.0, 1.0)
        segment = str(item.get('sequence', ''))
        if end > start:
            parsed_attention.append(
                {
                    'start': start,
                    'end': end,
                    'score': att_score,
                    'segment': segment,
                }
            )

    page_width = 595
    commands: List[str] = []

    # en tete.
    _pdf_rect(commands, 0, 785, page_width, 57, fill=(0.06, 0.22, 0.48))
    _pdf_text(commands, 42, 815, 'Detection OGM - Rapport Diagnostique', font='F2', size=18, color=(1, 1, 1))
    _pdf_text(commands, 42, 798, f'Generer: {now}', font='F1', size=10, color=(0.82, 0.9, 1))

    # Resume.
    _pdf_rect(commands, 38, 680, 519, 93, fill=(0.96, 0.98, 1.0), stroke=(0.7, 0.79, 0.92))
    _pdf_text(commands, 52, 748, f'Sequence: {sequence_name}', font='F2', size=12)
    _pdf_text(commands, 52, 731, f'Sequence ID: {seq_id}', size=10, color=(0.23, 0.3, 0.42))
    _pdf_text(commands, 52, 714, f'Morceaux: {chunks_processed}', size=10, color=(0.23, 0.3, 0.42))

    label_text = 'OGM' if predicted_label == 'gmo' else 'NON OGM'
    label_color = (0.87, 0.23, 0.33) if predicted_label == 'gmo' else (0.18, 0.61, 0.34)
    _pdf_rect(commands, 368, 716, 172, 39, fill=(0.99, 0.99, 0.99), stroke=(0.84, 0.87, 0.92))
    _pdf_text(commands, 382, 742, 'Resultat', size=9, color=(0.34, 0.41, 0.54))
    _pdf_text(commands, 382, 724, label_text, font='F2', size=14, color=label_color)
    _pdf_text(commands, 469, 724, f'{score * 100:.1f}%', font='F2', size=12, color=(0.09, 0.15, 0.26))

    # Graphe de confiance
    _pdf_rect(commands, 38, 540, 254, 126, fill=(0.98, 0.99, 1), stroke=(0.72, 0.82, 0.93))
    _pdf_text(commands, 52, 645, 'Graphe de confiance', font='F2', size=12)

    graph_x = 52
    graph_width = 200
    _pdf_text(commands, graph_x, 620, 'GMO', size=10)
    _pdf_rect(commands, graph_x, 607, graph_width, 10, fill=(0.9, 0.92, 0.96))
    _pdf_rect(commands, graph_x, 607, graph_width * gmo, 10, fill=(0.92, 0.29, 0.4))
    _pdf_text(commands, graph_x + graph_width + 6, 608, f'{gmo * 100:.1f}%', size=9)

    _pdf_text(commands, graph_x, 585, 'Non-GMO', size=10)
    _pdf_rect(commands, graph_x, 572, graph_width, 10, fill=(0.9, 0.92, 0.96))
    _pdf_rect(commands, graph_x, 572, graph_width * no_gmo, 10, fill=(0.29, 0.75, 0.45))
    _pdf_text(commands, graph_x + graph_width + 6, 573, f'{no_gmo * 100:.1f}%', size=9)

    # Graphe d'attention
    _pdf_rect(commands, 303, 540, 254, 126, fill=(0.98, 0.99, 1), stroke=(0.72, 0.82, 0.93))
    _pdf_text(commands, 317, 645, 'Graphe d\'attention', font='F2', size=12)

    attention_x = 317
    attention_width = 150
    row_top = 624
    for idx, item in enumerate(parsed_attention[:5], start=0):
        y = row_top - (idx * 20)
        seg_label = f"{item['start']}-{item['end']}"
        seg_score = float(item['score'])
        _pdf_text(commands, attention_x, y, seg_label, size=8, color=(0.28, 0.36, 0.5))
        _pdf_rect(commands, attention_x + 55, y - 1, attention_width, 8, fill=(0.9, 0.92, 0.96))
        _pdf_rect(commands, attention_x + 55, y - 1, attention_width * seg_score, 8, fill=(0.95, 0.68, 0.24))
        _pdf_text(commands, attention_x + 55 + attention_width + 4, y - 1, f'{seg_score * 100:.1f}%', size=8)

    # Aperçu de la séquence avec mise en évidence des intervalles d'attention.
    _pdf_rect(commands, 38, 18, 519, 512, fill=(0.99, 1.0, 0.99), stroke=(0.72, 0.85, 0.74))
    _pdf_text(commands, 52, 518, 'Vue de la sequence', font='F2', size=11)

    if not sequence_cleaned:
        _pdf_text(commands, 52, 498, 'La sequence est inutilisable.', size=10, color=(0.32, 0.38, 0.48))
    else:
        # Police Courier 8 pt : largeur d'avance = 600/1000 * 8 = 4.8 pt par caractère (exact)
        chars_per_line = 90
        char_width = 4.8          # Courier à pas fixe à 8 pt
        font_size = 8
        line_step = 14            # espacement entre lignes (baseline à baseline) en pt
        seq_x = 96                # position x de départ des caractères de la séquence
        gutter_x = 52             # position x de la gouttière des numéros de ligne
        first_line_y = 498
        max_lines = 35            # (498 - 22) / 14 ≈ 34 lignes tiennent dans le panneau

        preview = sequence_cleaned[: chars_per_line * max_lines]

        # --- Passe 1 : collecte des rectangles de surbrillance ---
        highlight_rects = []
        lines_data = []
        for line_idx in range(max_lines):
            line_start = line_idx * chars_per_line
            if line_start >= len(preview):
                break
            line_end = min(line_start + chars_per_line, len(preview))
            line_text = preview[line_start:line_end]
            line_y = first_line_y - line_idx * line_step
            lines_data.append((line_start, line_y, line_text))

            for item in parsed_attention:
                ol_start = max(int(item['start']), line_start)
                ol_end = min(int(item['end']), line_end)
                if ol_end <= ol_start:
                    continue
                hx = seq_x + (ol_start - line_start) * char_width
                hw = (ol_end - ol_start) * char_width
                # bas = baseline - 2 (descente), hauteur = font_size + 2
                highlight_rects.append((hx, line_y - 2, hw, font_size + 2))

        # --- Passe 2 : dessin de tous les fonds de surbrillance en premier ---
        for hx, hy, hw, hh in highlight_rects:
            _pdf_rect(commands, hx, hy, hw, hh, fill=(1.0, 0.91, 0.55))

        # --- Passe 3 : dessin de tout le texte par-dessus ---
        for line_start, line_y, line_text in lines_data:
            _pdf_text(commands, gutter_x, line_y, f'{line_start:04d}', font='F1', size=font_size, color=(0.52, 0.57, 0.65))
            _pdf_text(commands, seq_x, line_y, line_text, font='F3', size=font_size, color=(0.06, 0.1, 0.16))

        if len(sequence_cleaned) > len(preview) and lines_data:
            truncation_y = lines_data[-1][1] - line_step
            _pdf_text(commands, gutter_x, truncation_y, '... sequence tronquee ...', size=8, color=(0.4, 0.46, 0.56))

    content_bytes = ('\n'.join(commands) + '\n').encode('latin-1', errors='replace')

    objects: List[bytes] = []
    objects.append(b'1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n')
    objects.append(b'2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n')
    objects.append(
        b'3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] '
        b'/Resources << /Font << /F1 4 0 R /F2 5 0 R /F3 6 0 R >> >> /Contents 7 0 R >>\nendobj\n'
    )
    objects.append(b'4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n')
    objects.append(b'5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n')
    objects.append(b'6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>\nendobj\n')
    objects.append(
        f'7 0 obj\n<< /Length {len(content_bytes)} >>\nstream\n'.encode('ascii')
        + content_bytes
        + b'endstream\nendobj\n'
    )

    output = io.BytesIO()
    output.write(b'%PDF-1.4\n')
    offsets = [0]
    for obj in objects:
        offsets.append(output.tell())
        output.write(obj)

    xref_start = output.tell()
    output.write(f'xref\n0 {len(objects) + 1}\n'.encode('ascii'))
    output.write(b'0000000000 65535 f \n')
    for offset in offsets[1:]:
        output.write(f'{offset:010d} 00000 n \n'.encode('ascii'))

    output.write(
        (
            'trailer\n'
            f'<< /Size {len(objects) + 1} /Root 1 0 R >>\n'
            'startxref\n'
            f'{xref_start}\n'
            '%%EOF\n'
        ).encode('ascii')
    )
    return output.getvalue()


def save_pdf_report_file(file_name: str, pdf_bytes: bytes) -> str:
    media_root = Path(getattr(settings, 'MEDIA_ROOT', settings.BASE_DIR / 'media'))
    report_dir = media_root / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)

    file_path = report_dir / file_name
    file_path.write_bytes(pdf_bytes)

    media_url = getattr(settings, 'MEDIA_URL', '/media/')
    return f"{media_url.rstrip('/')}/reports/{file_name}"
