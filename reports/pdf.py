"""Generacion de informes PDF con reportlab."""

import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image, PageBreak, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from reports.text import format_result_text, generate_interpretation, _effect_label, _eta_label


# --- Estilos -----------------------------------------------------------------

def _get_styles():
    """Retorna estilos personalizados para el PDF."""
    base = getSampleStyleSheet()
    styles = {}
    styles['title'] = ParagraphStyle(
        'CustomTitle', parent=base['Title'],
        fontSize=28, textColor=colors.HexColor('#1a365d'),
        spaceAfter=6 * mm,
    )
    styles['subtitle'] = ParagraphStyle(
        'CustomSubtitle', parent=base['Normal'],
        fontSize=14, textColor=colors.HexColor('#4a5568'),
        alignment=TA_CENTER, spaceAfter=4 * mm,
    )
    styles['meta'] = ParagraphStyle(
        'Meta', parent=base['Normal'],
        fontSize=10, textColor=colors.HexColor('#718096'),
        alignment=TA_CENTER, spaceAfter=2 * mm,
    )
    styles['heading'] = ParagraphStyle(
        'CustomHeading', parent=base['Heading2'],
        fontSize=14, textColor=colors.HexColor('#2c5282'),
        spaceBefore=8 * mm, spaceAfter=4 * mm,
    )
    styles['body'] = ParagraphStyle(
        'CustomBody', parent=base['Normal'],
        fontSize=10, leading=14, spaceAfter=2 * mm,
    )
    styles['interpretation'] = ParagraphStyle(
        'Interpretation', parent=base['Normal'],
        fontSize=10, leading=14,
        textColor=colors.HexColor('#2d3748'),
        backColor=colors.HexColor('#f0f7ff'),
        borderColor=colors.HexColor('#3182ce'),
        borderWidth=1,
        borderPadding=6,
        spaceBefore=3 * mm, spaceAfter=3 * mm,
    )
    styles['label'] = ParagraphStyle(
        'Label', parent=base['Normal'],
        fontSize=9, textColor=colors.HexColor('#4a5568'),
        spaceAfter=1 * mm,
    )
    return styles


# --- Helpers -----------------------------------------------------------------

def _build_descriptive_table(result):
    """Construye una tabla descriptiva de grupos si aplica."""
    if not result.get('groups') or not isinstance(result.get('n'), list):
        return None

    header = ['Grupo', 'n']
    has_mean = result.get('mean') and isinstance(result['mean'], list)
    has_median = result.get('median') and isinstance(result['median'], list)
    if has_mean:
        header += ['Media', 'DE']
    if has_median:
        header += ['Mediana']

    rows = [header]
    for i, g in enumerate(result['groups']):
        row = [str(g), str(result['n'][i])]
        if has_mean:
            row += [f"{result['mean'][i]:.2f}", f"{result['std'][i]:.2f}"]
        if has_median:
            row += [f"{result['median'][i]:.2f}"]
        rows.append(row)

    table = Table(rows, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    return table


def _build_stats_table(result):
    """Construye una tabla con los resultados estadisticos principales."""
    rows = []

    test_name = result.get('test_name', result.get('test', 'N/A'))
    rows.append(['Test', test_name])

    stat = result.get('statistic')
    if isinstance(stat, (int, float)):
        rows.append(['Estadistico', f"{stat:.4f}"])

    p = result.get('p_value')
    if isinstance(p, (int, float)):
        rows.append(['p-valor', f"{p:.6f}"])

    sig = result.get('significant')
    if sig is True:
        rows.append(['Significativo', 'Si'])
    elif sig is False:
        rows.append(['Significativo', 'No'])

    if result.get('ci_lower') is not None and result.get('ci_upper') is not None:
        rows.append(['IC 95%', f"[{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]"])

    if result.get('cohens_d') is not None:
        d = result['cohens_d']
        rows.append(['d de Cohen', f"{d:.3f} ({_effect_label(d)})"])

    if result.get('eta_squared') is not None:
        eta = result['eta_squared']
        rows.append(['eta2', f"{eta:.3f} ({_eta_label(eta)})"])

    if result.get('r_squared') is not None:
        rows.append(['R2', f"{result['r_squared']:.4f}"])

    if result.get('slope') is not None:
        rows.append(['Pendiente', f"{result['slope']:.4f} +/- {result.get('std_error', 0):.4f}"])

    if result.get('auc') is not None:
        rows.append(['AUC', f"{result['auc']:.3f}"])
    if result.get('best_threshold') is not None:
        rows.append(['Corte optimo', f"{result['best_threshold']:.3f}"])
        rows.append(['Sensibilidad', f"{result['sensitivity']:.3f}"])
        rows.append(['Especificidad', f"{result['specificity']:.3f}"])

    if result.get('bias') is not None:
        rows.append(['Sesgo', f"{result['bias']:.3f}"])
        rows.append(['Limites de acuerdo',
                      f"[{result['loa_lower']:.3f}, {result['loa_upper']:.3f}]"])

    if not rows:
        return None

    table = Table(rows, colWidths=[45 * mm, 90 * mm], hAlign='LEFT')
    table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    return table


def _fig_to_image(fig, max_width=150 * mm, max_height=180 * mm):
    """Convierte una figura matplotlib a un Image de reportlab."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    # Calcular dimensiones manteniendo aspect ratio
    fig_w, fig_h = fig.get_size_inches()
    aspect = fig_h / fig_w if fig_w > 0 else 1
    width = max_width
    height = width * aspect
    if height > max_height:
        height = max_height
        width = height / aspect
    img = Image(buf, width=width, height=height)
    img.hAlign = 'CENTER'
    return img


# --- Generador principal ----------------------------------------------------

def generate_pdf_report(results, figures):
    """Genera un informe PDF profesional con resultados y figuras."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
    )
    styles = _get_styles()
    story = []

    # --- Portada ---
    story.append(Spacer(1, 40 * mm))
    story.append(Paragraph('StatLab', styles['title']))
    story.append(Paragraph('Informe de Analisis Estadistico', styles['subtitle']))
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph(
        f'Fecha: {datetime.datetime.now().strftime("%d/%m/%Y %H:%M")}',
        styles['meta']))
    story.append(Paragraph(f'Numero de analisis: {len(results)}', styles['meta']))
    story.append(PageBreak())

    # --- Resultados ---
    valid = [r for r in results if r.get('success')]
    for idx, result in enumerate(valid):
        test_name = result.get('test_name', result.get('test', 'N/A'))
        var_dep = result.get('var_dep', '')
        var_group = result.get('var_group', '')

        story.append(Paragraph(
            f'{idx + 1}. {test_name}: {var_dep} / {var_group}',
            styles['heading']))

        # Tabla descriptiva
        desc_table = _build_descriptive_table(result)
        if desc_table:
            story.append(Paragraph('Descriptivos por grupo:', styles['label']))
            story.append(desc_table)
            story.append(Spacer(1, 3 * mm))

        # Tabla de resultados
        stats_table = _build_stats_table(result)
        if stats_table:
            story.append(Paragraph('Resultados:', styles['label']))
            story.append(stats_table)
            story.append(Spacer(1, 3 * mm))

        # Interpretacion
        interp = generate_interpretation(result)
        if interp:
            story.append(Paragraph('Interpretacion:', styles['label']))
            story.append(Paragraph(interp, styles['interpretation']))

        # Separador
        story.append(Spacer(1, 4 * mm))
        story.append(HRFlowable(
            width='100%', thickness=0.5,
            color=colors.HexColor('#e2e8f0')))

    # --- Figuras ---
    if figures:
        story.append(PageBreak())
        story.append(Paragraph('Figuras', styles['heading']))
        for i, fig in enumerate(figures):
            try:
                img = _fig_to_image(fig)
                story.append(img)
                story.append(Spacer(1, 5 * mm))
            except Exception:
                story.append(Paragraph(
                    f'Error al incluir figura {i + 1}', styles['body']))

    doc.build(story)
    buf.seek(0)
    return buf
