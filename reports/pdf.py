"""Generacion de informes PDF."""

import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO

from reports.text import format_result_text


def generate_pdf_report(results, figures):
    """Genera un informe PDF con resultados y figuras."""
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # Portada
        fig_cover = plt.figure(figsize=(8.5, 11))
        fig_cover.text(0.5, 0.65, 'StatLab', fontsize=36, ha='center',
                       fontweight='bold', color='#1a365d')
        fig_cover.text(0.5, 0.58, 'Informe de Analisis Estadistico',
                       fontsize=16, ha='center', color='#4a5568')
        fig_cover.text(0.5, 0.50, 'Generado con StatLab',
                       fontsize=11, ha='center', color='#718096')
        fig_cover.text(0.5, 0.44,
                       f'Fecha: {datetime.datetime.now().strftime("%d/%m/%Y %H:%M")}',
                       fontsize=11, ha='center', color='#718096')
        fig_cover.text(0.5, 0.38, f'Numero de analisis: {len(results)}',
                       fontsize=11, ha='center', color='#718096')
        plt.axis('off')
        pdf.savefig(fig_cover)
        plt.close(fig_cover)

        # Resultados
        for result in results:
            if not result.get('success'):
                continue
            text = format_result_text(result)
            fig_res = plt.figure(figsize=(8.5, 11))
            fig_res.text(0.07, 0.95, text, fontsize=9, fontfamily='monospace',
                         verticalalignment='top', wrap=True)
            plt.axis('off')
            pdf.savefig(fig_res)
            plt.close(fig_res)

        # Figuras
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight')

    buf.seek(0)
    return buf
