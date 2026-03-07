# StatLab 📊

Aplicación de análisis estadístico y generación de figuras.  
Alternativa ligera a GraphPad, self-hosted, basada en Python.

## Instalación rápida

```bash
# Crear entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Lanzar
streamlit run app.py
```

La app se abre automáticamente en `http://localhost:8501`.

## Despliegue en Proxmox (LXC)

```bash
# En tu LXC dedicado
apt update && apt install -y python3 python3-pip python3-venv
mkdir -p /opt/statlab && cd /opt/statlab

# Copiar archivos (app.py + requirements.txt)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Lanzar como servicio
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Para acceso vía Tailscale, exponer el puerto 8501 en tu subnet router.

### Servicio systemd (opcional)

```ini
# /etc/systemd/system/statlab.service
[Unit]
Description=StatLab
After=network.target

[Service]
WorkingDirectory=/opt/statlab
ExecStart=/opt/statlab/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
User=root

[Install]
WantedBy=multi-user.target
```

```bash
systemctl enable --now statlab
```

## Tests disponibles (v1)

| Escenario | Paramétrico | No paramétrico |
|---|---|---|
| 2 grupos independientes | T-test / Welch | Mann-Whitney U |
| 2 grupos pareados | T-test pareado | Wilcoxon |
| >2 grupos | ANOVA + Tukey | Kruskal-Wallis + Dunn |
| Correlación | Pearson | Spearman |
| Categóricas | Chi-cuadrado | Fisher |

## Figuras

- Box plot + puntos individuales
- Violin plot
- Barras con error (SEM)
- Gráfico de datos pareados
- Scatter + regresión
- Histogramas por grupo

Todas exportables a PNG (300 dpi) y al informe PDF.

## Roadmap

- [ ] Kaplan-Meier (survival)
- [ ] Bland-Altman
- [ ] ROC curves
- [ ] Informe PDF mejorado (reportlab)
- [ ] Exportar figuras en SVG
- [ ] Múltiples comparaciones avanzadas
- [ ] Integración con DocBot vía API
