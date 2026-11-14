import os
import json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "results/metrics/mp02_sampling_quantization.json"
EXT_METRICS_PATH = ROOT / "results/metrics/mp02_extension_audio_metrics.json"
FIG1_PATH = ROOT / "results/figures/mp02_sampling_quantization.png"
FIG2_PATH = ROOT / "results/figures/mp02_extension_error_analysis.png"
PDF_OUT = ROOT / "Jawaban_Soal_Mini_Project_02.pdf"

# Load metrics
with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics = json.load(f)

ext_metrics = {}
if EXT_METRICS_PATH.exists():
    with open(EXT_METRICS_PATH, "r", encoding="utf-8") as f:
        ext_metrics = json.load(f)

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))
        
        # Running header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(45, 805, "Teknik Pemrosesan Suara dan AI pada Sistem Embedded | Mini Project 02")
            self.drawRightString(550, 805, "Laporan Jawaban Pertanyaan")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(45, 798, 550, 798)
            
        # Running footer
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(45, 45, 550, 45)
        self.drawString(45, 33, "Keyword Spotting Embedded - Laboratorium Sistem Komputer & Cerdas")
        page_str = f"Halaman {self._pageNumber} dari {total_pages}"
        self.drawRightString(550, 33, page_str)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_OUT),
        pagesize=A4,
        leftMargin=45,
        rightMargin=45,
        topMargin=50,
        bottomMargin=55
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    primary_color = colors.HexColor("#1A365D")   # Deep navy
    secondary_color = colors.HexColor("#2B6CB0") # Slate blue
    accent_dark = colors.HexColor("#2D3748")     # Dark charcoal body
    highlight_color = colors.HexColor("#C53030") # Subdued red

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=primary_color,
        alignment=1, # Center
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        textColor=secondary_color,
        alignment=1,
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Header2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=secondary_color,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.8,
        textColor=accent_dark,
        spaceAfter=5
    )

    body_bold = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=accent_dark
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold',
        textColor=colors.whitesmoke
    )

    box_text = ParagraphStyle(
        'BoxText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1A202C")
    )

    story = []

    # 1. Title Banner
    story.append(Paragraph("TEKNIK PEMROSESAN SUARA DAN AI PADA SISTEM EMBEDDED", subtitle_style))
    story.append(Paragraph("LAPORAN JAWABAN PRAKTIKUM - MINI PROJECT 02", title_style))
    story.append(Paragraph("Analisis Eksperimen Sampling Rate, Kuantisasi Audio, dan Trade-off Resource TinyML", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=8, spaceBefore=0))

    # 2. Anggota Kelompok Table
    anggota_data = [
        [
            Paragraph("<b>Identitas Kelompok:</b>", table_cell_bold),
            Paragraph("<b>Program Studi / Institusi:</b>", table_cell_bold)
        ],
        [
            Paragraph(
                "1. Christama Ezra Yudianto &nbsp;&nbsp;&nbsp;(245150307111009)<br/>"
                "2. Abdillah Fatah Suryanto &nbsp;&nbsp;&nbsp;&nbsp;(245150300111008)<br/>"
                "3. Ivang Arfandia &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(245150307111003)<br/>"
                "4. Jahzeel Erason Nicodemus (245150300111002)",
                box_text
            ),
            Paragraph(
                "<b>Topik:</b> Audio Preprocessing & Resource Optimization<br/>"
                "<b>Modul:</b> Mini Project 02 (Sampling & Quantization)<br/>"
                "<b>File Target:</b> <code>data/raw/nyala_001.wav</code><br/>"
                "<b>Status Handoff:</b> <code>ready_for_mp03</code>",
                box_text
            )
        ]
    ]
    t_anggota = Table(anggota_data, colWidths=[290, 215])
    t_anggota.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#F7FAFC")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_anggota)
    story.append(Spacer(1, 10))

    # 3. Data Pengukuran Eksperimen Table
    story.append(Paragraph("A. Data Hasil Pengukuran Eksperimen (<code>results/metrics/</code>)", h1_style))
    story.append(Paragraph(
        "Tabel berikut menyajikan data empiris yang diperoleh dari eksekusi <code>mini_project_02.py</code> "
        "yang membandingkan 4 konfigurasi representasi audio digital terhadap file baseline ucapan kata <b>'nyala'</b> (durasi 2,0 detik):",
        body_style
    ))

    sizes = metrics["file_size_bytes"]
    red = metrics["size_reduction_percent_vs_baseline"]
    snr_q = metrics["quantization_8bit_snr_db"]
    snr_res = metrics["resample_8k_roundtrip_snr_db"]

    data_table = [
        [
            Paragraph("<b>Konfigurasi</b>", table_cell_bold),
            Paragraph("<b>Sampling Rate</b>", table_cell_bold),
            Paragraph("<b>Bit Depth</b>", table_cell_bold),
            Paragraph("<b>Ukuran File</b>", table_cell_bold),
            Paragraph("<b>Reduksi Ukuran</b>", table_cell_bold),
            Paragraph("<b>Keterangan Kualitas / SNR</b>", table_cell_bold)
        ],
        [
            Paragraph("<b>16k16_baseline</b>", table_cell),
            Paragraph("16.000 Hz", table_cell),
            Paragraph("16-bit signed", table_cell),
            Paragraph(f"{sizes['16k16_baseline']:,} B ({sizes['16k16_baseline']/1024:.2f} KiB)", table_cell),
            Paragraph(f"{red['16k16_baseline']:.2f}% (Acuan)", table_cell),
            Paragraph("Format Baku Proyek (Baseline Murni)", table_cell)
        ],
        [
            Paragraph("<b>8k16</b>", table_cell),
            Paragraph("8.000 Hz", table_cell),
            Paragraph("16-bit signed", table_cell),
            Paragraph(f"{sizes['8k16']:,} B ({sizes['8k16']/1024:.2f} KiB)", table_cell),
            Paragraph(f"{red['8k16']:.2f}%", table_cell),
            Paragraph(f"SNR Resampling = {snr_res:.2f} dB", table_cell)
        ],
        [
            Paragraph("<b>16k8</b>", table_cell),
            Paragraph("16.000 Hz", table_cell),
            Paragraph("8-bit unsigned", table_cell),
            Paragraph(f"{sizes['16k8']:,} B ({sizes['16k8']/1024:.2f} KiB)", table_cell),
            Paragraph(f"{red['16k8']:.2f}%", table_cell),
            Paragraph(f"SNR Kuantisasi = {snr_q:.2f} dB", table_cell)
        ],
        [
            Paragraph("<b>8k8</b>", table_cell),
            Paragraph("8.000 Hz", table_cell),
            Paragraph("8-bit unsigned", table_cell),
            Paragraph(f"{sizes['8k8']:,} B ({sizes['8k8']/1024:.2f} KiB)", table_cell),
            Paragraph(f"<b>{red['8k8']:.2f}% (Terkecil)</b>", table_cell),
            Paragraph("Reduksi Ukuran Maksimum (74,95%)", table_cell)
        ]
    ]

    t_metrics = Table(data_table, colWidths=[95, 75, 75, 95, 80, 85])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), secondary_color),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 10))

    # 4. Jawaban Tiga Pertanyaan Utama
    story.append(Paragraph("B. Jawaban Pertanyaan Praktikum (Langkah 5)", h1_style))

    # --- SOAL 1 ---
    story.append(Paragraph("Pertanyaan 1: Konfigurasi mana yang menghasilkan ukuran file paling kecil?", h2_style))
    p1_text = (
        "<b>Jawaban:</b><br/>"
        "Konfigurasi yang menghasilkan ukuran file paling kecil secara empiris adalah konfigurasi <b>8 kHz / 8-bit (8k8)</b>, "
        "dengan total ukuran file sebesar <b>16.044 bytes (~15,67 KiB)</b>. Konfigurasi ini menghasilkan efisiensi kompresi "
        "memori sebesar <b>74,95%</b> dibandingkan konfigurasi baseline 16 kHz / 16-bit (64.044 bytes).<br/><br/>"
        "<b>Pembuktian dan Formulasi Matematis:</b><br/>"
        "Ukuran berkas audio format PCM linier mentah tanpa kompresi ditentukan oleh persamaan bitrate digital:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Ukuran Data Audio (Bytes) = [ (Sampling Rate &times; Bit Depth &times; Channels &times; Durasi Detik) / 8 ] + Header WAV (44 Bytes)</b><br/>"
        "Dengan parameter durasi <i>T = 2,0 detik</i>, saluran tunggal <i>mono (Channels = 1)</i>, dan <i>Header RIFF WAV = 44 bytes</i>:<br/>"
        "&bull; <b>16k16_baseline:</b> (16.000 &times; 16 &times; 1 &times; 2,0) / 8 + 44 = 64.000 + 44 = <b>64.044 Bytes</b> (100% kapasitas acuan)<br/>"
        "&bull; <b>8k16:</b> (8.000 &times; 16 &times; 1 &times; 2,0) / 8 + 44 = 32.000 + 44 = <b>32.044 Bytes</b> (Reduksi 49,97%)<br/>"
        "&bull; <b>16k8:</b> (16.000 &times; 8 &times; 1 &times; 2,0) / 8 + 44 = 32.000 + 44 = <b>32.044 Bytes</b> (Reduksi 49,97%)<br/>"
        "&bull; <b>8k8:</b> (8.000 &times; 8 &times; 1 &times; 2,0) / 8 + 44 = 16.000 + 44 = <b>16.044 Bytes</b> (Reduksi 74,95%)<br/>"
        "Karena sampling rate diturunkan menjadi 1/2 (16k &rarr; 8k) dan kuantisasi diturunkan menjadi 1/2 (16-bit &rarr; 8-bit), "
        "maka total data throughput turun menjadi <b>(1/2) &times; (1/2) = 1/4 (25%)</b> dari throughput baseline. "
        "Hal ini membuktikan secara matematis dan empiris bahwa konfigurasi 8k8 merupakan format paling hemat memori."
    )
    story.append(Paragraph(p1_text, body_style))
    story.append(Spacer(1, 6))

    # --- SOAL 2 ---
    story.append(Paragraph("Pertanyaan 2: Apa yang terjadi pada batas frekuensi ketika sampling rate turun dari 16 kHz menjadi 8 kHz?", h2_style))
    p2_text = (
        "<b>Jawaban:</b><br/>"
        "Berdasarkan <b>Teorema Sampling Nyquist-Shannon</b>, batas frekuensi tertinggi yang dapat direkonstruksi secara unik "
        "dan sempurna oleh sistem diskrit tanpa distorsi aliasing disebut <i>Frekuensi Nyquist</i> (f<sub>Nyquist</sub>), "
        "yaitu bernilai tepat setengah dari laju sampling yang digunakan: <b>f<sub>Nyquist</sub> = f<sub>s</sub> / 2</b>.<br/><br/>"
        "<b>1. Pergeseran Batas Frekuensi Maksimum:</b><br/>"
        "&bull; Pada <b>f<sub>s</sub> = 16.000 Hz (16 kHz):</b> Batas frekuensi Nyquist berada pada <b>8.000 Hz (8 kHz)</b>. "
        "Spektrum audio mampu merepresentasikan komponen gelombang dari 0 Hz hingga 8 kHz secara utuh.<br/>"
        "&bull; Pada <b>f<sub>s</sub> = 8.000 Hz (8 kHz):</b> Batas frekuensi Nyquist turun drastis menjadi <b>4.000 Hz (4 kHz)</b>. "
        "Lebar pita (bandwidth) sinyal audio menyempit sebesar 50%.<br/><br/>"
        "<b>2. Mekanisme Filtering Anti-Aliasing dan Spektrum:</b><br/>"
        "Ketika proses resampling dijalankan menggunakan <code>scipy.signal.resample_poly</code>, algoritma secara otomatis menyisipkan "
        "filter lolos-rendah (<i>low-pass anti-aliasing FIR filter</i>) dengan cutoff 4 kHz sebelum desimasi. "
        "Tujuannya adalah meredam seluruh energi di atas 4 kHz agar tidak 'melipat' (aliasing) ke dalam spektrum suara 0-4 kHz.<br/><br/>"
        "<b>3. Konsekuensi Akustik terhadap Suara Manusia (Speech Perception):</b><br/>"
        "Pemotongan frekuensi di atas 4 kHz menghilangkan komponen harmonik formants atas dan energi bunyi konsonan frikatif/plosif "
        "(seperti desis fonem /s/, /sy/, /f/, /t/, /k/). Akibatnya:<br/>"
        "&bull; Kualitas audio terdengar teredam (<i>muffled</i>), menyerupai standar sambungan telepon analog narrowband (PSTN standard 300-3.400 Hz).<br/>"
        "&bull; Berdasarkan metrik fitur tambahan pada file eksperimen, kehilangan energi frekuensi tinggi (&gt;4 kHz) tercatat sebesar "
        f"<b>{ext_metrics.get('resampling_8k_analysis', {}).get('hf_energy_loss_percent', 0.40):.2f}%</b> dengan degradasi SNR roundtrip menjadi <b>{snr_res:.2f} dB</b>."
    )
    story.append(Paragraph(p2_text, body_style))
    story.append(Spacer(1, 6))

    # --- SOAL 3 ---
    story.append(Paragraph("Pertanyaan 3: Mengapa proyek tetap memilih 16 kHz/16-bit walaupun file 8 kHz/8-bit lebih kecil?", h2_style))
    p3_text = (
        "<b>Jawaban:</b><br/>"
        "Meskipun varian 8 kHz / 8-bit menghemat memori hingga 74,95%, proyek embedded Keyword Spotting ini <b>secara tegas menetapkan format "
        "16 kHz / 16-bit mono sebagai format standar sistem</b> (terekam pada <code>config/audio_format_selected.json</code>). "
        "Keputusan desain teknik (engineering trade-off) ini didasarkan pada empat pertimbangan fundamental:<br/><br/>"
        "<b>1. Preservasi Integritas Fitur Akustik untuk AI (Keyword Spotting):</b><br/>"
        "Tujuan akhir sistem embedded ini adalah mengenali kata kunci (misal: 'nyala', 'mati', 'naik', 'turun') menggunakan modul ekstraksi "
        "spektrogram dan MFCC (Mel-Frequency Cepstral Coefficients) pada Mini Project berikutnya. Fonem konsonan seperti 'ny', 'l', 'k', dan 't' "
        "memiliki pola formant dan transisi energi unik pada rentang 4 kHz - 8 kHz. Jika informasi ini dihilangkan oleh sampling 8 kHz, "
        "vektor fitur akan kehilangan diskriminasi fonemik, yang secara signifikan menurunkan akurasi klasifikasi model machine learning / neural network.<br/><br/>"
        "<b>2. Rentang Dinamis (Dynamic Range) dan Noise Floor Kuantisasi:</b><br/>"
        "&bull; <b>Format 16-bit:</b> Memiliki 65.536 level kuantisasi diskrit dengan rentang dinamis teoritis ~96,3 dB (6,02 dB &times; 16 bit). "
        "Kuantisasi ini menghasilkan noise floor yang sangat rendah, menjaga detail sinyal ucapan bahkan pada suku kata bervolume pelan.<br/>"
        "&bull; <b>Format 8-bit:</b> Hanya memiliki 256 level diskrit dengan SNR terukur hanya <b>36,62 dB</b>. Galat kuantisasi pada 8-bit "
        "menghasilkan derau desis (<i>quantization noise hiss</i>) yang cukup kuat. Derau ini mengontaminasi energi pita filterbank MFCC, "
        "sehingga meningkatkan False Acceptance Rate (salah deteksi kata) pada model inferensi TinyML.<br/><br/>"
        "<b>3. Kompatibilitas Standar Ekosistem TinyML dan Pretrained Model:</b><br/>"
        "Standar industri open-source model suara embedded (seperti Google Speech Commands dataset, Edge Impulse TinyML pipeline, dan "
        "TensorFlow Lite for Microcontrollers) secara universal dibangun di atas format audio 16 kHz / 16-bit. Menggunakan format baku ini menjamin "
        "interoperabilitas arsitektur neural network tanpa memerlukan lapisan normalisasi atau resampling runtime yang membebani CPU mikrokontroler.<br/><br/>"
        "<b>4. Kelayakan Sumber Daya Memori Mikrokontroler Modern:</b><br/>"
        "Kebutuhan memori untuk buffer audio 16 kHz / 16-bit mono selama 2,0 detik adalah 64.000 bytes (~62,5 KiB). "
        "Mikrokontroler target Edge AI modern (seperti ESP32 dengan 520 KiB SRAM, STM32F4/F7 dengan 128-512 KiB SRAM, atau Raspberry Pi Pico 2) "
        "memiliki kapasitas memori internal yang sangat memadai. Mengorbankan akurasi AI hanya untuk menghemat beberapa puluh kilobyte memori "
        "merupakan kompromi yang buruk (<i>suboptimal engineering trade-off</i>)."
    )
    story.append(Paragraph(p3_text, body_style))
    
    # 5. Visualisasi Hasil Eksperimen
    story.append(PageBreak()) # New page for figures and deeper analysis
    story.append(Paragraph("C. Lampiran Visualisasi Eksperimen dan Interpretasi Grafik", h1_style))

    if FIG1_PATH.exists():
        story.append(Paragraph("<b>Gambar 1: Visualisasi Sampling Rate dan Kuantisasi (results/figures/mp02_sampling_quantization.png)</b>", h2_style))
        # Keep aspect ratio width ~480, height ~270
        story.append(Image(str(FIG1_PATH), width=6.8*inch, height=3.8*inch))
        story.append(Spacer(1, 4))
        
        analisis_fig1 = (
            "<b>Analisis Tiga Subplot Gambar 1:</b><br/>"
            "&bull; <b>Subplot 1 (Waveform Kuantisasi):</b> Memperlihatkan perbandingan sinyal asli 16-bit (garis biru halus kontinu) terhadap sinyal hasil kuantisasi 8-bit (garis oranye bertangga diskrit). Undakan diskrit menunjukkan proses pembulatan amplitudo sinyal ke 256 nilai integer terdekat (0..255), yang menjadi sumber timbulnya galat kuantisasi.<br/>"
            "&bull; <b>Subplot 2 (Spektrum Nyquist):</b> Kurva spektral 16 kHz (garis biru) memperlihatkan distribusi energi frekuensi membentang hingga batas 8 kHz. Sebaliknya, kurva spektral hasil downsampling 8 kHz (garis oranye) terputus tegak tepat pada batas Nyquist 4 kHz, membuktikan pemangkasan energi frekuensi tinggi akibat anti-aliasing filter.<br/>"
            "&bull; <b>Subplot 3 (Perbandingan Ukuran File):</b> Diagram batang mengonfirmasi perbandingan ukuran file riil: 16k16 (~62,5 KiB), 8k16 (~31,3 KiB), 16k8 (~31,3 KiB), dan 8k8 (~15,7 KiB). Varian 8k8 terbukti menghemat memori hingga 74,95%."
        )
        story.append(Paragraph(analisis_fig1, body_style))
        story.append(Spacer(1, 10))

    if FIG2_PATH.exists():
        story.append(Paragraph("<b>Gambar 2: Analisis Lanjutan Fitur Tambahan (results/figures/mp02_extension_error_analysis.png)</b>", h2_style))
        story.append(Image(str(FIG2_PATH), width=6.8*inch, height=3.8*inch))
        story.append(Spacer(1, 4))
        
        analisis_fig2 = (
            "<b>Analisis Fitur Tambahan (Advanced Sampling & Quantization Quality Analysis):</b><br/>"
            "&bull; <b>Extension 1 (Sinyal Galat Kuantisasi e[n]):</b> Plot galat time-domain memperlihatkan residual perbedaan antara sinyal floating point asli dan sinyal kuantisasi 8-bit. Galat berosilasi di sekitar nol dengan nilai mutlak maksimum (Max Absolute Error) sebesar <b>0,003922</b> dan RMSE sebesar <b>0,003172</b>.<br/>"
            "&bull; <b>Extension 2 (Histogram PDF Galat):</b> Distribusi galat kuantisasi membentuk kurva seragam (uniform distribution) pada rentang interval step kuantisasi [-&Delta;/2, +&Delta;/2]. Hal ini mengonfirmasi sifat derau kuantisasi yang mendekati <i>White Quantization Noise</i>.<br/>"
            "&bull; <b>Extension 3 (Distribusi Kumulatif Energi Spektral):</b> Kurva kumulatif energi spektrum menunjukkan bahwa <b>99,60%</b> energi akustik ucapan 'nyala' terpusat di bawah 4 kHz (garis putus-putus merah), sementara <b>0,40%</b> energi berada di atas 4 kHz yang hilang saat downsampling ke 8 kHz."
        )
        story.append(Paragraph(analisis_fig2, body_style))

    # Summary Conclusion Box
    story.append(Spacer(1, 10))
    summary_box_data = [
        [
            Paragraph("<b>RINGKASAN KESIMPULAN REKAYASA (ENGINEERING TAKEAWAY)</b>", table_cell_bold)
        ],
        [
            Paragraph(
                "1. <b>Kapasitas File Minimum:</b> Format <b>8 kHz / 8-bit</b> menghasilkan file terkecil (16.044 bytes, efisiensi 74,95%), namun menimbulkan degradasi akustik signifikan.<br/>"
                "2. <b>Batas Frekuensi Nyquist:</b> Penurunan sampling rate dari 16 kHz ke 8 kHz menurunkan frekuensi cutoff Nyquist dari <b>8.000 Hz menjadi 4.000 Hz</b>, menghilangkan komponen fonem frikatif frekuensi tinggi.<br/>"
                "3. <b>Keputusan Format Proyek:</b> Proyek memilih <b>16 kHz / 16-bit mono</b> karena menjaga rasio SNR tinggi (36,62 dB pada 8-bit vs ~96 dB pada 16-bit), mempertahankan fitur MFCC untuk akurasi klasifikasi AI kata kunci, dan sepenuhnya kompatibel dengan kapasitas RAM mikrokontroler TinyML modern.",
                box_text
            )
        ]
    ]
    t_sum = Table(summary_box_data, colWidths=[505])
    t_sum.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#EDF2F7")),
        ('BOX', (0, 0), (-1, -1), 1, primary_color),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_sum)

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print("PDF successfully generated at:", PDF_OUT)

if __name__ == "__main__":
    build_pdf()
