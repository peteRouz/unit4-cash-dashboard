import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
import io

from scipy.interpolate import make_interp_spline
from datetime import datetime
from streamlit_card import card
from matplotlib.colors import LinearSegmentedColormap

# 1) Paginação/Configuração de página deve ser o primeiro st.*
st.set_page_config(layout="wide", page_title="Unit4 Cash Dashboard")

# 2) Injeta todo o CSS (cards + botões)
st.markdown("""
<style>
[data-testid="stApp"] .card {
  background: #fff !important;
  border-radius: 12px !important;
  box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
  padding: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# --- Carregar dados reais do Excel ---
EXCEL_FILE = "CASH DISTRIBUTION_TSR 1 1 1 2.xlsx"
if not os.path.exists(EXCEL_FILE):
    st.error(f"O ficheiro '{EXCEL_FILE}' não foi encontrado.")
    st.stop()

# --- Carregar abas principais ---
dash_sheet     = pd.read_excel(EXCEL_FILE, sheet_name="Information to feed dash", header=None)
sheet7         = pd.read_excel(EXCEL_FILE, sheet_name="Sheet7", header=None)
term_df        = pd.read_excel(EXCEL_FILE, sheet_name="Term Deposits", header=None)
collections_df = pd.read_excel(EXCEL_FILE, sheet_name="Colections", header=None)

# --- NET CASH PER BANK ---
net_cash_df = sheet7.iloc[77:91, [1, 2]].copy()
net_cash_df.columns = ["Banco", "Valor_EUR"]
net_cash_df.dropna(inplace=True)
net_cash_df.reset_index(drop=True, inplace=True)

# --- CASH EOW ---
labels_row = dash_sheet.iloc[30, :]
mask       = labels_row.notna()

# Converter raw_labels e criar eow_labels
raw_labels = labels_row[mask].tolist()
eow_labels = pd.to_datetime(raw_labels, dayfirst=True, errors='coerce')

# Extrair valores da linha 31 e alinhar com mask
raw_values = dash_sheet.iloc[31].where(mask).dropna().tolist()
eow_values = [float(v) for v in raw_values]

eow_data = pd.Series(eow_values, index=eow_labels) / 1_000_000

# --- Cabeçalho ---
st.markdown(f"<h1 align='center'>Unit4 Cash Daily Position Dashboard - {datetime.today().strftime('%d/%m/%Y')}</h1>", unsafe_allow_html=True)
st.markdown("<hr class='dashboard-divider'>", unsafe_allow_html=True)

# --- Primeira linha: Balance e Net Cash per Bank ---
col1, col2 = st.columns([2, 2])
with col1:
    # 1) soma de Sheet7
    total_balance = net_cash_df["Valor_EUR"].sum()

    # 2) lê indicadores da Lista contas (linhas 101 e 102)
    contas = pd.read_excel(EXCEL_FILE, sheet_name="Lista contas", header=None)
    row101 = contas.iloc[100, :]
    last101 = row101.last_valid_index()
    val101 = pd.to_numeric(row101[last101], errors='coerce') if pd.notna(last101) else 0
    row102 = contas.iloc[101, :]
    last102 = row102.last_valid_index()
    raw102 = row102[last102]
    if isinstance(raw102, str) and raw102.endswith('%'):
        val102 = float(raw102.replace('%','').replace(',','.')) / 100
    else:
        val102 = pd.to_numeric(raw102, errors='coerce') or 0

    # 3) seta e cor baseada em val101
    arrow = '↑' if val101 >= 0 else '↓'
    color = 'green' if val101 >= 0 else 'red'

    # 4) exibir indicador
    # Formatar total_balance com separadores europeus: pontos nos milhares e vírgula nos decimais
    total_str = f"{total_balance:,.2f}"
    total_str = total_str.replace(",", "X").replace(".", ",").replace("X", ".")

    # Exibição com % entre parênteses e colorido
    st.markdown(f"""
    <h6 class='small-font'>BALANCE</h6>
    <h2 style='margin:0; font-size:1.75rem; color:#212121;'>
      {total_str} EUR
      <span style='color:{color}; font-size:1rem; margin-left:0.5rem;'>
        {arrow} {val101:,.2f} EUR
        (<span style='color:{color};'>{val102*100:.2f}%</span>)
      </span>
    </h2>
    """, unsafe_allow_html=True)

# gráfico EOW (gradiente suave abaixo da curva)
    width = max(5, len(eow_data)*0.3)
    fig_eow, ax_eow = plt.subplots(figsize=(width, 1.5), dpi=150)

    line_color = '#1f77b4'
    white      = '#ffffff'
    x = np.arange(len(eow_data))
    y = eow_data.values

# Gradiente de branco → azul claríssimo
    light_blue = '#d0e4f8'  # azul muito suave
    cmap = LinearSegmentedColormap.from_list('fade', [white, light_blue])

# 2) Gera a matriz de degradê vertical (altura 256 px × largura len(x))
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.tile(gradient, (1, len(x)))

# 3) Desenha o degradê em todo o retângulo abaixo do máximo de y
    im = ax_eow.imshow(
        gradient,
        aspect='auto',
        cmap=cmap,
        extent=[x.min(), x.max(), 0, y.max()],
        origin='lower',
        zorder=1
)

# 4) Cria o polígono que corresponde à área sob a curva
    from matplotlib.patches import Polygon
    verts = np.vstack([
        np.column_stack([x, y]),  # parte superior (a curva)
        [x.max(), 0],             # canto inferior direito
        [x.min(), 0]              # canto inferior esquerdo
])
    poly = Polygon(verts, facecolor='none', edgecolor='none', closed=True)
    ax_eow.add_patch(poly)

# 5) Recorta o degradê para dentro desse polígono
    im.set_clip_path(poly)

# 6) Desenha a linha principal por cima
    ax_eow.plot(x, y, color=line_color, linewidth=1, zorder=2)

# 7) Formatação de eixos
    font_x = 5
    font_y = 5

    ax_eow.set_xticks(x)
    ax_eow.set_xticklabels(
        [dt.strftime('%d/%b') for dt in eow_data.index],
        rotation=45, ha='right', fontsize=font_x
)
    ax_eow.set_ylabel('Millions', fontsize=font_y)

    y_max = y.max() if len(y)>0 else 1
    ticks = [0, y_max*0.25, y_max*0.5, y_max*0.75, y_max]
    labels = ['0', '25', '50', '75', '100']
    ax_eow.set_yticks(ticks)
    ax_eow.set_yticklabels(labels, fontsize=font_y)
    ax_eow.set_ylim(0, y_max * 1.05)

# 8) Remove margens extras e spines desnecessários
    ax_eow.spines['top'].set_visible(False)
    ax_eow.spines['right'].set_visible(False)
    ax_eow.margins(x=0)

    plt.tight_layout()
    plt.show()

    # Renderiza no Streamlit
    buf = io.BytesIO()
    fig_eow.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    st.image(buf.getvalue(), use_container_width=True)
    buf.close()


      # --- Toggle Buttons under CASH EOW ---
    if 'toggle_rec' not in st.session_state:
        st.session_state.toggle_rec = False
    if st.button("Receivables - TOP 10", key="btn_rec"):
        st.session_state.toggle_rec = not st.session_state.toggle_rec
    if st.session_state.toggle_rec:
        # Cabeçalhos na linha 51 (index 50) e dados de 52 em diante
        headers_rec = dash_sheet.iloc[50, 0:4].tolist()
        data_rec = dash_sheet.iloc[51:60, 0:4]
        df_rec = pd.DataFrame(data_rec.values, columns=headers_rec)
        df_rec.iloc[:, -1] = pd.to_numeric(df_rec.iloc[:, -1], errors='coerce') \
                            .map(lambda x: f"{x:,.0f}")

        # —> Aplica o mesmo estilo que usaste em Incoming Cash Position
        styled_rec = df_rec.style \
            .set_table_styles([
                {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('width', '100%')]},
                {'selector': 'th, td', 'props': [('border', '1px solid black'), ('padding', '0.5rem')]},
                {'selector': 'th', 'props': [('background-color', '#e0e0e0'), ('font-weight', 'bold'), ('text-align', 'center')]},
                {'selector': 'td', 'props': [('text-align', 'center')]},
                {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#f8f8f8')]},
                {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', 'white')]}
            ]) \
            .hide(axis="index")

        html_rec = styled_rec.to_html()
        st.markdown("<h5 style='text-align:left;'>Receivables - TOP 10</h5>", unsafe_allow_html=True)
        st.markdown(html_rec, unsafe_allow_html=True)


    if 'toggle_pay' not in st.session_state:
        st.session_state.toggle_pay = False
    if st.button("Payments >50k", key="btn_pay"):
        st.session_state.toggle_pay = not st.session_state.toggle_pay
    if st.session_state.toggle_pay:
        # Cabeçalhos na linha 51 (index 50) e dados de 52 em diante para Payments
        headers_pay = dash_sheet.iloc[50, 5:9].tolist()
        data_pay = dash_sheet.iloc[51:60, 5:9]
        df_pay = pd.DataFrame(data_pay.values, columns=headers_pay)
        df_pay.iloc[:, -1] = pd.to_numeric(df_pay.iloc[:, -1], errors='coerce') \
                            .map(lambda x: f"{x:,.0f}")

        # —> Aplica o mesmo estilo
        styled_pay = df_pay.style \
            .set_table_styles([
                {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('width', '100%')]},
                {'selector': 'th, td', 'props': [('border', '1px solid black'), ('padding', '0.5rem')]},
                {'selector': 'th', 'props': [('background-color', '#e0e0e0'), ('font-weight', 'bold'), ('text-align', 'center')]},
                {'selector': 'td', 'props': [('text-align', 'center')]},
                {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#f8f8f8')]},
                {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', 'white')]}
            ]) \
            .hide(axis="index")

        html_pay = styled_pay.to_html()
        st.markdown("<h5 style='text-align:left;'>Payments >50k</h5>", unsafe_allow_html=True)
        st.markdown(html_pay, unsafe_allow_html=True)


# --- NET CASH PER BANK & CURRENCY SWITCH ---
with col2:
    # 1) Escolha de visualização
    view = st.radio("", ["EUR", "Currency"], horizontal=True)

    # 1a) Título dinâmico
    chart_title = "NET CASH PER BANK EUR" if view == "EUR" else "CASH PER CURRENCY"
    st.markdown(f"<h5 align='center'>{chart_title}</h5>", unsafe_allow_html=True)

    # 2) Prepara dados conforme view
    if view == "EUR":
        labels   = net_cash_df["Banco"].astype(str).values
        values   = net_cash_df["Valor_EUR"].astype(float).values
        q_values = None
    else:
        raw   = sheet7.iloc[23:37, [0, 14]].reset_index(drop=True)
        raw_q = sheet7.iloc[23:37, 16].reset_index(drop=True)

        # extra ao EUR
        raw_val = sheet7.iloc[77, 2]
        extra   = pd.to_numeric(raw_val, errors="coerce")
        if pd.isna(extra): extra = 0
        mask_eur = raw.iloc[:, 0] == "EUR"
        raw.loc[mask_eur, raw.columns[1]] = (
            pd.to_numeric(raw.loc[mask_eur, raw.columns[1]], errors="coerce").fillna(0)
            + extra
        )

        # remove IDR
        raw   = raw.drop(11).reset_index(drop=True)
        raw_q = raw_q.drop(11).reset_index(drop=True)

        labels   = raw.iloc[:, 0].astype(str).values
        values   = pd.to_numeric(raw.iloc[:, 1], errors="coerce").fillna(0).values
        q_values = pd.to_numeric(raw_q, errors="coerce").fillna(0).values

    # 3) Desenha o gráfico
    fig_bar, ax_bar = plt.subplots(figsize=(5.2, 2.4))
    fig_bar.patch.set_alpha(0)
    ax_bar.patch.set_alpha(0)

    # cores
    colors = (["lightblue"] + ["lightgreen"]*(len(values)-1)) if view=="EUR" else ["lightgreen"]*len(values)
    ax_bar.barh(labels, values, color=colors)
    ax_bar.tick_params(axis="y", labelsize=7)

    # calcula limites para garantir espaço
    max_val   = values.max() if len(values) else 0
    min_val   = values.min() if len(values) else 0
    range_val = max_val - min_val
    # expandimos até +15% do intervalo
    ax_bar.set_xlim(0, max_val + range_val * 0.15)

    # 4) Anotações com offset em pontos
    for i, v in enumerate(values):
        num_str = f"{v:,.0f}".replace(",", ".")
        # 4.1) O valor da coluna O, 5 pontos após o fim da barra
        ax_bar.annotate(
            num_str,
            xy=(v, i),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=6,
            color="black",
        )
        # 4.2) O valor da coluna Q, 60 pontos após o fim da barra
        if view == "Currency" and labels[i] != "EUR":
            q_str = f"({int(q_values[i]):,} EUR)".replace(",", ".")
            ax_bar.annotate(
                q_str,
                xy=(v, i),
                xytext=(50, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=5,
                color="black",
            )

    # estilo final
    ax_bar.xaxis.set_visible(False)
    for spine in ["bottom", "top", "right"]:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines["left"].set_visible(True)

    plt.tight_layout()
    st.pyplot(fig_bar, transparent=True)


# --- Cashflow Forecast & Actual vs Forecast ---
col1, col2 = st.columns(2)
with col1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)

    # 1) Carregar e combinar dados de 2024 (P–AA) e 2025 (B–N)
    m25 = dash_sheet.iloc[2, 1:13].astype(str).tolist()
    i25 = dash_sheet.iloc[5, 1:13].astype(float).tolist()
    o25 = dash_sheet.iloc[6, 1:13].astype(float).tolist()
    n25 = dash_sheet.iloc[4, 1:13].astype(float).tolist()

    m24 = dash_sheet.iloc[2, 15:27].astype(str).tolist()
    i24 = dash_sheet.iloc[5, 15:27].astype(float).tolist()
    o24 = dash_sheet.iloc[6, 15:27].astype(float).tolist()
    n24 = dash_sheet.iloc[4, 15:27].astype(float).tolist()

    months   = m24 + m25
    inflows  = i24 + i25
    outflows = o24 + o25
    netflows = n24 + n25

    # Inicializa posição de scroll
    if 'cf_start' not in st.session_state:
        st.session_state.cf_start = len(m24)

    # --- Definir aqui manualmente os meses com dados reais ---
    # Atualiza esta lista no código sempre que adicionar dados reais ao Excel
    real_months = ['Jan', 'Feb', 'Mar', 'Apr']  # ex: adicionar 'May' quando tiver valores reais de Maio

    # Parâmetros de cores
    default_past_inflow  = '#81c784'
    default_past_outflow = '#ffb74d'
    default_new_inflow   = '#4caf50'
    default_new_outflow  = '#ff9800'
    default_future_color = '#FFF9C4'  # amarelo claro para meses futuros

    # Número de meses de 2025 já tratados como históricos (até Abril 2025)
    historical_new_count = len(real_months)
    start_future = len(m24) + historical_new_count

    # Montar listas de cores para cada mês
    inflow_cols, outflow_cols = [], []
    for idx, m in enumerate(months):
        # cor padrão baseada no status do mês
        if m in real_months:
            ci, co = default_new_inflow, default_new_outflow
        elif idx < len(m24):
            ci, co = default_past_inflow, default_past_outflow
        elif idx < start_future:
            ci, co = default_new_inflow, default_new_outflow
        else:
            ci = co = default_future_color
        inflow_cols.append(ci)
        outflow_cols.append(co)

    # Janela de exibição de 12 meses
    start = st.session_state.cf_start
    window_months   = months[start:start+12]
    window_inflows  = inflows[start:start+12]
    window_outflows = outflows[start:start+12]
    window_netflows = netflows[start:start+12]

    # Plot do gráfico
    x   = np.arange(12)
    x_s = np.linspace(0, 11, 300)
    nf  = np.nan_to_num(window_netflows)
    spline = make_interp_spline(x, nf, k=3)
    nf_s   = spline(x_s)

    fig_cf, ax_cf = plt.subplots(figsize=(7.5, 2.8), dpi=150)
    ax_cf.set_title("Cashflow Forecast", loc='center', fontsize=10, weight='bold')

    # Limites verticais fixos
    y_min = min(min(inflows), min(outflows))
    y_max = max(max(inflows), max(outflows))
    margin = (y_max - y_min) * 0.1
    ax_cf.set_ylim(y_min - margin, y_max + margin)

    # Barras
    bar_w = 0.35
    ax_cf.bar(x, window_inflows, bar_w, color=inflow_cols[start:start+12], edgecolor='black', linewidth=0.5, label='Inflow')
    ax_cf.bar(x, window_outflows, bar_w, color=outflow_cols[start:start+12], edgecolor='black', linewidth=0.5, label='Outflow')

    # Anotações
    for i, v in enumerate(window_inflows):
        ax_cf.text(i, v + y_max * 0.02, f"{int(v):,}".replace(",", "."), ha='center', va='bottom', fontsize=5)
    for i, v in enumerate(window_outflows):
        ax_cf.text(i, v - abs(y_min) * 0.02, f"{int(v):,}".replace(",", "."), ha='center', va='top', fontsize=5)

    # Linha de netflow
    ax_cf.plot(x_s, nf_s, color='#303f9f', linewidth=0.75, label='Net Flow')

    # Eixos
    ax_cf.set_xticks(x)
    ax_cf.set_xticklabels(window_months, fontsize=6, rotation=45)
    ax_cf.set_autoscalex_on(False)
    ax_cf.set_xlim(-0.5, 11.5)

    ticks = [-100000, -75000, -50000, -25000, 0, 25000, 50000, 75000, 100000]
    ax_cf.set_yticks(ticks)
    ax_cf.set_yticklabels([f"{t:,}".replace(",", ".") for t in ticks])
    ax_cf.set_ylabel('Millions', fontsize=8)

    ax_cf.axhline(0, color='black', linewidth=0.5)
    ax_cf.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=6)
    ax_cf.tick_params(axis='x', labelsize=6)
    ax_cf.tick_params(axis='y', labelsize=5)
    plt.tight_layout()

    # Renderizar gráfico
    buf_cf = io.BytesIO()
    fig_cf.savefig(buf_cf, format='png', bbox_inches='tight', dpi=150)
    buf_cf.seek(0)
    st.image(buf_cf.getvalue(), use_container_width=False)
    buf_cf.close()

    # Callbacks para scroll
    def go_prev():
        if st.session_state.cf_start > 0:
            st.session_state.cf_start -= 1

    def go_next():
        # permite avançar até ao último mês disponível
        if st.session_state.cf_start < len(months) - 0:
            st.session_state.cf_start += 1

    # Navegação de meses em duas colunas iguais
    nav_l, nav_r = st.columns(2, gap="large")
    with nav_l:
        st.button(
            "← Previous Months",
            key="btn_prev",
            on_click=go_prev,
            use_container_width=True
        )
    with nav_r:
        st.button(
            "Next Months →",
            key="btn_next",
            on_click=go_next,
            use_container_width=True
        )

    # Fecha o cartão
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    block = dash_sheet.iloc[11:14, 5:17].copy()
    block.index = ['Months', 'Actual', 'Forecast']
    months     = block.loc['Months'].astype(str).tolist()
    actual_m   = pd.to_numeric(block.loc['Actual'].astype(str).str.replace('€','').str.replace(' ', '').str.replace(',','.'), errors='coerce') / 1e6
    forecast_m = pd.to_numeric(block.loc['Forecast'].astype(str).str.replace('€','').str.replace(' ', '').str.replace(',','.'), errors='coerce') / 1e6
    fig_af, ax_af = plt.subplots(figsize=(7.5, 2.8))
    ax_af.set_title("ACTUAL CASH vs CASHFLOW FORECAST", loc='center', fontsize=10, weight='bold')
    ax_af.plot(months, actual_m, linewidth=1.5, label='Actual (M€)')
    ax_af.plot(months, forecast_m, linewidth=1.5, linestyle='--', label='Forecast (M€)')
    max_y = max(actual_m.max(), forecast_m.max()) * 1.1
    offset = max_y * 0.03
    for i, v in enumerate(actual_m):
        ax_af.text(i, v + offset, f"{v:,.1f}M€", ha='center', va='bottom', fontsize=6)
    for i, v in enumerate(forecast_m):
        ax_af.text(i, v - offset, f"{v:,.1f}M€", ha='center', va='top', fontsize=6)
    ax_af.set_xticks(range(len(months)))
    ax_af.set_xticklabels(months, fontsize=8)
    ax_af.set_ylabel("Milhões (€)", fontsize=8)
    ax_af.set_ylim(0, max_y)
    ax_af.set_yticks([0, 50, 100])
    ax_af.tick_params(axis='y', labelsize=7)
    ax_af.axhline(0, color='black', linewidth=0.5)
    ax_af.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=6)
    plt.tight_layout()
    buf_af = io.BytesIO()
    fig_af.savefig(buf_af, format='png', bbox_inches='tight', dpi=150)
    st.image(buf_af.getvalue(), use_container_width=False)
    buf_af.close()

# --- Incoming Cash Position ---
collections_df = collections_df.iloc[3:33, 16:20].copy()
collections_df.columns = ["Entity", "Forecast", "Received", "%"]
collections_df.dropna(how='all', inplace=True)
collections_df = collections_df[collections_df['Entity'] != "Entity"]
collections_df.reset_index(drop=True, inplace=True)
collections_df['Forecast'] = pd.to_numeric(collections_df['Forecast'], errors='coerce').fillna(0)
collections_df['Received'] = pd.to_numeric(collections_df['Received'], errors='coerce').fillna(0)
# Formatar percentual como string com '%'
collections_df['%'] = collections_df.apply(
    lambda r: f"{(r['Received']/r['Forecast']*100):.2f}%" if r['Forecast'] > 0 else "0.00%",
    axis=1
)
collections_df['Forecast'] = collections_df['Forecast'].apply(lambda x: f"{x:,.0f}".replace(",", " "))
collections_df['Received'] = collections_df['Received'].apply(lambda x: f"{x:,.0f}".replace(",", " "))

# Estilizar e exibir tabela com destaque em verde para % > 100%
styled = (
    collections_df.style
    .set_table_styles([
        {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('width', '100%')]},
        {'selector': 'th, td', 'props': [('border', '1px solid black'), ('padding', '0.5rem')]},
        {'selector': 'th', 'props': [('background-color', '#e0e0e0'), ('font-weight', 'bold'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]},
        {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#f8f8f8')]},
        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', 'white')]}    
    ])
    .applymap(
        lambda v: 'color: green; font-weight: bold' if (
            isinstance(v, str) and v.endswith('%') and float(v.rstrip('%')) > 100
        ) else '',
        subset=['%']
    )
    .hide(axis="index")
)

doc_html = styled.to_html()
st.markdown("<h5 style='text-align:left;'>Incoming Cash Position</h5>", unsafe_allow_html=True)
st.markdown(doc_html, unsafe_allow_html=True)

# --- Footer com créditos ---
st.markdown(
"""
<div style='text-align: center; font-size:11px; margin-top:2rem; line-height:1;'>
Created by:<br>
Pedro Miguel M
</div>
""",
unsafe_allow_html=True
)

