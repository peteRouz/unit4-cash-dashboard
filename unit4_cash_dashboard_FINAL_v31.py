import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
import io
from scipy.interpolate import make_interp_spline
from datetime import datetime

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

# --- Configuração de layout ---
st.set_page_config(layout="wide")
st.markdown("""
<style>
  .small-font { font-size:12px !important; }
  .card { padding:0.5rem 1rem; border:1px solid #ccc; border-radius:0.5rem; margin-bottom:1rem; background:#f9f9f9; }
  .dashboard-divider {
    border: none;
    height: 4px;
    background-color: #f8cbd1;
    margin: 0 0 1rem 0;
  }
  [data-testid="stColumns"] > div {
    background: transparent !important;
    padding: 0 !important;
  }
</style>
""", unsafe_allow_html=True)

# --- Cabeçalho ---
st.markdown(f"<h1 align='center'>Unit4 Cash Daily Position Dashboard - {datetime.today().strftime('%d/%m/%Y')}</h1>", unsafe_allow_html=True)
st.markdown("<hr class='dashboard-divider'>", unsafe_allow_html=True)

# --- Primeira linha: Balance e Net Cash per Bank ---
col1, col2 = st.columns([2, 2])
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

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
    total_str = f"{total_balance:,.2f}"  # e.g. '51,571,074.66'
    total_str = total_str.replace(",", "X").replace(".", ",").replace("X", ".")  # '51.571.074,66'
    # Formatar val101 e val102 já existente
    val101_str = f"{val101:,.2f} EUR".replace(",", " ")
    val102_str = f"{val102*100:.2f}%"
    combined = f"{val101_str} {val102_str}"

    st.markdown(f"""
<h6 class='small-font'>BALANCE</h6>
<h2 style='text-align:left; margin:0;'>
    {total_str} EUR
    <span style='color:{color}; font-size:1rem; vertical-align:middle; margin-left:0.2rem;'>{arrow}</span>
    <span style='font-size:1rem; vertical-align:middle; margin-left:0.5rem;'>{combined}</span>
</h2>
""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Gráfico CASH EOW
    width = max(5, len(eow_data) * 0.3)
    fig_eow, ax_eow = plt.subplots(figsize=(width, 2.0), dpi=150)
    x = np.arange(len(eow_data))
    ax_eow.plot(x, eow_data.values, color='green', linewidth=1)
    ax_eow.set_xticks(x)
    ax_eow.set_xticklabels([dt.strftime('%d/%b') for dt in eow_data.index], rotation=45, fontsize=7)
    ax_eow.set_ylabel('Milhões', fontsize=8)
    ax_eow.set_ylim(0, max(eow_data.values) * 1.15)
    ax_eow.set_yticks([0, 50, 100, 150])
    ax_eow.set_title('CASH EOD', fontsize=10, weight='bold')
    ax_eow.tick_params(axis='y', labelsize=7)
    plt.tight_layout()
    buf_eow = io.BytesIO()
    fig_eow.savefig(buf_eow, format='png', bbox_inches='tight', dpi=150)
    st.image(buf_eow.getvalue(), use_container_width=False)
    buf_eow.close()

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
        df_rec.iloc[:, -1] = pd.to_numeric(df_rec.iloc[:, -1], errors='coerce')\
                            .map(lambda x: f"{x:,.0f}")
        st.dataframe(
            df_rec,
            hide_index=True,
            use_container_width=True
        )

    if 'toggle_pay' not in st.session_state:
        st.session_state.toggle_pay = False
    if st.button("Payments >50k", key="btn_pay"):
        st.session_state.toggle_pay = not st.session_state.toggle_pay
    if st.session_state.toggle_pay:
        # Cabeçalhos na linha 51 (index 50) e dados de 52 em diante para Payments
        headers_pay = dash_sheet.iloc[50, 5:9].tolist()
        data_pay = dash_sheet.iloc[51:60, 5:9]
        df_pay = pd.DataFrame(data_pay.values, columns=headers_pay)
        df_pay.iloc[:, -1] = pd.to_numeric(df_pay.iloc[:, -1], errors='coerce')\
                        .map(lambda x: f"{x:,.0f}")
        st.dataframe(
            df_pay,
            hide_index=True,
            use_container_width=True
        )

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h5 align='center'>NET CASH PER BANK</h5>", unsafe_allow_html=True)
    fig_bar, ax_bar = plt.subplots(figsize=(5.2,2.4))
    highlight_idx = 0
    bar_colors = ['lightblue' if i == highlight_idx else 'lightgreen' for i in range(len(net_cash_df))]
    ax_bar.barh(net_cash_df['Banco'], net_cash_df['Valor_EUR'], color=bar_colors)
    ax_bar.tick_params(axis='y', labelsize=7)
    max_val = net_cash_df['Valor_EUR'].max()
    for i, v in enumerate(net_cash_df['Valor_EUR']):
        ax_bar.text(v + max_val * 0.01, i, f"{v:,.0f}".replace(',', '.'), va='center', fontsize=6)
    ax_bar.xaxis.set_visible(False)
    for spine in ['bottom', 'top', 'right']:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines['left'].set_visible(True)
    plt.tight_layout()
    st.pyplot(fig_bar)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Cashflow Forecast & Actual vs Forecast ---
col1, col2 = st.columns(2)
with col1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    cashflow_months = dash_sheet.iloc[2, 1:13].astype(str).tolist()
    inflow  = dash_sheet.iloc[5, 1:13].astype(float).tolist()
    outflow = dash_sheet.iloc[6, 1:13].astype(float).tolist()
    netflow = dash_sheet.iloc[4, 1:13].astype(float).tolist()
    x = np.arange(len(cashflow_months))
    x_s = np.linspace(x.min(), x.max(), 300)
    spline = make_interp_spline(x, netflow, k=3)
    netflow_s = spline(x_s)
    inflow_colors  = ['#4caf50']*3 + ['lightyellow'] * 9
    outflow_colors = ['#ff9800']*3 + ['lightyellow'] * 9
    fig_cf, ax_cf = plt.subplots(figsize=(7.5, 2.8))
    ax_cf.set_title("Cashflow Forecast", loc='center', fontsize=10, weight='bold')
    bar_w = 0.35
    ax_cf.bar(x - bar_w/2, inflow,  bar_w, color=inflow_colors, edgecolor='black', linewidth=0.5, label='Inflow')
    ax_cf.bar(x + bar_w/2, outflow, bar_w, color=outflow_colors, edgecolor='black', linewidth=0.5, label='Outflow')
    ax_cf.plot(x_s, netflow_s, color='#303f9f', linewidth=0.75, label='Net Flow')
    for i, v in enumerate(inflow):
        ax_cf.text(i - bar_w/2, v + 2000, f"{v:,.0f}".replace(',', ' '), ha='center', fontsize=5)
    for i, v in enumerate(outflow):
        ax_cf.text(i + bar_w/2, v - 7000, f"{v:,.0f}".replace(',', ' '), ha='center', fontsize=5)
    ax_cf.set_xticks(x)
    ax_cf.set_xticklabels(cashflow_months, fontsize=6)
    ax_cf.axhline(0, color='black', linewidth=0.5)
    ax_cf.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=6)
    ax_cf.tick_params(axis='x', labelsize=6)
    ax_cf.tick_params(axis='y', labelsize=5)
    plt.tight_layout()
    buf_cf = io.BytesIO()
    fig_cf.savefig(buf_cf, format='png', bbox_inches='tight', dpi=150)
    st.image(buf_cf.getvalue(), use_container_width=False)
    buf_cf.close()

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
collections_df['%'] = collections_df.apply(lambda r: f"{(r['Received']/r['Forecast']*100):.2f}%" if r['Forecast'] > 0 else "0.00%", axis=1)
collections_df['Forecast'] = collections_df['Forecast'].apply(lambda x: f"{x:,.0f}".replace(",", " "))
collections_df['Received'] = collections_df['Received'].apply(lambda x: f"{x:,.0f}".replace(",", " "))

# Estilizar e exibir tabela
styled = collections_df.style \
    .set_table_styles([
        {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('width', '100%')]},
        {'selector': 'th, td', 'props': [('border', '1px solid black'), ('padding', '0.5rem')]},
        {'selector': 'th', 'props': [('background-color', '#e0e0e0'), ('font-weight', 'bold'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]},
        {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#f8f8f8')]},
        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', 'white')]}    
    ]) \
    .hide(axis="index")

doc_html = styled.to_html()
st.markdown("<h5 style='text-align:left;'>Incoming Cash Position</h5>", unsafe_allow_html=True)
st.markdown(doc_html, unsafe_allow_html=True)

# --- Footer com créditos ---
st.markdown(
"""
<div style='text-align: center; font-size:11px; margin-top:2rem; line-height:1;'>
Created by:<br>
Pedro Miguel
</div>
""",
unsafe_allow_html=True
)

