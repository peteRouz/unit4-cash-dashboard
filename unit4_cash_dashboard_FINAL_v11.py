import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
import io

# --- Carregar dados reais do Excel ---
EXCEL_FILE = "CASH DISTRIBUTION_TSR 1 1 1 2.xlsx"

if not os.path.exists(EXCEL_FILE):
    st.error(f"O ficheiro '{EXCEL_FILE}' não foi encontrado na pasta onde este script está guardado.")
    st.stop()

# --- NET CASH PER BANK (Sheet7, linhas 77 a 89, colunas B e C) ---
sheet7 = pd.read_excel(EXCEL_FILE, sheet_name="Sheet7", header=None)
net_cash_df = sheet7.iloc[77:90, [1, 2]].copy()
net_cash_df.columns = ["Banco", "Valor_EUR"]
net_cash_df.dropna(inplace=True)
net_cash_df.reset_index(drop=True, inplace=True)

# --- CASH EOW (ler todas as colunas com dados nas linhas 31 e 32) ---
dash_sheet = pd.read_excel(EXCEL_FILE, sheet_name="Information to feed dash", header=None)
eow_labels = dash_sheet.iloc[30].dropna().tolist()
eow_values = dash_sheet.iloc[31].dropna().tolist()

# Tentar converter os labels em datas (se possível)
try:
    eow_labels = pd.to_datetime(eow_labels, dayfirst=True)
except:
    pass

eow_data = pd.Series(eow_values, index=eow_labels) / 1_000_000  # Converter para milhões

# --- Layout e título ---
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .small-font { font-size:12px !important; }
    .card { padding: 0.5rem 1rem; border: 1px solid #ccc; border-radius: 0.5rem; margin-bottom: 1rem; background-color: #f9f9f9; }
    </style>
""", unsafe_allow_html=True)

from datetime import datetime
hoje = datetime.today().strftime('%d/%m/%Y')
st.markdown(f"<h1 style='text-align:center; font-size:26px; font-weight:bold;'>Unit4 Cash Daily Position Dashboard - {hoje}</h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# --- Linha superior: Cash Balance, Cash EOW e Net Cash ---
col1, col2 = st.columns([2, 2])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-size:20px; font-weight:bold;'>CASH DAILY POSITION</h4>", unsafe_allow_html=True)
    total_balance = net_cash_df["Valor_EUR"].sum()
    st.metric(label="BALANCE", value=f"{total_balance:,.2f} EUR")

    st.markdown("<h6 class='small-font'>CASH EOW</h6>", unsafe_allow_html=True)
    fig_eow, ax_eow = plt.subplots(figsize=(6.5, 1.7))
    ax_eow.plot(eow_data.index, eow_data.values, color='green', linewidth=1)
    ax_eow.set_ylabel("Milhões", fontsize=8)
    ax_eow.set_ylim(0, 200)
    ax_eow.set_title("CASH EOW", fontsize=10, weight='bold')
    ax_eow.tick_params(axis='x', labelrotation=45, labelsize=7)
    ax_eow.tick_params(axis='y', labelsize=7)
    ax_eow.set_xlim([min(eow_data.index) - pd.Timedelta(days=1), max(eow_data.index) + pd.Timedelta(days=1)])
    ax_eow.set_xticks(eow_data.index)
    ax_eow.xaxis.set_major_formatter(mdates.DateFormatter('%d/%b'))
    buf_eow = io.BytesIO()
    fig_eow.savefig(buf_eow, format='png', bbox_inches='tight', dpi=150)
    st.image(buf_eow.getvalue(), use_container_width=False)
    buf_eow.close()
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h6 class='small-font'>NET CASH PER BANK</h6>", unsafe_allow_html=True)
    fig_bar, ax_bar = plt.subplots(figsize=(5.2, 2.4))
    ax_bar.barh(net_cash_df["Banco"], net_cash_df["Valor_EUR"], color="lightgreen")
    for i, v in enumerate(net_cash_df["Valor_EUR"]):
        ax_bar.text(v + max(net_cash_df["Valor_EUR"]) * 0.01, i, f"{v:,.0f}".replace(",", "."), va='center', fontsize=6)
    ax_bar.set_ylabel("")
    ax_bar.set_title("", fontsize=9)
    ax_bar.tick_params(labelsize=6)
    ax_bar.get_xaxis().set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_bar)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Coluna 2: Financial Activities ---




# --- Preparar dados do gráfico Pie ---
from datetime import datetime
month_now = datetime.now().strftime('%b').capitalize()

term_df = pd.read_excel(EXCEL_FILE, sheet_name="Term Deposits", header=None)
header_row = term_df.iloc[4, 16:24].tolist()

if month_now in header_row:
    col_index = 16 + header_row.index(month_now)
    mmf_value = term_df.iloc[5, col_index] if not pd.isna(term_df.iloc[5, col_index]) else 0
    td_value = term_df.iloc[6, col_index] if not pd.isna(term_df.iloc[6, col_index]) else 0
else:
    mmf_value = td_value = 0

labels = ['MMF', 'TD']
sizes = [0 if pd.isna(mmf_value) or mmf_value == 0 else mmf_value, 0 if pd.isna(td_value) or td_value == 0 else td_value]
if sum(sizes) == 0:
    sizes = [1e-5, 1e-5]
colors = ['green', 'lightblue']

# --- CASHFLOW FORECAST ---
cashflow_months = dash_sheet.iloc[2, 1:13].tolist()
inflow = dash_sheet.iloc[5, 1:13].tolist()
outflow = dash_sheet.iloc[6, 1:13].tolist()
netflow = dash_sheet.iloc[4, 1:13].tolist()

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align:center; font-size:18px; font-weight:bold;'>Cashflow Forecast</h5>", unsafe_allow_html=True)
fig_cf, ax_cf = plt.subplots(figsize=(6.5, 2.5))

bar_width = 0.35
x = range(len(cashflow_months))

ax_cf.bar([i - bar_width/2 for i in x], inflow, width=bar_width, color='green', edgecolor='black', linewidth=0.5, label='Inflow')
ax_cf.bar([i + bar_width/2 for i in x], outflow, width=bar_width, color='skyblue', edgecolor='black', linewidth=0.5, label='Outflow')
ax_cf.plot(x, netflow, color='red', linewidth=1.5, label='Net Flow')

for i, v in enumerate(inflow):
    ax_cf.text(i - bar_width/2, v + 1500, f'{v:,.0f}'.replace(',', ' '), ha='center', fontsize=5, va='bottom')
for i, v in enumerate(outflow):
    ax_cf.text(i + bar_width/2, v - 1500, f'{v:,.0f}'.replace(',', ' '), ha='center', fontsize=5, va='top')

ax_cf.set_xticks(x)
ax_cf.set_xticklabels(cashflow_months)
ax_cf.axhline(0, color='black', linewidth=0.5)
ax_cf.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=6)
ax_cf.tick_params(axis='both', labelsize=6)

plt.tight_layout()
st.pyplot(fig_cf)
st.markdown("</div>", unsafe_allow_html=True)

# --- Gráfico Pie abaixo ---
st.markdown("<div class='card'>", unsafe_allow_html=True)
fig_pie, ax_pie = plt.subplots(figsize=(1.2, 0.6), dpi=150)
ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.0fM€', textprops={'fontsize': 6})
ax_pie.axis('equal')
buf = io.BytesIO()
fig_pie.savefig(buf, format='png', bbox_inches='tight', transparent=True)
st.image(buf.getvalue(), use_container_width=False)
buf.close()
st.markdown("</div>", unsafe_allow_html=True)

# --- Tabela COLECTIONS ---
collections_df = pd.read_excel(EXCEL_FILE, sheet_name="Colections", header=None)
collections_df = collections_df.iloc[3:33, 16:20].copy()
collections_df.columns = ["Entity", "Forecast", "Received", "%"]
collections_df.dropna(how="all", inplace=True)
collections_df = collections_df[collections_df["Entity"] != "Entity"]
collections_df.reset_index(drop=True, inplace=True)
collections_df["Forecast"] = pd.to_numeric(collections_df["Forecast"], errors="coerce").fillna(0)
collections_df["Received"] = pd.to_numeric(collections_df["Received"], errors="coerce").fillna(0)
collections_df["%"] = collections_df.apply(lambda row: f"{(row['Received'] / row['Forecast'] * 100):.2f}%" if row['Forecast'] > 0 else "0,00%", axis=1)
st.markdown("<h5 style='text-align:center; font-size:18px; font-weight:bold;'>Incoming Cash Position</h5>", unsafe_allow_html=True)
collections_df["Forecast"] = collections_df["Forecast"].apply(lambda x: f"{x:,.0f}".replace(",", " "))
collections_df["Received"] = collections_df["Received"].apply(lambda x: f"{x:,.0f}".replace(",", " "))
st.dataframe(collections_df.style.set_table_styles([
    {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold')]},
    {'selector': 'td', 'props': [('padding', '0rem 0.5rem')]}]).set_properties(**{'text-align': 'center'}), height=600, use_container_width=False)














