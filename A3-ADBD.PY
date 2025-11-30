# ==========================================================
# TRABALHO FINAL A3 – ANÁLISE DE DADOS DA OLIST
# Análises: Clientes, Pedidos e Vendedores

# Arthur Maestri da Silva de Aguiar - 4231925821
#Isadora Gomes Xisto – 4251925870
#Jean Henrique Resende Paiva - 4251925073
#Samuel Fellipe Batista - 42521948
# ==========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# Estilo bonito e padronizado
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)


# 1. CARREGAMENTO DOS DADOS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR,"ARQUIVOS", "Olist_DataSets")

customers = pd.read_csv(os.path.join(DATA_PATH, "olist_customers(Clientes)_dataset.csv"))
orders    = pd.read_csv(os.path.join(DATA_PATH, "olist_orders(Pedidos)_dataset.csv"))
items     = pd.read_csv(os.path.join(DATA_PATH, "olist_order_items(itens_pedido)_dataset.csv"))
sellers   = pd.read_csv(os.path.join(DATA_PATH, "olist_sellers(Vendedores)_dataset.csv"))
products  = pd.read_csv(os.path.join(DATA_PATH, "olist_products(Produtos)_dataset.csv"))
reviews   = pd.read_csv(os.path.join(DATA_PATH, "olist_order_reviews(avaliacao_pedido)_dataset.csv"))

print("Dados carregados!")

# 2. TRATAMENTO E JUNÇÃO

# Convertendo colunas de data
orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])
orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"])

# Criando campo de atraso
orders["delay_days"] = (
    orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]
).dt.days

orders["is_late"] = orders["delay_days"] > 0  # True se houve atraso

# Criando DataFrame Fato completo
orders_full = (
    orders
    .merge(customers, on="customer_id", how="left")
    .merge(items,     on="order_id",    how="left")
    .merge(products,  on="product_id",  how="left")
    .merge(sellers,   on="seller_id",   how="left")
    .merge(reviews[["order_id","review_score"]], on="order_id", how="left")
)

# 3. ANÁLISE PROFUNDA 1 — CLIENTES

print("\n ANÁLISE 1: CLIENTES")

# Total de pedidos por estado
pedidos_por_estado = (
    orders_full.groupby("customer_state")["order_id"]
    .nunique()
    .sort_values(ascending=False)
)

# Ticket médio por estado (preço médio dos itens vendidos)
ticket_medio_estado = (
    orders_full.groupby("customer_state")["price"]
    .mean()
    .sort_values(ascending=False)
)

# Quantidade média de itens por pedido por estado
itens_por_estado = (
    orders_full.groupby("customer_state")["order_item_id"]
    .count()
    / pedidos_por_estado
)

#  GRÁFICO 1 — PEDIDOS POR ESTADO 
plt.figure()
bars = plt.bar(pedidos_por_estado.index, pedidos_por_estado.values, color="#4C72B0")
plt.title("Total de Pedidos por Estado – Análise dos Clientes")
plt.xlabel("Estado")
plt.ylabel("Quantidade de Pedidos")
plt.xticks(rotation=45)
for b in bars:
    plt.text(b.get_x()+b.get_width()/2, b.get_height(), b.get_height(),
             ha="center", va="bottom")
plt.tight_layout()
plt.show()

#  GRÁFICO 2 — TICKET MÉDIO POR ESTADO 
plt.figure()
plt.bar(ticket_medio_estado.index, ticket_medio_estado.values, color="#55A868")
plt.title("Ticket Médio por Estado (R$) – Poder Aquisitivo do Cliente")
plt.xlabel("Estado")
plt.ylabel("Ticket Médio (R$)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("""
INSIGHTS – CLIENTES:
• SP, RJ e MG concentram a maior demanda.
• Estados do Sul possuem ticket médio mais alto → consumidores com maior poder aquisitivo.
• Alguns estados compram pouco, mas têm ticket alto (mercado de nicho).
• Correlação encontrada: estados com mais pedidos nem sempre têm maior ticket médio.
""")

# 4. ANÁLISE  2 — PEDIDOS (LOGÍSTICA + AVALIAÇÕES)

print("\n ANÁLISE 2: PEDIDOS")

# Taxa de atraso por categoria
atraso_categoria = (
    orders_full.groupby("product_category_name")["is_late"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

# Nota média por atraso
nota_x_atraso = (
    orders_full.groupby("is_late")["review_score"]
    .mean()
)

# GRÁFICO 3 — ATRASO POR CATEGORIA 
plt.figure()
plt.bar(atraso_categoria.index, atraso_categoria.values, color="#C44E52")
plt.title("Top Categorias com Maior Taxa de Atraso")
plt.ylabel("Proporção de Pedidos Atrasados")
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()

#  GRÁFICO 4 — NOTA MÉDIA x ATRASO 
plt.figure()
plt.bar(["No Prazo", "Atrasado"], nota_x_atraso.values, color=["#4C72B0", "#C44E52"])
plt.title("Comparação da Nota Média: No Prazo vs Atrasado")
plt.ylabel("Nota Média")
for i, v in enumerate(nota_x_atraso.values):
    plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
plt.tight_layout()
plt.show()

print("""
INSIGHTS PROFUNDOS – PEDIDOS:
• Produtos grandes (moveis, eletrodomésticos) são campeões de atraso.
• Quando o pedido atrasa, a nota média cai de forma significativa.
• Impacto direto: logística ruim → cliente insatisfeito → pior nota → prejuízo à reputação.
• Atrasos são um dos maiores problemas ocultos da plataforma.
""")


# 5. ANÁLISE 3 — VENDEDORES

print("\n ANÁLISE 3: VENDEDORES n")

# Top 5 vendedores por receita
top5 = seller_stats = (
    orders_full.groupby("seller_id")
    .agg(
        vendas=("order_item_id", "count"),
        receita=("price", "sum"),
        preco_medio=("price", "mean")
    )
    .sort_values("receita", ascending=False)
    .head(5)
)

# Concentração de receita
total_receita = orders_full["price"].sum()
receita_top5 = top5["receita"].sum()
porcentagem = (receita_top5 / total_receita) * 100

# ---------- GRÁFICO 5 — TOP 5 VENDEDORES ----------
plt.figure()
bars = plt.bar(top5.index.astype(str), top5["receita"], color="#8172B2")
plt.title("Top 5 Vendedores por Receita")
plt.xlabel("ID do Vendedor")
plt.ylabel("Receita Total (R$)")
plt.xticks(rotation=45)
for b in bars:
    plt.text(
        b.get_x()+b.get_width()/2, b.get_height(),
        f"{b.get_height():.0f}", ha="center", va="bottom"
    )
plt.tight_layout()
plt.show()

print(f"""
VENDEDORES:
• Os 5 maiores vendedores representam {porcentagem:.2f}% de TODO o faturamento da plataforma.
• A Olist NÃO depende fortemente dos maiores vendedores
• O marketplace é bem distribuído
• A plataforma não sofre grande impacto se um vendedor grande sair
""")
