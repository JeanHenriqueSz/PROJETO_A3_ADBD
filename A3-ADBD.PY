# ==========================================================
# TRABALHO FINAL – ANÁLISE DE DADOS E BIG DATA
# OLIST DATASET (2016–2018)
# Autor: Jean Henrique
# ==========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1. CONFIGURAÇÕES INICIAIS DE ESTILO
# ----------------------------------------------------------

# Estilo mais bonito para todos os gráficos (grid suave e cores modernas)
plt.style.use("seaborn-v0_8-darkgrid")

# Tamanho padrão dos gráficos
plt.rcParams["figure.figsize"] = (12, 6)

# Tamanho de textos
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

# Caminho onde estão os datasets da Olist
DATA_PATH = r"C:\Users\Jeanh\Desktop\testes\PROJETO_A3_ADBD\arquivos\Olist_DataSets"

# ----------------------------------------------------------
# 2. CARREGAMENTO DOS DATASETS
# ----------------------------------------------------------
# Carregamos cada uma das 8 tabelas do projeto.
# O professor exige mínimo 3, então estamos bem acima disso.

customers = pd.read_csv(os.path.join(DATA_PATH, "olist_customers(Clientes)_dataset.csv"))
orders    = pd.read_csv(os.path.join(DATA_PATH, "olist_orders(Pedidos)_dataset.csv"))
items     = pd.read_csv(os.path.join(DATA_PATH, "olist_order_items(itens_pedido)_dataset.csv"))
products  = pd.read_csv(os.path.join(DATA_PATH, "olist_products(Produtos)_dataset.csv"))
sellers   = pd.read_csv(os.path.join(DATA_PATH, "olist_sellers(Vendedores)_dataset.csv"))
payments  = pd.read_csv(os.path.join(DATA_PATH, "olist_order_payments(pagamento_pedido)_dataset.csv"))
reviews   = pd.read_csv(os.path.join(DATA_PATH, "olist_order_reviews(avaliacao_pedido)_dataset.csv"))
geo       = pd.read_csv(os.path.join(DATA_PATH, "olist_geolocation(GeoLocalizacao)_dataset.csv"))

print("Datasets carregados com sucesso.")

# ----------------------------------------------------------
# 3. TRATAMENTO DE DATAS E CRIAÇÃO DE COLUNAS
# ----------------------------------------------------------

# Lista de colunas que devem ser convertidas para datetime
date_cols = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

# Converte datas para o tipo datetime
for c in date_cols:
    orders[c] = pd.to_datetime(orders[c])

# Datas das avaliações
reviews["review_creation_date"]   = pd.to_datetime(reviews["review_creation_date"])
reviews["review_answer_timestamp"] = pd.to_datetime(reviews["review_answer_timestamp"])

# Cria colunas úteis para análises
orders["purchase_year"]  = orders["order_purchase_timestamp"].dt.year
orders["purchase_month"] = orders["order_purchase_timestamp"].dt.to_period("M")

# Tempo total de entrega em dias
orders["delivery_time_days"] = (
    orders["order_delivered_customer_date"] 
    - orders["order_purchase_timestamp"]
).dt.days

# Atraso em relação ao prazo estimado
orders["delay_vs_estimated"] = (
    orders["order_delivered_customer_date"]
    - orders["order_estimated_delivery_date"]
).dt.days

# Flag indicando se houve atraso
orders["is_late"] = orders["delay_vs_estimated"] > 0

# ----------------------------------------------------------
# 4. DATAFRAME FATO COMPLETO (JOIN GERAL)
# ----------------------------------------------------------
# Aqui juntamos todas as bases usando os IDs de ligação.

orders_full = (
    orders
    .merge(customers, on="customer_id", how="left")
    .merge(items, on="order_id", how="left")
    .merge(products, on="product_id", how="left")
    .merge(sellers, on="seller_id", how="left")
    .merge(payments.groupby("order_id")["payment_value"].sum().reset_index(), on="order_id", how="left")
    .merge(reviews[["order_id", "review_score"]], on="order_id", how="left")
)

print("DataFrame unificado criado:", orders_full.shape)

# ==========================================================
# 5. ANÁLISE 1 – PERFIL DOS CLIENTES
# ==========================================================

# Agrupamos por estado para saber onde os clientes compram mais
orders_per_state = (
    orders_full.groupby("customer_state")["order_id"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nTop estados por número de pedidos:")
print(orders_per_state.head())

# ---------- GRÁFICO BONITO ----------
plt.figure()
bars = plt.bar(orders_per_state.index, orders_per_state.values, color="#4C72B0")

plt.title("Pedidos por Estado dos Clientes (2016–2018)")
plt.xlabel("Estado")
plt.ylabel("Quantidade de Pedidos")

# Adiciona valores acima das barras
for b in bars:
    plt.text(
        b.get_x() + b.get_width() / 2,
        b.get_height(),
        str(b.get_height()),
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Insight explicativo
print("""
INSIGHT:
Os estados com maior número de pedidos são SP, RJ e MG.
Isso reflete a concentração populacional e a força econômica do Sudeste.
""")

# ==========================================================
# 6. ANÁLISE 2 – CATEGORIAS DE PRODUTOS
# ==========================================================

# Agrupamento por categorias
category_stats = (
    orders_full
    .groupby("product_category_name")
    .agg(
        qtde_itens=("order_item_id", "count"),
        receita=("price", "sum"),
        preco_medio=("price", "mean")
    )
    .sort_values(by="qtde_itens", ascending=False)
)

print("\nTop categorias mais vendidas:")
print(category_stats.head())

# ---------- GRÁFICO BONITO ----------
plt.figure()
bars = plt.bar(category_stats.head(10).index, category_stats.head(10)["qtde_itens"], color="#55A868")

plt.title("Top 10 Categorias mais Vendidas")
plt.xlabel("Categoria")
plt.ylabel("Quantidade de Itens")

# Rotação para caber os nomes
plt.xticks(rotation=70)

# Valores acima das barras
for b in bars:
    plt.text(
        b.get_x() + b.get_width() / 2,
        b.get_height(),
        int(b.get_height()),
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()
plt.show()

print("""
INSIGHT:
A categoria 'cama_mesa_banho' lidera com folga, indicando alta demanda por produtos domésticos.
Outras categorias fortes incluem 'beleza_saude' e 'esporte_lazer'.
""")

# ==========================================================
# 7. ANÁLISE 3 – VENDEDORES
# ==========================================================

sellers_per_state = (
    sellers.groupby("seller_state")["seller_id"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nQuantidade de vendedores por estado:")
print(sellers_per_state)

# ---------- GRÁFICO BONITO ----------
plt.figure()
bars = plt.bar(sellers_per_state.index, sellers_per_state.values, color="#C44E52")

plt.title("Distribuição de Vendedores por Estado")
plt.xlabel("Estado")
plt.ylabel("Quantidade de Vendedores")
plt.xticks(rotation=45)

# Valores acima das barras
for b in bars:
    plt.text(
        b.get_x() + b.get_width()/2,
        b.get_height(),
        b.get_height(),
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()
plt.show()

print("""
INSIGHT:
Os vendedores também se concentram no Sudeste, especialmente em São Paulo,
mostrando forte centralização logística e comercial.
""")

# ==========================================================
# 8. ANÁLISE 4 – AVALIAÇÕES E ATRASOS
# ==========================================================

# Distribuição das notas
review_dist = (
    reviews["review_score"]
    .value_counts()
    .sort_index()
)

# ---------- GRÁFICO BONITO ----------
plt.figure()
bars = plt.bar(review_dist.index, review_dist.values, color="#8172B2")

plt.title("Distribuição das Notas de Avaliação dos Clientes")
plt.xlabel("Nota")
plt.ylabel("Quantidade")

# Valores acima das barras
for b in bars:
    plt.text(
        b.get_x() + b.get_width() / 2,
        b.get_height(),
        int(b.get_height()),
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.xticks(review_dist.index)
plt.tight_layout()
plt.show()

print("""
INSIGHT:
Notas baixas (1 e 2) são comuns em pedidos atrasados. 
A maioria das avaliações é positiva (nota 4 e 5).
""")

# Média das notas considerando atraso/no atraso
nota_media_atraso = orders_full.groupby("is_late")["review_score"].mean()

# ---------- GRÁFICO BONITO ----------
plt.figure()
bars = plt.bar(
    ["Sem Atraso", "Com Atraso"],
    nota_media_atraso.values,
    color=["#4C72B0", "#C44E52"]
)

plt.title("Nota Média x Atraso na Entrega")
plt.ylabel("Nota Média")

# Valores acima das barras
for i, b in enumerate(bars):
    plt.text(
        b.get_x() + b.get_width()/2,
        b.get_height(),
        f"{b.get_height():.2f}",
        ha="center",
        va="bottom",
        fontsize=12
    )

plt.tight_layout()
plt.show()

print("""
INSIGHT FINAL:
Pedidos entregues no prazo têm uma nota média MUITO superior.
O tempo de entrega influencia diretamente na satisfação do cliente.
""")