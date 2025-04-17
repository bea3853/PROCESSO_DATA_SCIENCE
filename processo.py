import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


url = "https://bea3853.github.io/PROCESSO_DATA_SCIENCE/"
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')


nomes = []
precos = []
avaliacoes = []


for produto in soup.find_all('div', class_='produto'):
    nomes.append(produto.find('h2').text)
    precos.append(float(produto.find('span', class_='preco').text.replace('R$', '').replace('.', '').replace(',', '.')))
    avaliacoes.append(float(produto.find('span', class_='avaliacao').text))


df = pd.DataFrame({
    'Modelo': nomes,
    'Preco': precos,
    'Avaliacao': avaliacoes
})

# Salvando em CSV (opcional)
df.to_csv('smartphones.csv', index=False)

#### 2. Limpeza e Análise Exploratória (Pandas + NumPy)  

# Carregar dados (se não vier do scraping)
df = pd.read_csv('smartphones.csv')

# Verificar dados faltantes
print(df.isnull().sum())

# Limpeza: Remover duplicatas e outliers
df = df.drop_duplicates()
df = df[df['Preco'] < 10000]  # Filtrar preços absurdos

# Extrair marca do modelo (ex.: "iPhone 15" -> "Apple")
df['Marca'] = df['Modelo'].str.split().str[0]

# Estatísticas básicas
print(df.describe())

# Preço médio por marca
preco_medio = df.groupby('Marca')['Preco'].mean().sort_values(ascending=False)
print(preco_medio)




# 3. Visualização com Matplotlib  

# Configurar estilo
# plt.style.use('seaborn')

# Gráfico 1: Distribuição de preços
plt.figure(figsize=(10, 6))
plt.hist(df['Preco'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribuição de Preços de Smartphones')
plt.xlabel('Preço (R$)')
plt.ylabel('Quantidade')
plt.grid(True)
plt.show()

# Gráfico 2: Preço médio por marca
plt.figure(figsize=(12, 6))
preco_medio.plot(kind='bar', color='orange')
plt.title('Preço Médio por Marca')
plt.xlabel('Marca')
plt.ylabel('Preço Médio (R$)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Gráfico 3: Relação Preço x Avaliação
plt.figure(figsize=(10, 6))
plt.scatter(df['Preco'], df['Avaliacao'], alpha=0.6, color='green')
plt.title('Relação entre Preço e Avaliação')
plt.xlabel('Preço (R$)')
plt.ylabel('Avaliação (1-5)')
plt.grid(True)
plt.show()




# Insight: 
#   A - Distribuição de Preços:  
#    - A maioria dos smartphones está na faixa de R$ 1.000 a R$ 3.000.  
#    - Poucos produtos premium (acima de R$ 5.000).  

# B  -  Marcas Dominantes:  
#    - Apple (iPhone) tem o preço médio mais alto.  
#    - Samsung e Xiaomi dominam a faixa intermediária.  

# C  -  Relação Preço x Avaliação:  
#    - Produtos mais caros nem sempre têm melhor avaliação.  
#    - Boas opções de custo-benefício podem ser encontradas na faixa de R$ 1.500 a R$ 2.500.  



# Recomendações para a Empresa  
# Aumentar estoque de marcas com melhor custo-benefício (ex.: Xiaomi).  
#  Promoções estratégicas em produtos com alta avaliação e preço médio.  
# monitorar concorrentes para ajustar preços de iPhones e outros premium.  

#  Próximos Passos (approfundando)
# aplicar ml para prever tendências de preço.  
 

