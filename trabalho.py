import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import datetime

# Configurando a semente para reprodutibilidade
np.random.seed(42)

# Parâmetros e criação do DataFrame
n = 100
data = {
    'Categoria': np.random.choice(['econômico', 'compacto', 'SUV'], n),
    'Numero_Passageiros': np.random.randint(2, 8, n),
    'Capacidade_Porta_Malas': np.random.randint(1, 6, n),
    'Ar_Condicionado': np.random.choice(['Sim', 'Não'], n),
    'Tipo_Cambio': np.random.choice(['Automático', 'Manual'], n),
}

# Cálculo do valor do aluguel
def calcular_valor_aluguel(row):
    base_price = 30 + (row['Numero_Passageiros'] - 2) * 5 + (row['Capacidade_Porta_Malas'] - 1) * 5
    return base_price + (50 if row['Categoria'] == 'SUV' else 20 if row['Categoria'] == 'compacto' else 0) + np.random.uniform(-10, 10)

data['Valor_Aluguel'] = [calcular_valor_aluguel(row) for _, row in pd.DataFrame(data).iterrows()]
df = pd.DataFrame(data)

# Transformação e preparação dos dados
X = df.drop('Valor_Aluguel', axis=1)
y = df['Valor_Aluguel']
X = pd.get_dummies(X, drop_first=True)

# Dividindo os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizando os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinando e avaliando o modelo
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Resultados
results = pd.DataFrame({'Valor Real': y_test, 'Valor Previsto': y_pred})
results['Valor Real'] = results['Valor Real'].apply(lambda x: f'R$ {x:.2f}')
results['Valor Previsto'] = results['Valor Previsto'].apply(lambda x: f'R$ {x:.2f}')
print(results.head())

# Salvando o dataset em um arquivo Excel com formatação
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'dataset_carros_{timestamp}.xlsx'

with pd.ExcelWriter(output_file) as writer:
    df.to_excel(writer, sheet_name='Dados', index=False)
    results.to_excel(writer, sheet_name='Resultados', index=False)

print(f'Dataset salvo como {output_file}')
