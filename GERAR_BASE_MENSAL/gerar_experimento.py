import pandas as pd

# Carrega a base limpa que acabamos de gerar
df_master = pd.read_csv("BASE_MASTER_CONSULTA.csv")

# Filtra o Hospital e o Mês escolhido
hospital_alvo = 21583042000172.0
ano_alvo = 2023
mes_alvo = 3

df_final = df_master[
    (df_master['cnpj_unidade'] == hospital_alvo) & 
    (df_master['ano_internacao'] == ano_alvo) & 
    (df_master['mes_internacao'] == mes_alvo)
].copy()

# Ordena por chegada para facilitar a leitura do Gantt
df_final = df_final.sort_values('tempo_chegada_hora')

# Salva a base para o otimizador
df_final.to_csv("base_final_30dias.csv", index=False)

print(f"✅ Base Final Gerada: 'base_final_30dias.csv'")
print(f"   Total de Pacientes: {len(df_final)}")
print(f"   Gravidade Média: {df_final['gravidade_score'].mean():.2f}")