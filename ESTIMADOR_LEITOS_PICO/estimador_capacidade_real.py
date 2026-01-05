import pandas as pd
import glob
import os
import numpy as np
from pathlib import Path

# Configurações
BASE_DIR = Path(r"D:\2025\UFJF\ETLSIH")
COD_JF = 313670 
HOSPITAL_ALVO = 104400000228.0

def calcular_capacidade_maxima_simultanea():
    print(f"🔍 Analisando ocupação simultânea em: {BASE_DIR}")
    # Busca arquivos t.csv (MG)
    arquivos = glob.glob(str(BASE_DIR / "ETLSIH.ST_MG_*_t.csv"))
    arquivos.sort()

    lista_df = []
    # Colunas necessárias
    cols_required = ['MUNIC_MOV', 'DT_INTER', 'DT_SAIDA', 'N_AIH', 'CGC_HOSP', 'DIAS_PERM']
    
    for arquivo in arquivos:
        try:
            df = pd.read_csv(arquivo, usecols=cols_required, low_memory=False)
            df_jf = df[df['MUNIC_MOV'] == COD_JF].copy()
            if not df_jf.empty:
                lista_df.append(df_jf)
                print(f"   Lido: {os.path.basename(arquivo)}")
        except Exception as e:
            continue

    if not lista_df:
        print("Erro: Nenhum dado encontrado no diretório informado.")
        return

    df_total = pd.concat(lista_df)
    
    # 1. Tratamento de Datas
    df_total['DT_INTER'] = pd.to_datetime(df_total['DT_INTER'], format='%Y%m%d', errors='coerce')
    df_total['DT_SAIDA'] = pd.to_datetime(df_total['DT_SAIDA'], format='%Y%m%d', errors='coerce')
    df_total = df_total.dropna(subset=['DT_INTER', 'DT_SAIDA'])
    
    # 2. Consolidação de AIHs (Tratando o erro de faturamento que você encontrou)
    # Agrupamos por AIH e Unidade para pegar a data real de entrada e a última data de saída
    print("🧹 Consolidando registros de faturamento duplicados...")
    df_total = df_total.groupby(['N_AIH', 'CGC_HOSP']).agg({
        'DT_INTER': 'min',
        'DT_SAIDA': 'max'
    }).reset_index()

    # 3. Cálculo do Pico por Unidade (Algoritmo de Eventos)
    def calcular_pico_hospital(df_hosp):
        # Criar eventos de entrada (+1) e saída (-1)
        entradas = pd.DataFrame({'tempo': df_hosp['DT_INTER'], 'peso': 1})
        saidas = pd.DataFrame({'tempo': df_hosp['DT_SAIDA'], 'peso': -1})
        eventos = pd.concat([entradas, saidas]).sort_values(by=['tempo', 'peso'], ascending=[True, False])
        
        # Soma acumulada (Ocupação simultânea histórica)
        eventos['ocupacao_no_momento'] = eventos['peso'].cumsum()
        return eventos['ocupacao_no_momento'].max()

    print("\n🏢 CENSO DE LEITOS (CAPACIDADE MÁXIMA SIMULTÂNEA OBSERVADA):")
    print("-" * 60)
    resumo_leitos = []
    
    for cnpj, grupo in df_total.groupby('CGC_HOSP'):
        pico_historico = calcular_pico_hospital(grupo)
        
        resumo_leitos.append({
            'CNPJ': str(cnpj),
            'Leitos_Estimados_Pico': int(pico_historico),
            'Total_Pacientes_Historico': len(grupo)
        })
        
    df_leitos = pd.DataFrame(resumo_leitos).sort_values('Leitos_Estimados_Pico', ascending=False)
    
    # Correção do display para print
    print(df_leitos.to_string(index=False))
    
    # 4. Foco no Hospital Alvo
    hosp_alvo_info = df_leitos[df_leitos['CNPJ'].str.contains(str(int(HOSPITAL_ALVO)))]
    if not hosp_alvo_info.empty:
        pico_alvo = hosp_alvo_info['Leitos_Estimados_Pico'].values[0]
        print("\n" + "*"*60)
        print(f"🎯 HOSPITAL FOCO: {HOSPITAL_ALVO}")
        print(f"   CAPACIDADE MÁXIMA DETECTADA: {pico_alvo} leitos")
        print(f"   SUGESTÃO PARA TESTE CENTRAL: {int(pico_alvo * 0.9)} leitos (90% de ocupação)")
        print(f"   SUGESTÃO PARA TESTE OVERLOAD: {int(pico_alvo * 0.7)} leitos (Criar gargalo)")
        print("*"*60)
    
    df_leitos.to_csv("CENSO_LEITOS_ESTIMADO_JF.csv", index=False)
    print("\n✅ Arquivo 'CENSO_LEITOS_ESTIMADO_JF.csv' gerado.")

if __name__ == "__main__":
    calcular_capacidade_maxima_simultanea()