import pandas as pd
import glob
import os
import numpy as np
from pathlib import Path
from datetime import timedelta

# Configurações
BASE_DIR = Path(r"D:\2025\UFJF\ETLSIH")
COD_JF = 313670 

def realizar_data_profiling():
    print(f"🔍 Iniciando Profiling Estatístico em: {BASE_DIR}")
    arquivos = glob.glob(str(BASE_DIR / "ETLSIH.ST_MG_*_t.csv"))
    arquivos.sort()

    # Colunas necessárias para o otimizador e para a análise gerencial
    cols = ['MUNIC_MOV', 'DT_INTER', 'DT_SAIDA', 'DIAS_PERM', 'IDADE', 'COMPLEX', 'DIAG_PRINC', 'CGC_HOSP']
    
    stats_periodo = []
    perfil_unidades = []

    for arquivo in arquivos:
        try:
            df = pd.read_csv(arquivo, usecols=cols, low_memory=False)
            df_jf = df[df['MUNIC_MOV'] == COD_JF].copy()
            if df_jf.empty: continue

            # Tratamento de Datas
            df_jf['DT_INTER'] = pd.to_datetime(df_jf['DT_INTER'], format='%Y%m%d', errors='coerce')
            df_jf['DT_SAIDA'] = pd.to_datetime(df_jf['DT_SAIDA'], format='%Y%m%d', errors='coerce')
            df_jf = df_jf.dropna(subset=['DT_INTER', 'DIAS_PERM'])
            
            # Cálculo de Variáveis do Otimizador
            df_jf['LOS_H'] = df_jf['DIAS_PERM'].astype(float) * 24
            df_jf.loc[df_jf['LOS_H'] <= 0, 'LOS_H'] = 24
            
            def calc_grav(row):
                s = 6 if str(row['COMPLEX']) == '03' else 3
                if row['IDADE'] > 70: s += 4
                return min(10, s)
            df_jf['grav_score'] = df_jf.apply(calc_grav, axis=1)

            # Agrupamento por Mês/Ano
            ano = df_jf['DT_INTER'].dt.year.iloc[0]
            mes = df_jf['DT_INTER'].dt.month.iloc[0]
            
            # 1. ANÁLISE CATEGÓRICA (CID-10)
            top_cids = df_jf['DIAG_PRINC'].value_counts().head(5).to_dict()

            # 2. ESTIMATIVA DE LEITOS (OCUPAÇÃO SIMULTÂNEA MÁXIMA)
            # Criamos uma linha do tempo para contar pacientes ao mesmo tempo
            def estimar_pico(df_unidade):
                if df_unidade.empty: return 0
                eventos = []
                for _, r in df_unidade.iterrows():
                    eventos.append((r['DT_INTER'], 1))
                    eventos.append((r['DT_SAIDA'], -1))
                ev_df = pd.DataFrame(eventos, columns=['tempo', 'val']).sort_values('tempo')
                return ev_df['val'].cumsum().max()

            # 3. ANÁLISE POR UNIDADE (CGC_HOSP)
            for cnpj, grupo in df_jf.groupby('CGC_HOSP'):
                pico = estimar_pico(grupo)
                perfil_unidades.append({
                    'Ano': ano, 'Mes': mes, 'Hospital_CNPJ': cnpj,
                    'Qtd_Pacientes': len(grupo),
                    'Pico_Simultaneo_Estimado': pico,
                    'LOS_Medio_H': grupo['LOS_H'].mean(),
                    'Gravidade_Media': grupo['grav_score'].mean()
                })

            # 4. ESTATÍSTICA GERAL DO PERÍODO
            stats_periodo.append({
                'Ano': ano, 'Mes': mes,
                'N': len(df_jf),
                'Idade_Media': df_jf['IDADE'].mean(),
                'Idade_Std': df_jf['IDADE'].std(),
                'LOS_Medio_H': df_jf['LOS_H'].mean(),
                'LOS_Max_H': df_jf['LOS_H'].max(),
                'Gravidade_Media': df_jf['grav_score'].mean(),
                'Gravidade_Std': df_jf['grav_score'].std(),
                'Top_CIDs': str(top_cids)
            })
            
            print(f"✅ Processado {ano}-{mes:02d} | Unidades JF: {df_jf['CGC_HOSP'].nunique()}")

        except Exception as e:
            print(f"❌ Erro no arquivo {os.path.basename(arquivo)}: {e}")

    # Consolidação
    df_geral = pd.DataFrame(stats_periodo)
    df_hospitais = pd.DataFrame(perfil_unidades)
    
    df_geral.to_csv("SINTESE_ESTATISTICA_JF.csv", index=False, encoding='utf-8-sig')
    df_hospitais.to_csv("ESTIMATIVA_LEITOS_POR_HOSPITAL.csv", index=False, encoding='utf-8-sig')
    
    # Insights para o Cristiano
    print("\n" + "="*50)
    print("💎 INSIGHTS PARA O DIMENSIONAMENTO")
    print("="*50)
    print(f"Total de Unidades de Saúde detectadas em JF: {df_hospitais['Hospital_CNPJ'].nunique()}")
    
    # Estimativa média de leitos em JF
    leitos_totais_estimados = df_hospitais.groupby(['Ano', 'Mes'])['Pico_Simultaneo_Estimado'].sum().max()
    print(f"Estimativa de Pico de Leitos Necessários (JF Total): {leitos_totais_estimados}")
    
    media_los = df_geral['LOS_Medio_H'].mean() / 24
    print(f"Permanência Média na Rede: {media_los:.2f} dias")
    
    print("\nArquivos gerados: 'SINTESE_ESTATISTICA_JF.csv' e 'ESTIMATIVA_LEITOS_POR_HOSPITAL.csv'")

if __name__ == "__main__":
    realizar_data_profiling()