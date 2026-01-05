import pandas as pd
import glob
import os
import numpy as np
from pathlib import Path

# Configurações
BASE_DIR = Path(r"D:\2025\UFJF\ETLSIH")
COD_JF = 313670 

def construir_base_master():
    print(f"📂 Iniciando consolidação da Base Master em: {BASE_DIR}")
    arquivos = glob.glob(str(BASE_DIR / "ETLSIH.ST_MG_*_t.csv"))
    arquivos.sort()

    colunas_sih = ['N_AIH', 'MUNIC_MOV', 'DT_INTER', 'DT_SAIDA', 'DIAS_PERM', 'IDADE', 'COMPLEX', 'CGC_HOSP']
    master_list = []

    for arquivo in arquivos:
        try:
            df = pd.read_csv(arquivo, usecols=colunas_sih, low_memory=False)
            df_jf = df[df['MUNIC_MOV'] == COD_JF].copy()
            if df_jf.empty: continue

            # 1. Tratamento de Datas
            df_jf['DT_INTER_DT'] = pd.to_datetime(df_jf['DT_INTER'], format='%Y%m%d', errors='coerce')
            df_jf = df_jf.dropna(subset=['DT_INTER_DT', 'DIAS_PERM'])

            # 2. Extração de Ano/Mês
            df_jf['ano_internacao'] = df_jf['DT_INTER_DT'].dt.year
            df_jf['mes_internacao'] = df_jf['DT_INTER_DT'].dt.month

            # 3. Cálculo do Tempo de Chegada (Horas dentro do próprio mês)
            df_jf['inicio_mes'] = df_jf['DT_INTER_DT'].apply(lambda x: x.replace(day=1, hour=0, minute=0))
            df_jf['tempo_chegada_hora'] = (df_jf['DT_INTER_DT'] - df_jf['inicio_mes']).dt.total_seconds() / 3600

            # 4. Cálculo do Tempo Estimado UTI (Oráculo)
            df_jf['tempo_estimado_uti_horas'] = df_jf['DIAS_PERM'].astype(float) * 24
            df_jf.loc[df_jf['tempo_estimado_uti_horas'] <= 0, 'tempo_estimado_uti_horas'] = 24

            # 5. Cálculo da Gravidade Score
            def calc_grav(row):
                s = 6 if str(row['COMPLEX']) == '03' else 3
                if row['IDADE'] > 70: s += 4
                return min(10, s)
            df_jf['gravidade_score'] = df_jf.apply(calc_grav, axis=1)

            # Seleção das colunas
            df_step = df_jf[[
                'N_AIH', 'tempo_chegada_hora', 'gravidade_score', 'tempo_estimado_uti_horas',
                'ano_internacao', 'mes_internacao', 'CGC_HOSP'
            ]]
            df_step.columns = ['id_paciente', 'tempo_chegada_hora', 'gravidade_score', 
                              'tempo_estimado_uti_horas', 'ano_internacao', 'mes_internacao', 'cnpj_unidade']
            
            master_list.append(df_step)
            print(f"   ✅ {df_jf['ano_internacao'].iloc[0]}-{df_jf['mes_internacao'].iloc[0]} lido.")

        except Exception as e:
            print(f"   ❌ Erro no arquivo {os.path.basename(arquivo)}: {e}")

    # --- CONSOLIDAÇÃO E LIMPEZA DE DUPLICATAS ADMINISTRATIVAS ---
    print("\n🧹 Iniciando limpeza de duplicatas de faturamento...")
    df_master = pd.concat(master_list, ignore_index=True)
    
    # REGRA DE OURO: Se o mesmo paciente chega na mesma hora na mesma unidade, 
    # consolidamos pegando a maior gravidade e a maior permanência (evita fracionamento)
    total_antes = len(df_master)
    df_master = df_master.groupby([
        'id_paciente', 'tempo_chegada_hora', 'ano_internacao', 'mes_internacao', 'cnpj_unidade'
    ]).agg({
        'gravidade_score': 'max',
        'tempo_estimado_uti_horas': 'max'
    }).reset_index()
    
    # Limpeza de Outliers (Acima de 90 dias)
    df_master = df_master[df_master['tempo_estimado_uti_horas'] <= 2160]
    
    total_depois = len(df_master)
    print(f"✨ Limpeza concluída: {total_antes - total_depois} registros redundantes removidos.")
    
    df_master.to_csv("BASE_MASTER_CONSULTA.csv", index=False, encoding='utf-8-sig')
    print(f"🏆 Base Master final gerada com {total_depois} registros únicos.")

    # --- RANKING DE CENÁRIOS ---
    print("\n🧐 Reanalisando cenários para teste de 30 dias...")
    
    # Little's Law para estimar leitos necessários (Pacientes * Média Permanência / 720h)
    ranking = df_master.groupby(['ano_internacao', 'mes_internacao', 'cnpj_unidade']).agg({
        'id_paciente': 'count',
        'gravidade_score': ['mean', 'std'],
        'tempo_estimado_uti_horas': 'mean'
    }).reset_index()
    
    ranking.columns = ['ano', 'mes', 'cnpj', 'n_pacientes', 'grav_media', 'grav_std', 'perm_media']
    
    # Filtro: Hospitais com volume relevante e diversidade clínica
    melhores_opcoes = ranking[
        (ranking['n_pacientes'] >= 40) & 
        (ranking['grav_std'] > 1.2)
    ].sort_values(by='n_pacientes', ascending=False)

    print("\n🎯 TOP 3 CENÁRIOS REAIS PARA O EXPERIMENTO:")
    for i, row in melhores_opcoes.head(3).iterrows():
        leitos_necessarios = (row['n_pacientes'] * row['perm_media']) / 720
        print(f"- Hospital {row['cnpj']} em {int(row['ano'])}/{int(row['mes'])}:")
        print(f"  * {row['n_pacientes']} internações únicas no mês.")
        print(f"  * Gravidade Média: {row['grav_media']:.2f} (±{row['grav_std']:.2f})")
        print(f"  * Leitos necessários para Espera Zero: {int(leitos_necessarios)}")
        print(f"  * Sugestão de Estresse para Artigo: {int(leitos_necessarios * 0.8)} leitos.")
        print("-" * 40)

if __name__ == "__main__":
    construir_base_master()