import pandas as pd
import glob
import os
from pathlib import Path

# Configurações
BASE_DIR = Path(r"D:\2025\UFJF\ETLSIH")
COD_JUIZ_DE_FORA = 313670 
LEITOS = 12
HORAS_MES = 720 # 30 dias
CAPACIDADE_TOTAL = LEITOS * HORAS_MES # 8.640 horas-leito

def minerar_cenarios_cientificos():
    print(f"1. Minerando arquivos em: {BASE_DIR}")
    arquivos = glob.glob(str(BASE_DIR / "ETLSIH.ST_MG_*_t.csv"))
    colunas = ['N_AIH', 'MUNIC_MOV', 'DT_INTER', 'DIAS_PERM', 'IDADE', 'COMPLEX']
    lista_df = []
    
    for arquivo in arquivos:
        try:
            df_temp = pd.read_csv(arquivo, usecols=colunas, low_memory=False)
            df_jf = df_temp[df_temp['MUNIC_MOV'] == COD_JUIZ_DE_FORA].copy()
            if not df_jf.empty:
                lista_df.append(df_jf)
        except: continue

    df_total = pd.concat(lista_df)
    df_total['DT_INTER'] = pd.to_datetime(df_total['DT_INTER'], format='%Y%m%d', errors='coerce')
    df_total = df_total.dropna(subset=['DT_INTER', 'DIAS_PERM'])
    
    # Identifica o mês de maior movimento para servir de base
    df_total['mes_ano'] = df_total['DT_INTER'].dt.to_period('M')
    mes_pico = df_total.groupby('mes_ano').size().idxmax()
    df_base = df_total[df_total['mes_ano'] == mes_pico].copy()
    
    print(f"2. Mês de referência: {mes_pico}")

    # Engenharia de Score (0 a 10)
    def calc_gravity(row):
        s = 6 if str(row['COMPLEX']) == '03' else 3
        if row['IDADE'] > 70: s += 4
        return min(10, s)
    df_base['gravidade_score'] = df_base.apply(calc_gravity, axis=1)

    def gerar_e_validar_cenario(alvo_ocupacao, nome_arq):
        # Ocupação Alvo em horas
        horas_alvo = CAPACIDADE_TOTAL * alvo_ocupacao
        
        # Embaralha os pacientes para pegar uma amostra aleatória do mês
        df_pool = df_base.sample(frac=1, random_state=42).copy()
        
        selecionados = []
        horas_acumuladas = 0
        
        for _, pac in df_pool.iterrows():
            dias = float(pac['DIAS_PERM']) if float(pac['DIAS_PERM']) > 0 else 1.0
            horas_pac = dias * 24
            
            if horas_acumuladas + horas_pac <= horas_alvo * 1.1: # Margem de 10%
                selecionados.append(pac)
                horas_acumuladas += horas_pac
            
            if horas_acumuladas >= horas_alvo:
                break
        
        df_res = pd.DataFrame(selecionados)
        referencia = pd.to_datetime(f"{mes_pico.year}-{mes_pico.month:02d}-01")
        df_res['tempo_chegada_hora'] = (df_res['DT_INTER'] - referencia).dt.total_seconds() / 3600
        df_res['tempo_estimado_uti_horas'] = df_res['DIAS_PERM'].astype(float).apply(lambda x: x*24 if x>0 else 24)
        
        # Saída final
        out = df_res[['N_AIH', 'tempo_chegada_hora', 'gravidade_score', 'tempo_estimado_uti_horas']]
        out.columns = ['id_paciente', 'tempo_chegada_hora', 'gravidade_score', 'tempo_estimado_uti_horas']
        out = out.sort_values('tempo_chegada_hora')
        
        out.to_csv(f"{nome_arq}.csv", index=False)
        ocup_real = (horas_acumuladas / CAPACIDADE_TOTAL) * 100
        print(f"✓ {nome_arq}.csv: {len(out)} pacientes | Ocupação Teórica: {ocup_real:.1f}%")

    print("\n3. Criando cenários baseados na capacidade de 12 leitos:")
    # Underload: ~45% ocupação
    gerar_e_validar_cenario(0.45, 'base_real_underload')
    # Central: ~90% ocupação
    gerar_e_validar_cenario(0.90, 'base_real_central')
    # Overload: ~130% ocupação (força o estouro)
    gerar_e_validar_cenario(1.30, 'base_real_overload')

if __name__ == "__main__":
    minerar_cenarios_cientificos()