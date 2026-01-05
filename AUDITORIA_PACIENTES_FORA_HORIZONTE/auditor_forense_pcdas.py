import pandas as pd
import glob
import os
from pathlib import Path

# ==============================================================================
# CONFIGURAÇÕES DE CAMINHO
# ==============================================================================
BASE_DIR = Path(r"D:\2025\UFJF\ETLSIH")
ARQUIVO_IDS_ENTRADA = "ids_para_auditoria.csv"  # Arquivo com a coluna 'id_paciente'
ARQUIVO_SAIDA = "RELATORIO_AUDITORIA_DETALHADA.csv"

# Lista de colunas para extração máxima de relevância (Audit Trail)
COLUNAS_RELEVANTES = [
    'N_AIH', 'DT_INTER', 'DT_SAIDA', 'DIAS_PERM', 'IDADE', 'SEXO', 
    'DIAG_PRINC', 'DIAG_SECUN', 'DIAGSEC1', 'DIAGSEC2', 
    'PROC_REA', 'VAL_TOT', 'VAL_UTI', 'MORTE', 
    'COMPLEX', 'CGC_HOSP', 'CAR_INT', 'MUNIC_RES'
]

def realizar_auditoria_forense():
    print(f"🕵️ Iniciando Auditoria Forense...")
    
    # 1. Carregar IDs solicitados
    if not os.path.exists(ARQUIVO_IDS_ENTRADA):
        print(f"❌ Erro: Arquivo {ARQUIVO_IDS_ENTRADA} não encontrado.")
        return
    
    df_procura = pd.read_csv(ARQUIVO_IDS_ENTRADA)
    # Converte para set para busca ultra-rápida (O(1))
    ids_alvo = set(df_procura.iloc[:, 0].astype(str).unique()) 
    print(f"🔎 Procurando por {len(ids_alvo)} IDs de pacientes nas bases originais...")

    # 2. Listar arquivos no Drive D
    arquivos = glob.glob(str(BASE_DIR / "ETLSIH.ST_MG_*_t.csv"))
    arquivos.sort(reverse=True) # Começa pelos mais recentes

    encontrados_list = []
    ids_restantes = ids_alvo.copy()

    # 3. Varredura nos arquivos
    for arquivo in arquivos:
        if not ids_restantes: break # Para se já achou todos
        
        nome_arq = os.path.basename(arquivo)
        print(f"Scanning: {nome_arq} ...", end="\r")
        
        try:
            # Leitura otimizada: apenas colunas necessárias e processamento em chunks se necessário
            df_temp = pd.read_csv(arquivo, usecols=COLUNAS_RELEVANTES, low_memory=False)
            df_temp['N_AIH'] = df_temp['N_AIH'].astype(str)
            
            # Filtra registros que estão na nossa lista alvo
            match = df_temp[df_temp['N_AIH'].isin(ids_restantes)].copy()
            
            if not match.empty:
                match['fonte_arquivo'] = nome_arq
                encontrados_list.append(match)
                # Remove os IDs já encontrados da lista de busca
                achados = set(match['N_AIH'].unique())
                ids_restantes -= achados
                print(f"✅ Achados {len(achados)} IDs em {nome_arq}. Restam {len(ids_restantes)}.")
                
        except Exception as e:
            print(f"x Erro ao ler {nome_arq}: {e}")

    # 4. Consolidação e Relatório
    if encontrados_list:
        df_final = pd.concat(encontrados_list, ignore_index=True)
        
        # Enriquecimento básico para o log
        df_final['DESFECHO'] = df_final['MORTE'].apply(lambda x: 'ÓBITO' if str(x) == '1' else 'ALTA/PERM')
        
        # Salvar CSV de Auditoria
        df_final.to_csv(ARQUIVO_SAIDA, index=False, encoding='utf-8-sig')
        
        print("\n\n" + "="*80)
        print("📊 SUMÁRIO DA AUDITORIA FORENSE")
        print("="*80)
        print(f"Total de IDs buscados: {len(ids_alvo)}")
        print(f"Total de IDs localizados: {len(df_final['N_AIH'].unique())}")
        print(f"Hospitalizações processadas: {len(df_final)}")
        print(f"Custo Total das AIHs auditadas: R$ {df_final['VAL_TOT'].sum():,.2f}")
        print(f"Óbitos detectados na amostra: {len(df_final[df_final['MORTE'] == 1])}")
        print("-" * 80)
        print(f"📁 Relatório completo gerado: {ARQUIVO_SAIDA}")
        
        # Exibe Top 5 casos mais caros ou mais longos no log
        print("\n🔝 TOP 5 CASOS POR PERMANÊNCIA (Saturação):")
        print(df_final.sort_values('DIAS_PERM', ascending=False)[['N_AIH', 'DIAG_PRINC', 'DIAS_PERM', 'DESFECHO']].head(5).to_string(index=False))
    else:
        print("\n\n❌ Nenhum dos IDs informados foi localizado nas bases do Drive D.")

if __name__ == "__main__":
    realizar_auditoria_forense()