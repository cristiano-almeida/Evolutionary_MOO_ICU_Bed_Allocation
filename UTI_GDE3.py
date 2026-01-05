import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Any

# Importações do Pymoo
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.pcp import PCP

try:
    from pymoode.algorithms import GDE3
    PYMODE_AVAILABLE = True
    print("✓ pymoode.algorithms importado com sucesso")
except ImportError as e:
    print(f"✗ Erro na importação alternativa: {e}")
    PYMODE_AVAILABLE = False

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO DE LOGGING
# ---------------------------------------------------------------------------

def configurar_logging(nivel_logging: str = 'INFO') -> logging.Logger:
    """Configura e retorna logger para monitoramento detalhado."""
    logger = logging.getLogger('OtimizacaoUTI_GDE3')
    logger.setLevel(getattr(logging, nivel_logging))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# ---------------------------------------------------------------------------
# 1. GERAÇÃO E VALIDAÇÃO DE DADOS
# ---------------------------------------------------------------------------

def gerar_dados_simulacao_realista(numero_pacientes: int, horizonte_tempo: int, logger: logging.Logger) -> List[Dict]:
    """
    Gera dados de pacientes realistas para simulação de UTI com distribuições
    baseadas em padrões clínicos reais.
    """
    np.random.seed(42)
    
    pacientes = []
    for i in range(numero_pacientes):
        # Tempo de chegada: distribuído exponencialmente (mais chegadas no início)
        tempo_chegada = int(np.random.exponential(scale=24))
        tempo_chegada = min(tempo_chegada, horizonte_tempo // 2)
        
        # Gravidade: distribuição realista (mais pacientes moderados)
        gravidade = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                                   p=[0.05, 0.10, 0.15, 0.15, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04])
        
        # Tempo estimado UTI baseado na gravidade (mais realista)
        if gravidade >= 8:  # Pacientes críticos
            tempo_uti = np.random.randint(72, 120)   # 3-5 dias 
        elif gravidade >= 5:  # Pacientes moderados
            tempo_uti = np.random.randint(48, 96)   # 2-4 dias
        else:  # Pacientes leves
            tempo_uti = np.random.randint(24, 72)   # 1-3 dias
        
        pacientes.append({
            "id_paciente": i + 1,
            "tempo_chegada_hora": tempo_chegada,
            "gravidade_score": gravidade,
            "tempo_estimado_uti_horas": tempo_uti
        })
    
    logger.info(f"Dados gerados: {len(pacientes)} pacientes com distribuição realista")
    return pacientes

def carregar_ou_gerar_dados(nome_arquivo: str, numero_pacientes: int, horizonte_tempo: int, logger: logging.Logger) -> List[Dict]:
    """Carrega dados existentes ou gera nova base."""
    if os.path.exists(nome_arquivo):
        logger.info(f"Carregando base de dados existente: '{nome_arquivo}'")
        df_pacientes = pd.read_csv(nome_arquivo)
        pacientes = df_pacientes.to_dict('records')
    else:
        logger.info("Gerando nova base de dados realista...")
        pacientes = gerar_dados_simulacao_realista(numero_pacientes, horizonte_tempo, logger)
        df_pacientes = pd.DataFrame(pacientes)
        df_pacientes.to_csv(nome_arquivo, index=False, encoding='utf-8')
        logger.info(f"Base de dados salva em: '{nome_arquivo}'")
    
    return pacientes

def ajustar_tempos_uti_para_limites_realistas(pacientes: List[Dict], limite_maximo_horas: int = 120, logger: logging.Logger = None) -> List[Dict]:
    """Ajusta tempos de UTI para limites clinicamente realistas."""
    pacientes_ajustados = []
    alteracoes_realizadas = 0
    
    for paciente in pacientes:
        # CORREÇÃO CRUCIAL: Casting para float e depois int para lidar com dados do PCDaS (ex: '48.0')
        t_original_raw = float(paciente['tempo_estimado_uti_horas'])
        tempo_ajustado = int(min(t_original_raw, limite_maximo_horas))
        
        if t_original_raw != tempo_ajustado:
            alteracoes_realizadas += 1
            if logger:
                logger.info(f"Paciente {paciente['id_paciente']}: Tempo UTI ajustado de {t_original_raw} para {tempo_ajustado} horas")
        
        pacientes_ajustados.append({
            'id_paciente': paciente['id_paciente'],
            # CORREÇÃO: Garantir que chegada e gravidade também sejam inteiros puros
            'tempo_chegada_hora': int(float(paciente['tempo_chegada_hora'])),
            'gravidade_score': int(float(paciente['gravidade_score'])),
            'tempo_estimado_uti_horas': tempo_ajustado
        })
    
    if logger and alteracoes_realizadas > 0:
        logger.info(f"Tempos de UTI ajustados: {alteracoes_realizadas} pacientes modificados")
        logger.info(f"Limite máximo estabelecido: {limite_maximo_horas} horas ({limite_maximo_horas//24} dias)")
    
    return pacientes_ajustados

def verificar_viabilidade_sistema(pacientes: List[Dict], numero_leitos: int, horizonte_tempo: int, logger: logging.Logger) -> Dict[str, Any]:
    """Verifica a viabilidade do sistema e fornece estatísticas detalhadas."""
    # CORREÇÃO: Garantir que parâmetros de cálculo sejam inteiros
    capacidade_total = int(numero_leitos) * int(horizonte_tempo)
    tempo_total_requerido = sum(int(paciente['tempo_estimado_uti_horas']) for paciente in pacientes)
    taxa_ocupacao_teorica = (tempo_total_requerido / capacidade_total) * 100 if capacidade_total > 0 else 0
    
    # Estatísticas adicionais para análise
    tempos_chegada = [p['tempo_chegada_hora'] for p in pacientes]
    gravidades = [p['gravidade_score'] for p in pacientes]
    tempos_uti = [p['tempo_estimado_uti_horas'] for p in pacientes]
    
    analise = {
        'numero_leitos': numero_leitos,
        'horizonte_tempo': horizonte_tempo,
        'capacidade_total': capacidade_total,
        'tempo_total_requerido': tempo_total_requerido,
        'taxa_ocupacao_teorica': taxa_ocupacao_teorica,
        'numero_pacientes': len(pacientes),
        'tempo_chegada_min': min(tempos_chegada) if tempos_chegada else 0,
        'tempo_chegada_max': max(tempos_chegada) if tempos_chegada else 0,
        'gravidade_media': np.mean(gravidades) if gravidades else 0,
        'gravidade_max': max(gravidades) if gravidades else 0,
        'tempo_uti_medio': np.mean(tempos_uti) if tempos_uti else 0,
        'tempo_uti_max': max(tempos_uti) if tempos_uti else 0,
        'viavel': taxa_ocupacao_teorica <= 150 # Flexibilizado para permitir cenário Overload real
    }
    
    logger.info("=" * 80)
    logger.info("ANÁLISE DE VIABILIDADE DO CENÁRIO")
    logger.info("=" * 80)
    logger.info(f"Número de leitos: {numero_leitos}")
    logger.info(f"Horizonte de planejamento: {horizonte_tempo} horas ({horizonte_tempo//24} dias)")
    logger.info(f"Capacidade total do sistema: {capacidade_total} horas-leito")
    logger.info(f"Demanda total dos pacientes: {tempo_total_requerido} horas-leito")
    logger.info(f"Taxa de ocupação teórica: {taxa_ocupacao_teorica:.1f}%")
    logger.info(f"Número de pacientes: {len(pacientes)}")
    logger.info(f"Tempo de UTI médio: {analise['tempo_uti_medio']:.1f} horas")
    logger.info(f"Gravidade média: {analise['gravidade_media']:.2f}")
    
    if taxa_ocupacao_teorica > 95:
        logger.warning("AVISO: Sistema com alta probabilidade de ser super-restringido.")
    elif taxa_ocupacao_teorica > 85:
        logger.info("Sistema com boa capacidade mas próximo do limite.")
    else:
        logger.info("Sistema com capacidade adequada.")
    
    return analise

def calcular_valores_referencia_realistas(pacientes: List[Dict], horizonte_tempo: int, logger: logging.Logger) -> Tuple[float, float, float]:
    """Calcula valores de referência realistas para normalização."""
    tempo_espera_maximo = 0
    risco_clinico_maximo = 0
    custo_terminal_maximo = 0 
    
    for paciente in pacientes:
        # CORREÇÃO: Garantir cálculos com inteiros puros
        chegada = int(paciente['tempo_chegada_hora'])
        uti = int(paciente['tempo_estimado_uti_horas'])
        grav = int(paciente['gravidade_score'])
        
        # Tempo máximo de espera realista: mínimo entre horizonte restante e tempo de UTI
        tempo_espera_max_paciente = min(int(horizonte_tempo) - chegada, uti)
        tempo_espera_maximo += tempo_espera_max_paciente
        
        # Risco clínico máximo realista
        risco_clinico_maximo += tempo_espera_max_paciente * grav
        
        # Custo terminal máximo realista
        if chegada + uti > int(horizonte_tempo):
            excesso_maximo = (chegada + uti - int(horizonte_tempo))
            custo_terminal_maximo += excesso_maximo * grav
    
    # CORREÇÃO: Piso de segurança de 500 para evitar que o custo terminal seja 1 (log original)
    tempo_espera_maximo = max(float(tempo_espera_maximo), 100.0)
    risco_clinico_maximo = max(float(risco_clinico_maximo), 500.0)
    custo_terminal_maximo = max(float(custo_terminal_maximo), 500.0)
    
    logger.info("VALORES DE REFERÊNCIA REALISTAS PARA NORMALIZAÇÃO:")
    logger.info(f"Tempo de espera máximo de referência: {tempo_espera_maximo:.0f} horas")
    logger.info(f"Risco clínico máximo de referência: {risco_clinico_maximo:.0f}")
    logger.info(f"Custo terminal máximo de referência: {custo_terminal_maximo:.0f}")
    
    return tempo_espera_maximo, risco_clinico_maximo, custo_terminal_maximo

def calcular_ocupacao_precisa(tempo_inicio_uti: int, tempo_fim_uti: int, horizonte_tempo: int) -> np.ndarray:
    """Calcula a ocupação de leitos com precisão."""
    # CORREÇÃO DEFINITIVA DO ERRO DE SLICE:
    ocupacao = np.zeros(int(horizonte_tempo), dtype=int)
    inicio = int(max(0, tempo_inicio_uti))
    fim = int(min(int(horizonte_tempo), tempo_fim_uti))
    
    if fim > inicio:
        # Fatiamento exige inteiros (__index__)
        ocupacao[inicio:fim] += 1
    
    return ocupacao

# ---------------------------------------------------------------------------
# 2. DEFINIÇÃO DO PROBLEMA DE OTIMIZAÇÃO (COMPATÍVEL COM GDE3)
# ---------------------------------------------------------------------------

class ProblemaOtimizacaoUTI(ElementwiseProblem):
    def __init__(self, pacientes: List[Dict], numero_leitos: int, horizonte_tempo: int, logger: logging.Logger):
        self.pacientes = pacientes
        self.numero_leitos = int(numero_leitos)
        self.horizonte_tempo = int(horizonte_tempo)
        self.numero_pacientes = len(pacientes)
        self.logger = logger
        
        self.tempo_espera_max_ref, self.risco_clinico_max_ref, self.custo_terminal_max_ref = \
            calcular_valores_referencia_realistas(pacientes, horizonte_tempo, logger)
        
        # Limites inferiores: tempo de chegada de cada paciente (int)
        limites_inferiores = [int(p['tempo_chegada_hora']) for p in pacientes]
        
        # Limites superiores: horizonte de tempo (int)
        limites_superiores = [self.horizonte_tempo for _ in pacientes]
        
        super().__init__(
            n_var=self.numero_pacientes, 
            n_obj=4,
            n_constr=2,  # Restrições rígidas (capacidade e precedência)
            xl=np.array(limites_inferiores), 
            xu=np.array(limites_superiores),
            vtype=int 
        )
        
        self.logger.info(f"Problema de otimização inicializado com {self.numero_pacientes} variáveis")

    def _evaluate(self, x, out, *args, **kwargs):
        # x vem do pymoo como float, convertemos para array de inteiros
        cronograma = np.round(x).astype(int)
        
        # Inicializa métricas e vetores
        tempo_espera_total = 0
        risco_clinico_total = 0
        custo_ocupacao_terminal = 0
        uso_leitos_por_hora = np.zeros(self.horizonte_tempo, dtype=int)
        
        for indice, paciente in enumerate(self.pacientes):
            tempo_inicio_uti = cronograma[indice]
            chegada = int(paciente['tempo_chegada_hora'])
            duracao = int(paciente['tempo_estimado_uti_horas'])
            grav = int(paciente['gravidade_score'])
            
            # Calcular tempo de espera
            tempo_espera = max(0, tempo_inicio_uti - chegada)
            tempo_espera_total += tempo_espera
            risco_clinico_total += tempo_espera * grav
            
            # Calcular tempo de fim
            tempo_fim_uti = tempo_inicio_uti + duracao
            
            # Calcular custo terminal (se ultrapassar o horizonte)
            if tempo_fim_uti > self.horizonte_tempo:
                excesso_tempo = tempo_fim_uti - self.horizonte_tempo
                custo_ocupacao_terminal += excesso_tempo * grav
            
            # Calcular ocupação horária (Usa a função com fix de slice)
            uso_leitos_por_hora += calcular_ocupacao_precisa(tempo_inicio_uti, tempo_fim_uti, self.horizonte_tempo)
        
        # --- CÁLCULO DAS RESTRIÇÕES RÍGIDAS ---
        maxima_ocupacao = np.max(uso_leitos_por_hora) if self.horizonte_tempo > 0 else 0
        violacao_capacidade = max(0, maxima_ocupacao - self.numero_leitos)
        violacao_precedencia = sum(max(0, p['tempo_chegada_hora'] - cronograma[i]) for i, p in enumerate(self.pacientes))

        # Passa as violações como float para o otimizador
        out["G"] = [float(violacao_capacidade), float(violacao_precedencia)]
        
        # --- CÁLCULO DOS OBJETIVOS (NORMALIZADOS) ---
        f1 = tempo_espera_total / self.tempo_espera_max_ref
        
        utilizacao_media = np.mean(uso_leitos_por_hora) if self.horizonte_tempo > 0 else 0
        taxa_utilizacao = utilizacao_media / self.numero_leitos if self.numero_leitos > 0 else 0
        f2 = 1.0 - taxa_utilizacao
        
        f3 = risco_clinico_total / self.risco_clinico_max_ref
        f4 = custo_ocupacao_terminal / self.custo_terminal_max_ref
        
        out["F"] = [f1, f2, f3, f4]

# ---------------------------------------------------------------------------
# 3. FUNÇÕES PARA ANÁLISE E RELATÓRIOS
# ---------------------------------------------------------------------------

def gerar_relatorio_detalhado_plano(pacientes: List[Dict], cronograma: np.ndarray, nome_arquivo_saida: str, 
                                  numero_leitos: int, horizonte_tempo: int, logger: logging.Logger) -> Dict[str, Any]:
    """Gera relatório completo do plano de alocação."""
    cronograma_inteiro = np.round(cronograma).astype(int)

    dados_relatorio = []
    metricas = {
        'tempo_espera_total': 0, 'risco_clinico_total': 0, 'custo_terminal_total': 0,
        'pacientes_fora_horizonte': 0, 'pacientes_com_espera': 0, 'pacientes_imediatos': 0,
        'utilizacao_media_leitos': 0, 'maxima_ocupacao_simultanea': 0, 'violacao_capacidade': 0,
        'horas_ociosas_total': 0, 'eficiencia_utilizacao': 0
    }
    
    ocupacao_horaria = np.zeros(int(horizonte_tempo), dtype=int)
    
    for indice, paciente in enumerate(pacientes):
        hora_internacao = int(cronograma_inteiro[indice])
        tempo_espera = max(0, hora_internacao - paciente['tempo_chegada_hora'])
        hora_alta = hora_internacao + paciente['tempo_estimado_uti_horas']
        custo_terminal = max(0, hora_alta - int(horizonte_tempo)) * paciente['gravidade_score']
        
        if hora_alta > int(horizonte_tempo):
            metricas['pacientes_fora_horizonte'] += 1
        
        if tempo_espera > 0:
            metricas['pacientes_com_espera'] += 1
        else:
            metricas['pacientes_imediatos'] += 1

        ocupacao_paciente = calcular_ocupacao_precisa(hora_internacao, hora_alta, horizonte_tempo)
        ocupacao_horaria += ocupacao_paciente
        
        dados_relatorio.append({
            "ID_Paciente": paciente['id_paciente'],
            "Hora_Chegada": paciente['tempo_chegada_hora'],
            "Gravidade_Score": paciente['gravidade_score'],
            "Tempo_UTI_Estimado": paciente['tempo_estimado_uti_horas'],
            "Hora_Internacao_Otimizada": hora_internacao,
            "Tempo_Espera_Horas": tempo_espera,
            "Hora_Estimada_Alta": hora_alta,
            "Custo_Terminal": custo_terminal,
            "Status": "Fora do Horizonte" if hora_alta > horizonte_tempo else "Dentro do Horizonte",
            "Internacao_Imediata": "Sim" if tempo_espera == 0 else "Não"
        })
        
        metricas['tempo_espera_total'] += tempo_espera
        metricas['risco_clinico_total'] += tempo_espera * paciente['gravidade_score']
        metricas['custo_terminal_total'] += custo_terminal

    metricas['utilizacao_media_leitos'] = np.mean(ocupacao_horaria)
    metricas['maxima_ocupacao_simultanea'] = np.max(ocupacao_horaria)
    metricas['violacao_capacidade'] = max(0, metricas['maxima_ocupacao_simultanea'] - numero_leitos)
    metricas['horas_ociosas_total'] = sum(max(0, numero_leitos - ocupacao) for ocupacao in ocupacao_horaria)
    metricas['eficiencia_utilizacao'] = (metricas['utilizacao_media_leitos'] / numero_leitos) * 100
    
    df_relatorio = pd.DataFrame(dados_relatorio)
    df_relatorio.to_csv(nome_arquivo_saida, index=False, encoding='utf-8')
    
    relatorio_metricas = f"""
RELATÓRIO DE MÉTRICAS GDE3 - DADOS REAIS SUS
==================================================
Arquivo: {nome_arquivo_saida}
Data: {pd.Timestamp.now()}

MÉTRICAS AGREGADAS:
-------------------
Tempo Total de Espera: {metricas['tempo_espera_total']:.1f} horas
Risco Clínico Total: {metricas['risco_clinico_total']:.1f}
Custo Terminal Total: {metricas['custo_terminal_total']:.1f}

UTILIZAÇÃO:
-----------
Eficiência: {metricas['eficiencia_utilizacao']:.1f}%
Ocupação Máxima: {int(metricas['maxima_ocupacao_simultanea'])} leitos
    """
    
    with open(nome_arquivo_saida.replace('.csv', '_metricas.txt'), 'w', encoding='utf-8') as arquivo:
        arquivo.write(relatorio_metricas)
    
    logger.info(f"Relatórios detalhados salvos (CSV e TXT).")
    return metricas

def analisar_resultados_otimizacao(resultado, problema, pacientes, numero_leitos, horizonte_tempo, logger, nome_algoritmo="GDE3"):
    """Analisa os resultados da otimização detalhadamente."""
    if resultado.F is not None and len(resultado.F) > 0:
        num_solucoes = len(resultado.F)
        logger.info(f"Número de soluções na fronteira de Pareto: {num_solucoes}")
        
        fronteira_pareto = resultado.F
        tempo_espera_real = fronteira_pareto[:, 0] * problema.tempo_espera_max_ref
        utilizacao_real = (1 - fronteira_pareto[:, 1]) * 100
        risco_clinico_real = fronteira_pareto[:, 2] * problema.risco_clinico_max_ref
        custo_terminal_real = fronteira_pareto[:, 3] * problema.custo_terminal_max_ref

        indices_otimos = {
            'menor_tempo_espera': np.argmin(tempo_espera_real),
            'maior_utilizacao': np.argmax(utilizacao_real),
            'menor_risco_clinico': np.argmin(risco_clinico_real),
            'menor_custo_terminal': np.argmin(custo_terminal_real),
            'solucao_balanceada': np.argmin(np.sum(fronteira_pareto, axis=1))
        }
        
        logger.info(f"SOLUÇÕES DE DESTAQUE ENCONTRADAS - {nome_algoritmo.upper()}:")
        for criterio, indice in indices_otimos.items():
            logger.info(f"{criterio.upper().replace('_', ' '):<25}: "
                      f"Espera={tempo_espera_real[indice]:.0f}h, "
                      f"Utilização={utilizacao_real[indice]:.1f}%, "
                      f"Risco={risco_clinico_real[indice]:.0f}, "
                      f"Custo={custo_terminal_real[indice]:.0f}")
        
        for criterio, indice in indices_otimos.items():
            gerar_relatorio_detalhado_plano(pacientes, resultado.X[indice], f"relatorio_{nome_algoritmo.lower()}_{criterio}.csv", numero_leitos, horizonte_tempo, logger)
        
        return True, {}
    
    return False, {}

def visualizar_fronteira_pareto(resultado, problema, logger):
    """Gera visualização mais intuitiva da fronteira de Pareto."""
    try:
        if resultado.F is not None and len(resultado.F) > 0:
            fig, ax = plt.subplots(figsize=(14, 8))
            for i in range(len(resultado.F)):
                ax.plot(range(4), resultado.F[i], color='lightgray', alpha=0.5)
            
            idx_b = np.argmin(np.sum(resultado.F, axis=1))
            ax.plot(range(4), resultado.F[idx_b], color='red', linewidth=3, label='Solução Balanceada')
            
            ax.set_xticks(range(4))
            ax.set_xticklabels(['Espera', 'Ociosidade', 'Risco', 'CustoT'])
            ax.set_title("Fronteira de Pareto GDE3 (Normalizada)")
            plt.legend()
            plt.savefig(f"fronteira_gde3_real_{datetime.now().strftime('%H%M%S')}.png")
            plt.close()
            logger.info(f"Visualização de Pareto gerada.")
    except Exception as e:
        logger.error(f"Erro visualização: {e}")

# ---------------------------------------------------------------------------
# 4. CONFIGURAÇÃO DO ALGORITMO GDE3
# ---------------------------------------------------------------------------

def configurar_algoritmo_gde3(config_gde3: Dict[str, Any]):
    return GDE3(
        pop_size=config_gde3["tamanho_populacao"],
        sampling=IntegerRandomSampling(),
        variant=config_gde3["variante"],
        CR=config_gde3["CR"],
        F=config_gde3["F"]
    )
    
# ---------------------------------------------------------------------------
# 5. SIMULAÇÃO PRINCIPAL COM GDE3
# ---------------------------------------------------------------------------

def executar_simulacao_gde3(config_sim: Dict[str, Any], config_gde3: Dict[str, Any]):
    """Executa a simulação completa com algoritmo GDE3."""
    if not PYMODE_AVAILABLE:
        print("ERRO: pymoode não está disponível.")
        return None, False, {}

    logger = configurar_logging('INFO')

    try:
        logger.info("=" * 100)
        logger.info("SISTEMA DE OTIMIZAÇÃO DE ALOCAÇÃO DE LEITOS DE UTI - ALGORITMO GDE3")
        logger.info(f"Variante: {config_gde3['variante']}")
        logger.info("=" * 100)

        pacientes_brutos = carregar_ou_gerar_dados(config_sim['nome_arquivo_base'], config_sim['numero_pacientes'], config_sim['horizonte_planejamento_horas'], logger)
        
        # Casting garantido para inteiros antes de qualquer operação
        pacientes = ajustar_tempos_uti_para_limites_realistas(pacientes_brutos, config_sim['limite_maximo_uti'], logger)
        
        verificar_viabilidade_sistema(pacientes, config_sim['numero_leitos_uti'], config_sim['horizonte_planejamento_horas'], logger)

        problema = ProblemaOtimizacaoUTI(pacientes, config_sim['numero_leitos_uti'], config_sim['horizonte_planejamento_horas'], logger)
        algoritmo = configurar_algoritmo_gde3(config_gde3)

        logger.info(f"Executando {config_gde3['numero_geracoes']} gerações com população {config_gde3['tamanho_populacao']}...")
        
        resultado = minimize(
            problema, algoritmo, ('n_gen', config_gde3['numero_geracoes']),
            verbose=True, seed=42, save_history=True
        )

        analisar_resultados_otimizacao(resultado, problema, pacientes, config_sim['numero_leitos_uti'], config_sim['horizonte_planejamento_horas'], logger, "GDE3")
        visualizar_fronteira_pareto(resultado, problema, logger)

        return resultado, True, {}

    except Exception as e:
        logger.error(f"Erro durante a simulação: {str(e)}")
        import traceback; traceback.print_exc() # Detalhamento do erro
        return None, False, {}

# ---------------------------------------------------------------------------
# 6. EXECUÇÃO PRINCIPAL
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Verificar disponibilidade do pymoode
    print("=" * 80)
    print("VERIFICANDO DISPONIBILIDADE DO PYMODE...")
    print("=" * 80)
    
    if PYMODE_AVAILABLE:
        print("✓ pymoode está disponível!")
    else:
        print("✗ pymoode não está disponível.")
        exit(1)
    
    # ========================================================================
    # == PAINEL DE CONTROLE: Ajustado para o Cenário Real PCDaS - Juiz de Fora ==
    # ========================================================================
    
    CONFIGURACAO_SIMULACAO = {
        "numero_leitos_uti": 12,
        "horizonte_planejamento_horas": 1200, # Aumentado para cobrir o mês real
        "numero_pacientes": 62,              # Ajuste para 31, 62 ou 78 dependendo da base
        "limite_maximo_uti": 800,             # Teto para aceitar pacientes reais de longa permanência
        "nome_arquivo_base": "base_real_central",
    }
    
    CONFIGURACAO_GDE3 = {
        "numero_geracoes": 250,
        "tamanho_populacao": 350,
        "variante": "DE/rand-tobest/1/bin", 
        "CR": 0.8,                    
        "F": (0.5, 0.8),              
    }

    # ========================================================================

    logger = configurar_logging('INFO')
    
    logger.info("=" * 100)
    logger.info("INICIANDO ESTUDO COMPARATIVO: NSGA-II vs GDE3 (DADOS REAIS)")
    logger.info("=" * 100)
    
    resultado_gde3, sucesso_gde3, _ = executar_simulacao_gde3(CONFIGURACAO_SIMULACAO, CONFIGURACAO_GDE3)
    
    if sucesso_gde3:
        print("\n" + "="*80)
        print("PROCESSO CONCLUÍDO COM SUCESSO!")
        print("Relatórios detalhados do GDE3 gerados para análise comparativa.")
        print("="*80)