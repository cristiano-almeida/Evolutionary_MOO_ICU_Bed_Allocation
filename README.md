# Evolutionary Multi-objective Optimization for ICU Bed Allocation  
### *(CEC 2026)*

Este repositório contém o **ecossistema completo de scripts, dados processados e relatórios de auditoria** do projeto de **otimização de leitos de UTI**, baseado em **dados reais do SIH/SUS**, obtidos via **PCDaS / Fiocruz**.

---

## 📂 Descrição da Estrutura de Pastas

### 🏥 `/AUDITORIA_PACIENTES_FORA_HORIZONTE`
Análise forense dos casos que excederam a capacidade de planejamento.

- **auditor_forense_pcdas.py**  
  Script para rastrear CIDs e desfechos clínicos nas bases brutas do Drive D.

- **ids_para_auditoria.csv**  
  Lista de identificadores únicos (AIH) selecionados para investigação.

- **RELATORIO_AUDITORIA_DETALHADA.csv**  
  Saída contendo custos, diagnósticos e complexidade dos pacientes em débito.

- **AUDITOR_FORENSE.txt** / **ANOTAÇÕES.odt**  
  Memória de cálculo e insights clínicos da auditoria.

---

### 📊 `/ESTIMADOR_LEITOS_PICO`
Dimensionamento da capacidade instalada por unidade hospitalar.

- **estimador_capacidade_real.py**  
  Script que calcula a ocupação simultânea máxima histórica.

- **CENSO_LEITOS_ESTIMADO_JF.csv**  
  Relatório consolidado de leitos estimados por CNPJ.

- **ESTIMADOR_LEITOS.txt**  
  Veredito sobre a capacidade de pico necessária.

---

### 🖼️ `/FIGURAS`
Assets visuais para o artigo científico.

- **/CEC/**  
  Figuras (1 a 8) formatadas para o template **IEEE / CEC**, incluindo logs e CSVs de suporte para reprodutibilidade.

- **fig1 a fig7**  
  Gráficos de Convergência, PCP, Inflexão, Equidade, Radar, Dívida Biológica e Estatística.

- **csv_massive_186.csv / csv_overload_***  
  Bases de dados que deram origem aos gráficos.

---

### 📅 `/GERAR_BASE_MENSAL`
Consolidação e limpeza de faturamento massivo.

- **analise_bases.py**  
  De-duplicação e tratamento de registros administrativos.

- **gerar_experimento.py**  
  Recorte de janelas específicas de 30 dias.

- **BASE_MASTER_CONSULTA.csv**  
  Banco de dados limpo com **97.309 internações únicas**.

- **base_final_30dias.csv**  
  Amostra selecionada para o experimento de escala real.

---

### 🛠️ `/GERAR_BASES_12_Leitos`
Cenários controlados para validação inicial de algoritmos.

- **gerar_bases_reais.py**  
  Script gerador dos cenários *Underload*, *Central* e *Overload*.

- **base_real_***  
  Arquivos de entrada com **31, 62 e 78 pacientes**.

---

### 🔍 `/MINERADOR_BASES`
Caracterização epidemiológica da base PCDaS.

- **profiler_estatistico_uti.py**  
  Extração de médias, desvios e Top CIDs.

- **SINTESE_ESTATISTICA_JF.csv**  
  Resumo estatístico da rede SUS/JF (2023–2025).

- **ESTIMATIVA_LEITOS_POR_HOSPITAL.csv**  
  Cruzamento entre volume de internações e permanência média.

---

## 🧪 Testes de Escala Massiva (Cenários de 30 Dias)

Resultados completos (logs, CSVs de Pareto e dossiês de auditoria) para diferentes configurações de leitos (L):

- **/TESTE_MENSAL_1171-186** — Crise real (74,5% de ocupação)
- **/TESTE_MENSAL_1171-233** — Eficiência ótima (*Inflexão*)
- **/TESTE_1_MENSAL_1171-278** — Redundância física (*Espera Zero*)
- **/TESTE_2_MENSAL_1027-265** — Validação sequencial (Abril/2023)

---

### ⚖️ `/TESTE_BASE_CENTRAL` & `/TESTE_BASE_OVERLOAD`
Baterias comparativas entre **NSGA-II** e **GDE3**.

Cada subpasta (**BALANCEADO / EXPLORATÓRIO / EXPLOTATIVO**) contém:
- Log de execução
- PDF de gráficos consolidados
- Cinco relatórios estratégicos (Maior Utilização, Menor Risco, etc.)

---

## 📂 Arquivos na Raiz

- **UTI_NSGA-II.py**  
  Motor principal de otimização via Algoritmo Genético Multiobjetivo.

- **UTI_GDE3.py**  
  Motor comparativo via Evolução Diferencial Multiobjetivo.

- **2026_cec_hospital_optimization.pdf**  
  Artigo científico completo submetido ao CEC 2026.

- **Guia Técnico-científico.pdf**  
  Documentação detalhada da metodologia multiobjetivo.

---

## 🛠️ Como Reproduzir os Experimentos

1. Acesse `/GERAR_BASE_MENSAL` para preparar os dados brutos.  
2. Execute `UTI_NSGA-II.py` (ou `UTI_GDE3.py`) na raiz, selecionando:
   - bases de `/GERAR_BASES_12_Leitos`, ou
   - cenários massivos de 30 dias.
3. Para auditoria detalhada, consulte o arquivo:
   ```
   AUDITORIA_V11_DETALHADA.txt
   ```
   gerado automaticamente em cada pasta de teste.

---

## 📌 Contexto Científico

Projeto desenvolvido como parte dos requisitos do **CEC / WCCI 2026**, utilizando **dados reais do SIH/SUS** via **PCDaS / Fiocruz**, com foco em otimização multiobjetivo, risco, equidade e reprodutibilidade científica.
