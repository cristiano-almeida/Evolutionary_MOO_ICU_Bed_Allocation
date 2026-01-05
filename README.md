# Evolutionary Multi-objective Optimization for ICU Bed Allocation  
### *(CEC / WCCI 2026)*

Este repositório contém o **ecossistema completo de scripts, bases processadas e relatórios de auditoria** do projeto de **otimização multiobjetivo de leitos de UTI**, desenvolvido a partir de **dados reais do SIH/SUS**, acessados via **PCDaS / Fiocruz**.

O projeto foi concebido com **rigor científico, reprodutibilidade total e aderência a cenários reais de gestão hospitalar**, servindo tanto a fins acadêmicos quanto aplicados.

---

## 📂 Estrutura de Pastas

### 🏥 `/AUDITORIA_PACIENTES_FORA_HORIZONTE`
Análise forense dos pacientes cuja internação ultrapassou o horizonte de planejamento.

- **auditor_forense_pcdas.py**  
  Script para rastrear CIDs, custos e desfechos clínicos diretamente nas bases brutas (Drive D).

- **ids_para_auditoria.csv**  
  Lista de identificadores únicos (AIH) selecionados para investigação aprofundada.

- **RELATORIO_AUDITORIA_DETALHADA.csv**  
  Saída consolidada com custos, diagnósticos, tempo de permanência e complexidade clínica.

- **AUDITOR__FORENSE.txt**  
  Memória de cálculo e decisões técnicas da auditoria.

- **ANOTAÇÕES.odt**  
  Insights clínicos e interpretações qualitativas dos casos críticos.

---

### 📊 `/ESTIMADOR_LEITOS_PICO`
Dimensionamento da capacidade instalada real da rede hospitalar.

- **estimador_capacidade_real.py**  
  Cálculo da ocupação simultânea máxima histórica por unidade hospitalar.

- **CENSO_LEITOS_ESTIMADO_JF.csv**  
  Relatório consolidado de leitos estimados por CNPJ.

- **ESTIMADOR_LEITOS.txt**  
  Veredito técnico sobre a capacidade de pico necessária para a rede SUS/JF.

---

### 🖼️ `/FIGURAS`
Assets visuais utilizados no artigo científico.

- **/CEC/**  
  Figuras (1 a 8) formatadas especificamente para o template **IEEE / CEC**.  
  Inclui logs e CSVs de suporte para **reprodutibilidade total**.

- **fig1 a fig7**  
  Gráficos de convergência, Gantt, PCP, ponto de inflexão, equidade, radar multiobjetivo e dívida biológica.

- **csv_massive_186.csv / csv_overload_***  
  Bases específicas que originaram os gráficos do artigo.

---

### 📅 `/GERAR_BASE_MENSAL`
Consolidação e limpeza de faturamento hospitalar massivo.

- **analise_bases.py**
- **gerar_experimento.py**
- **BASE_MASTER_CONSULTA.csv**
- **base_final_30dias.csv**

---

### 🛠️ `/GERAR_BASES_12_Leitos`
Cenários controlados para validação inicial dos algoritmos.

- **gerar_bases_reais.py**
- **base_real_***

---

### 🔍 `/MINERADOR_BASES`
Caracterização epidemiológica da base PCDaS.

- **profiler_estatistico_uti.py**
- **SINTESE_ESTATISTICA_JF.csv**
- **ESTIMATIVA_LEITOS_POR_HOSPITAL.csv**

---

## 🧪 Testes de Escala Massiva (Cenários de 30 Dias)

- **/TESTE_MENSAL_1171-186**
- **/TESTE_MENSAL_1171-233**
- **/TESTE_1_MENSAL_1171-278**
- **/TESTE_2_MENSAL_1027-265**

---

### ⚖️ Testes Comparativos

- **/TESTE_BASE_CENTRAL**
- **/TESTE_BASE_OVERLOAD**

---

## 📂 Arquivos na Raiz

- **UTI_NSGA-II.py**
- **UTI_GDE3.py**
- **2026_cec_hospital_optimization.pdf**
- **Guia Técnico.pdf**

---

## 🛠️ Como Reproduzir os Experimentos

1. Execute os scripts em `/GERAR_BASE_MENSAL`
2. Rode `UTI_NSGA-II.py` ou `UTI_GDE3.py`
3. Consulte `AUDITORIA_V11_DETALHADA.txt`

---

## 📌 Contexto Científico

Projeto desenvolvido para o **CEC / WCCI 2026**, com dados reais do **SIH/SUS via PCDaS / Fiocruz**, focado em otimização multiobjetivo, equidade, risco e reprodutibilidade.
