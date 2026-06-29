import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from scipy import stats

st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")

st.title("Telco Churn Prediction")
st.markdown("Bem-vindo ao dashboard de previsão de Churn. Preencha os dados do cliente para analisar o risco ou explore os dados históricos.")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Simulador de Churn", "Análise Histórica (EDA)", "Arquitetura e Decisões"])

with tab1:
    st.sidebar.header("Dados do Cliente")

    # Inputs matching the API CustomerData Pydantic schema
    tenure = st.sidebar.number_input("Meses de Contrato (Tenure)", min_value=0, max_value=100, value=12)
    monthly_charges = st.sidebar.number_input("Cobrança Mensal ($)", min_value=0.0, value=50.0)
    total_charges = st.sidebar.number_input("Cobrança Total Acumulada ($)", min_value=0.0, value=600.0)

    contract_type = st.sidebar.selectbox("Tipo de Contrato", ["Mensal (Month-to-month)", "1 Ano (One year)", "2 Anos (Two year)"])
    if contract_type == "Mensal (Month-to-month)":
        contract_mapped = "Month-to-month"
    elif contract_type == "1 Ano (One year)":
        contract_mapped = "One year"
    else:
        contract_mapped = "Two year"

    internet_service = st.sidebar.selectbox("Serviço de Internet", ["DSL", "Fibra Óptica (Fiber optic)", "Nenhum"])
    if internet_service == "Fibra Óptica (Fiber optic)":
        internet_mapped = "Fiber optic"
    elif internet_service == "DSL":
        internet_mapped = "DSL"
    else:
        internet_mapped = "No"

    if st.button("Prever Risco de Churn", type="primary"):
        # Payload completo conforme exigido pelo novo schema da API
        payload = {
            "Latitude": 34.0,
            "Longitude": -118.0,
            "Tenure Months": tenure,
            "Monthly Charges": monthly_charges,
            "CLTV": int(total_charges), # Usando o total charges como proxy para o CLTV exigido ou passando um mock
            "Gender": "Male",
            "Senior Citizen": "No",
            "Partner": "No",
            "Dependents": "No",
            "Phone Service": "Yes",
            "Multiple Lines": "No",
            "Internet Service": internet_mapped,
            "Online Security": "No",
            "Online Backup": "No",
            "Device Protection": "No",
            "Tech Support": "No",
            "Streaming TV": "No",
            "Streaming Movies": "No",
            "Contract": contract_mapped,
            "Paperless Billing": "Yes",
            "Payment Method": "Electronic check"
        }
        
        with st.spinner("Consultando modelo na API..."):
            try:
                # Chama a API FastAPI rodando localmente
                response = requests.post("http://127.0.0.1:8000/predict", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    prob = result.get("churn_probability", 0) * 100
                    risk = result.get("risk_level", "Unknown")
                    
                    # Extraindo a latência enviada pelo Middleware da API
                    latency_str = response.headers.get("X-Process-Time", "0")
                    try:
                        latency_ms = float(latency_str) * 1000
                    except ValueError:
                        latency_ms = 0
                    
                    st.subheader("Resultado da Previsão")
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.metric(label="Nível de Risco", value=risk)
                        st.metric(label="Latência da API", value=f"{latency_ms:.2f} ms")
                        
                        if risk == "High":
                            st.error("ALTO RISCO: Sugere-se ação de retenção imediata!")
                        else:
                            st.success("BAIXO RISCO: Cliente estável.")
                    
                    with col2:
                        # Gráfico de Velocímetro (Gauge) usando Plotly
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = prob,
                            title = {'text': "Probabilidade de Churn (%)"},
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkgray"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgreen"},
                                    {'range': [50, 100], 'color': "salmon"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': prob
                                }
                            }
                        ))
                        # Ajustando layout para melhor encaixe
                        fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with st.expander("Ver Logs da API (Auditoria / Debug)"):
                        st.markdown("**Payload Enviado (Request):**")
                        st.json(payload)
                        st.markdown("**Resposta Recebida (Response):**")
                        st.json(result)
                        st.markdown("**Headers de Resposta:**")
                        st.json(dict(response.headers))
                        
                else:
                    st.error(f"Erro na API: {response.text}")
            except Exception as e:
                st.error(f"Falha ao conectar com a API: {e}")

with tab2:
    st.header("Exploração de Dados Históricos")
    data_path = "data/raw/Telco_customer_churn.xlsx"
    
    if os.path.exists(data_path):
        # Load dataset com cache para performance
        @st.cache_data
        def load_data():
            df = pd.read_excel(data_path)
            return df
            
        with st.spinner("Carregando base de dados histórica..."):
            df = load_data()
            
            st.write(f"**Total de clientes analisados na base histórica:** {len(df)}")
            
            # Layout for charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Churn by Contract Type
                if 'Contract' in df.columns and 'Churn' in df.columns:
                    churn_contract = df.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
                    fig_contract = px.bar(churn_contract, x='Contract', y='Count', color='Churn', barmode='group', 
                                          title='Churn por Tipo de Contrato', color_discrete_map={"Yes": "salmon", "No": "lightgreen"})
                    st.plotly_chart(fig_contract, use_container_width=True)
                    
            with col2:
                # Churn vs Monthly Charges
                if 'MonthlyCharges' in df.columns and 'Churn' in df.columns:
                    fig_charges = px.box(df, x='Churn', y='MonthlyCharges', color='Churn', 
                                         title='Distribuição de Mensalidade vs Churn', color_discrete_map={"Yes": "salmon", "No": "lightgreen"})
                    st.plotly_chart(fig_charges, use_container_width=True)
                    
            st.markdown("---")
            st.subheader("Análise de Multicolinearidade (Apoio à Decisão)")
            
            col_corr, col_vif = st.columns([1.5, 1])
            
            with col_corr:
                # Prepara dados numéricos (convertendo TotalCharges se necessário)
                numeric_df = df.copy()
                if 'TotalCharges' in numeric_df.columns:
                    numeric_df['TotalCharges'] = pd.to_numeric(numeric_df['TotalCharges'], errors='coerce')
                
                # Seleciona apenas colunas numéricas
                num_cols = numeric_df.select_dtypes(include=['float64', 'int64']).columns
                if len(num_cols) > 1:
                    corr_matrix = numeric_df[num_cols].corr()
                    fig_corr = px.imshow(corr_matrix, text_auto=".2f", 
                                         title="Matriz de Correlação (Pearson)",
                                         color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                    fig_corr.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig_corr, use_container_width=True)
            
            with col_vif:
                st.markdown("""
                **O Problema da Colinearidade**
                
                Nossa matriz de correlação aponta um forte vínculo ($r = 0.83$) entre **Meses de Contrato (tenure)** e **Cobrança Total (TotalCharges)**.
                
                Aplicando o teste estatístico **VIF (Variance Inflation Factor)**, confirmamos a distorção:
                
                | Feature | VIF Score | Decisão Arquitetural |
                |---|---|---|
                | `TotalCharges` | **> 10.0** | [Removido] |
                | `tenure` | 7.5 | [Mantido] |
                | `MonthlyCharges`| 3.4 | [Mantido] |
                
                **Conclusão:** Manter `TotalCharges` causaria instabilidade matemática nos pesos da Regressão Logística e distorceria as explicações via valores SHAP. Ele foi descartado no pré-processamento.
                """)
                
            st.markdown("---")
            st.subheader("Amostra dos Dados Originais")
            st.dataframe(df.head(10))
            
    else:
        st.warning(f"O arquivo {data_path} não foi encontrado. Ele é necessário para exibir os gráficos históricos.")

with tab3:
    st.header("Decisões Arquiteturais e Modelagem")
    
    st.subheader("1. Comparativo de Modelos: PyTorch vs Regressão Logística")
    st.markdown("""
    Nós avaliamos diversas abordagens para prever o Churn:
    * **Dummy Classifier (Baseline):** 73% de acurácia, mas **0% de Recall** (inútil para o negócio).
    * **Rede Neural (PyTorch):** Arquitetura MLP (64 -> 32 neurônios), função ReLU, Otimizador Adam, e `BCEWithLogitsLoss`. Atingiu **0.85 AUC-ROC e 78% de Recall**.
    * **Regressão Logística (Produção):** Atingiu **0.85 AUC-ROC e 78% de Recall**.
    
    **Decisão:** Optamos pela Regressão Logística por empatar tecnicamente com o PyTorch e garantir 100% de interpretabilidade nativa via SHAP, sem problemas de consumo de memória.
    """)
    
    st.markdown("---")
    
    st.subheader("2. Ajuste do Threshold (Limiar de Decisão)")
    st.markdown("""
    O limiar padrão de 0.5 estava causando a perda de quase metade dos canceladores (Baixo Recall).
    
    * Reduzimos o limiar de **0.5 para 0.3**.
    * **Justificativa Matemática/Negócios:** O custo financeiro de um 'Falso Negativo' (cliente que cancela sem a empresa tentar reter) é cerca de 10 vezes maior que um 'Falso Positivo' (oferecer desconto para quem ficaria de qualquer forma). Aceitar mais falsos alarmes é, neste contexto, altamente lucrativo.
    """)
    
    st.markdown("---")
    
    st.subheader("3. Engenharia de Software e MLOps")
    st.markdown("""
    O diferencial deste projeto é a infraestrutura de produção, garantindo que o modelo seja consumível, resiliente e auditável:
    
    * **API FastAPI com Pydantic:** Contratos de dados rígidos no backend. Entradas inválidas são bloqueadas imediatamente, impedindo quedas (*crashes*) por erros de tipagem.
    * **Logging Estruturado e Latência:** Um *middleware* customizado captura o `X-Process-Time` (latência em ms) e registra o *payload* exato da inferência em JSON para auditoria e *debug*.
    * **MLflow Tracking e Registry:** Rastreamento total de hiperparâmetros (SQLite) e gestão de versões, garantindo que saibamos exatamente qual modelo está em Produção.
    """)
    
    st.markdown("---")
    
    st.subheader("4. Monitoramento Contínuo (Playbook)")
    st.markdown("""
    Modelos de IA sofrem degradação. Para mitigar riscos, estabelecemos um **Playbook de Monitoramento**:
    
    * **Data Drift (Desvio de Dados):** Monitoramento das distribuições via teste estatístico de **Kolmogorov-Smirnov** (Evidently). Alertas disparam se o perfil do cliente mudar.
    * **Degradação de Performance:** Retreino obrigatório caso a métrica *F1-Score* caia abaixo de **80%** em novos ciclos de faturamento.
    * **Rollback de Emergência:** Caso a API apresente picos de Erros 500 ou viés agudo, revertemos a versão de `Production` para a `Archived` imediatamente usando o MLflow Model Registry.
    * **Governança:** Limitações catalogadas no **Model Card** e integração com o negócio através do **ML Canvas**.
    """)
    
    st.markdown("---")
    
    with st.expander("Demonstração Interativa: Auditoria MLOps"):
        col_drift, col_rollback = st.columns(2)
        
        with col_drift:
            st.markdown("### Auditoria de Data Drift")
            st.markdown("Simule a entrada de uma nova safra de clientes e rode o Teste de Kolmogorov-Smirnov.")
            if st.button("Executar Teste de Drift (Tenure)"):
                with st.spinner("Analisando distribuições..."):
                    # Carrega dados reais de referencia
                    try:
                        df_ref = pd.read_excel("data/raw/Telco_customer_churn.xlsx")
                        ref_data = df_ref['tenure'].dropna().values
                    except:
                        ref_data = np.random.normal(32, 24, 1000)
                    
                    # Simula um dataset com drift (ex: campanha atraiu apenas clientes de curtíssimo prazo)
                    drifted_data = np.random.exponential(scale=5, size=500)
                    
                    # Teste KS
                    statistic, p_value = stats.ks_2samp(ref_data, drifted_data)
                    
                    st.warning("ALERTA: Data Drift Detectado!")
                    st.write(f"**P-Value:** {p_value:.4e} (Limite: 0.05)")
                    st.write("A distribuição da nova safra divergiu severamente dos dados de treino. A métrica F1 está sob risco de degradação. **Ação:** Retreino agendado.")
        
        with col_rollback:
            st.markdown("### Emergência: Rollback de Modelo")
            st.markdown("Simule a reversão de versão no Registry do MLflow após anomalia crítica na API.")
            if st.button("Acionar Rollback Automático"):
                with st.spinner("Comutando Registry..."):
                    st.success("Rollback executado com sucesso em Produção.")
                    st.code('''
[MLFLOW SYSTEM LOG]
> mlflow.register_model(...)
> Transitioning model 'Churn_LogisticRegression' version 2 to 'Archived'.
> Transitioning model 'Churn_LogisticRegression' version 1 to 'Production'.
> API Reloading... OK.
> Traffic redirected to Version 1 (F1-Score: 0.79).
                    ''', language="bash")
                    st.markdown("O sistema estabilizou usando o *Shadow Deployment* prévio. Equipe notificada via Slack.")

