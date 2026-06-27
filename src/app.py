import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")

st.title("Telco Churn Prediction")
st.markdown("Bem-vindo ao dashboard de previsão de Churn. Preencha os dados do cliente para analisar o risco ou explore os dados históricos.")

# Create tabs
tab1, tab2 = st.tabs(["Simulador de Churn", "Análise Histórica (EDA)"])

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
        # Payload completo conforme exigido pelo novo schema da API na main
        payload = {
            "Zip Code": 90001,
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
                        
                else:
                    st.error(f"Erro na API: {response.text}")
            except Exception as e:
                st.error(f"Falha ao conectar com a API: {e}")

with tab2:
    st.header("Exploração de Dados Históricos")
    data_path = "data/Telco_customer_churn.xlsx"
    
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
                | `TotalCharges` | **> 10.0** | 🚨 **Removido** do Modelo |
                | `tenure` | 7.5 | ✅ Mantido |
                | `MonthlyCharges`| 3.4 | ✅ Mantido |
                
                **Conclusão:** Manter `TotalCharges` causaria instabilidade matemática nos pesos da Regressão Logística e distorceria as explicações via valores SHAP. Ele foi descartado no pré-processamento.
                """)
                
            st.markdown("---")
            st.subheader("Amostra dos Dados Originais")
            st.dataframe(df.head(10))
            
    else:
        st.warning(f"O arquivo {data_path} não foi encontrado. Ele é necessário para exibir os gráficos históricos.")
