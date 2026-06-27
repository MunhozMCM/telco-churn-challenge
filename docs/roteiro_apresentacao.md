#  Roteiro de Apresentação: Detecção de Churn (Telco)

*Dica: Use este roteiro como um guia passo a passo para demonstrar o valor técnico e de negócios do projeto para os seus colegas.*

---

## 1. Introdução: O Desafio de Negócios (2 minutos)
* **O Problema:** Iniciar explicando que a operadora de telecomunicações está sofrendo com uma alta taxa de evasão de clientes (Churn). O custo de perder um cliente (perda de LTV) é muito maior do que o custo de oferecer um desconto para retê-lo.
* **A Solução:** "Para resolver isso, desenvolvemos um produto de Machine Learning de ponta a ponta. Não apenas treinamos uma Rede Neural (MLP), mas construímos toda a infraestrutura para que a equipe de retenção possa usar essa inteligência em tempo real."

## 2. O Motor do Projeto: A API (3 minutos)
*(Se possível, abra a tela do Swagger - `http://127.0.0.1:8000/docs` durante esta fala)*
* **Tecnologia:** Mostrar que a API foi construída com **FastAPI**, escolhido pela sua alta performance e suporte nativo a requisições assíncronas.
* **Segurança e Validação:** Destacar o uso do `Pydantic`, garantindo que a API só aceite os dados corretos (ex: tipos de contrato, valores mensais). "Se faltar algum dado do cliente, a API recusa a requisição automaticamente, evitando que o modelo faça predições erradas."
* **O Diferencial (Middleware):** Explicar que vocês implementaram um middleware de latência. "A API injeta no cabeçalho (header) de cada resposta o tempo exato que ela demorou para processar o dado. Além disso, ela gera logs estruturados em JSON de forma silenciosa, preparando o terreno para nosso monitoramento de *Data Drift*."

## 3. A Interface: O Dashboard Interativo (3 minutos)
*(Abra o Streamlit - `http://localhost:8501` ou a URL do Cloudflare)*
* **Experiência do Usuário:** Mostrar que enquanto a API é para sistemas, o **Dashboard em Streamlit** é para a equipe de negócios.
* **Demonstração ao Vivo:** 
  1. Preencha os dados de um cliente "Seguro" (contrato de 2 anos, muito tempo de casa) e clique em prever. Mostre que o velocímetro fica verde (Baixo Risco).
  2. Preencha os dados de um cliente de "Alto Risco" (contrato mensal recente, conta alta). Clique em prever e mostre o alerta vermelho.
* **A Métrica de Ouro:** Chame a atenção para a caixinha de **Latência da API** na tela. "Vejam que não apenas prevemos o risco, mas provamos que a nossa arquitetura responde em pouquíssimos milissegundos, permitindo que isso seja usado em larga escala num call center."

## 4. Governança e MLOps: A Documentação (3 minutos)
*(Mostre rapidamente os arquivos gerados ou resuma os conceitos)*
* **Model Card:** Explicar que vocês criaram uma "Bula" do modelo. "Nós documentamos abertamente as limitações do nosso modelo, reconhecendo que uma Rede Neural age como uma caixa-preta, e mapeamos as métricas técnicas priorizando o Recall (para evitar os falsos negativos, que são clientes saindo sem que a gente tente retê-los)."
* **Playbook de Monitoramento:** Citar que o projeto já nasceu pronto para o Dia 2. "Temos um manual claro de como monitorar tanto a saúde técnica do servidor (uso de CPU, latência) quanto a saúde preditiva do modelo, preparando a integração com Prometheus e Grafana."
* **ML Canvas:** Mostrar que a ponte entre Engenharia e Negócios está documentada no Canvas.

## 5. Conclusão e Próximos Passos (1 minuto)
* Concluir afirmando que o modelo já está servido via `systemd` (robusto contra reinicializações) e exposto via túneis seguros (Cloudflare).
* **Próximos passos (Fase 2):** Citar que a fundação está pronta para a criação do CI/CD, execução de testes unitários automatizados (que já estão escritos na pasta `tests`) e escalabilidade em nuvem.

---
**Boa apresentação!** Vá com confiança, a infraestrutura que vocês montaram está impecável.
