# Título do trabalho

INF0413 - Processamento Digital de Sinais e Imagens

---

## Autores

- João Victor Castro - joao_castro@discente.ufg.br
- João Victor Borges - joao.borges@discente.ufg.br
- José Ricardo Fleury - josefleury@discente.ufg.br
- Lucas Wanderley Alves - lucas.alves2@discente.ufg.br
- Priscila Rocha Maia - priscila.maia@discente.ufg.br

---

## Resumo

Este trabalho explora tecnologias de IA conversacional em tempo real aplicadas ao atendimento ao cliente, com foco em conversação multimodal. São avaliados modelos de reconhecimento de fala (STT) e síntese de voz (TTS) para inferências rápidas e precisas, utilizando ferramentas como Whisper, NVIDIA RIVA e Voice Activity Detection (VAD). Apesar dos desafios em fine-tuning e configuração de RAG, o estudo identifica limitações e propõe melhorias em streaming e recuperação de informações. O dataset DailyTalk apoia a criação de interações mais naturais e eficientes, destacando a relevância da integração de modelos otimizados para aplicações práticas e fluídas.

### Palavras-chave

Conversação Multimodal - Reconhecimento de Fala - Atendimento Automatizado - Transcrição em Tempo Real - IA Conversacional

---

## Índice

- ### [Seção I. Introdução e revisão bibliográfica](#introdução)
- ### [Seção II. Fundamentos teóricos](#fundamentos-teóricos)
- ### [Seção III. Metodologia](#metodologia)
- ### [Seção IV. Resultados e Conclusões](#resultados-e-conclusões)
- ### [Referências](#referências)
- ### [Apêndices](#apêndices)

## Introdução

A área de atendimento ao cliente enfrenta desafios ao buscar automações que realmente ofereçam uma experiência de comunicação *natural e satisfatória*. Tecnologias tradicionais, como as **Unidades de Resposta Audível (URAs)**, embora tragam praticidade, ainda limitam o usuário a um diálogo rígido, distante da fluidez e naturalidade da comunicação humana. Para superar essa limitação, surgem estudos e avanços em **Conversação Multimodal em Tempo Real**, que integram reconhecimento de fala, processamento de linguagem natural e síntese de voz para criar um atendimento que se assemelha a uma conversa entre pessoas.

Nosso estudo foca em avaliar como essas tecnologias de **IA Conversacional em Tempo Real** performam especificamente em relação ao reconhecimento automático de fala (STT). O objetivo é investigar em que medida essas soluções de IA conseguem **interpretar e responder de forma rápida e precisa**, reduzindo a necessidade de transcrição manual e melhorando o tempo de resposta em atendimento.

### Objetivo

Este trabalho foca em duas abordagens principais:

1. Avaliação de modelos que compreendam e respondam diretamente em áudio, minimizando o tempo gasto em transcrições.
2. AUso de modelos otimizados de **Text-to-Speech (TTS)** e **Speech-to-Text (STT)** para inferências em tempo real, permitindo respostas imediatas e contínuas.

### Dataset

O dataset escolhido, [DailyTalk](https://github.com/keonlee9420/DailyTalk?tab=readme-ov-file), é voltado para sistemas de **conversão de texto em fala (TTS)** em contextos conversacionais. Suas características incluem:
- **Categorias ricas**: emoções e intenções de fala anotadas.
- **Enfoque em diálogos reais**: facilita treinamento para respostas mais naturais.
- **Ampla cobertura temática**: ideal para modelos que precisam lidar com variados contextos.
Esse conjunto de dados possibilita a criação de respostas mais *naturais e eficientes* no atendimento ao cliente, ajustando-se às emoções e intenções do usuário.

### Literatura
- [DailyTalk](https://github.com/keonlee9420/DailyTalk)
- [HuggingFace - Speech to Speech](https://github.com/huggingface/speech-to-speech)
- [Using the Natural Language Paradigm (NLP)](https://pubmed.ncbi.nlm.nih.gov/16889934/)
- [Listen Again and Choose the Right Answer](https://arxiv.org/abs/2405.10025)
- [Realtime API](https://openai.com/index/introducing-the-realtime-api/)
- [E2 TTS](https://www.microsoft.com/en-us/research/project/e2-tts/)
- [ClarQ-LLM](https://arxiv.org/pdf/2409.06097)
- [Beyond Prompts: Dynamic Conversational
Benchmarking of Large Language Models](https://www.arxiv.org/pdf/2409.20222)
- [TTS Arena](https://huggingface.co/blog/arena-tts)

---

## Fundamentos Teóricos

### 1. Speech-to-Text (STT)
O módulo de transcrição utiliza o modelo **Whisper S2T**, desenvolvido pela OpenAI, fundamentado em arquiteturas de redes neurais Transformer. Esta escolha se justifica pela eficiência do modelo em capturar dependências temporais em sequências de áudio por meio de mecanismos de atenção. A capacidade de detecção precisa de bordas na fala — ou seja, a identificação de pausas e o fim das interações — é essencial para garantir que as respostas automatizadas sejam inseridas de forma otimizada. Essa funcionalidade é crítica em ambientes de atendimento ao cliente, onde a latência mínima proporcionada pela transcrição em tempo real melhora significativamente a experiência do usuário. A conversão ágil de áudio em texto garante que a compreensão da solicitação do usuário ocorra de maneira eficiente, criando a base para as etapas subsequentes de processamento.

---

### 2. Retrieval
A etapa de recuperação de informações emprega o modelo `e5-multilingual-large`, uma rede neural voltada para a geração de embeddings semânticos em múltiplos idiomas. A conversão do conteúdo do banco de dados em representações vetoriais numéricas facilita a busca por informações relevantes com alta eficiência, mesmo em consultas com variações linguísticas. Este modelo realiza comparações de similaridade semântica, permitindo recuperar resultados que correspondem à intenção do usuário, mesmo quando há desvio terminológico em relação ao texto original. Assim, a recuperação precisa garante que as informações fornecidas ao modelo de linguagem estejam contextualizadas e alinhadas com as necessidades do usuário.

---

### 3. Large Language Model (LLM)
A implementação do **LLaMA 3** envolve a utilização de técnicas de fine-tuning para especializar o modelo em contextos de atendimento. O processo de ajuste fino, realizado com a ferramenta Unsloth, permite adaptar o modelo pré-treinado a domínios específicos, garantindo respostas mais precisas e contextualizadas. A escolha da API ollama otimiza a inferência em tempo real, oferecendo um desempenho consistente em ambientes de produção. Essa combinação assegura que o LLM interprete corretamente as consultas dos usuários e gere respostas que não apenas refletem conhecimento geral, mas também atendem às particularidades do domínio de atendimento automatizado. A capacidade de inferência acelerada minimiza a latência, melhorando a fluidez das interações.

---

### 4. Text-to-Speech (TTS)
Na etapa de vocalização, após testes comparativos, a escolha recaiu sobre a solução de **Text-to-Speech da OpenAI**, devido ao equilíbrio entre naturalidade da voz e eficiência no processamento em streaming. A rede neural utilizada neste TTS mapeia sequências de texto para características acústicas, resultando em uma fala sintetizada que preserva entonação e clareza. A naturalidade da voz melhora a percepção do atendimento automatizado, enquanto a capacidade de processamento em tempo real assegura respostas rápidas e contínuas. Essa escolha foi fundamental para garantir uma experiência auditiva que não comprometa a qualidade do atendimento, mantendo a fluidez e a inteligibilidade necessárias em interações críticas.

Essa combinação de ferramentas, integradas de forma otimizada, sustenta uma pipeline robusta capaz de converter áudio em texto, recuperar informações relevantes, gerar respostas contextuais precisas e convertê-las novamente em fala com alta qualidade. Isso resolve o desafio de criar um sistema de atendimento automatizado eficiente e centrado no usuário.

---

## Metodologia

![pdsi_pitch_2](https://github.com/user-attachments/assets/17ec3a67-960f-4cf2-a21c-81065ca7c5bf)

---

## Resultados e Conclusões

### Limitantes do Fine-Tuning

O modelo que tentamos aplicar o fine-tuning não respondeu bem às nossas expectativas devido às limitações impostas pelo escopo do projeto. Nossa base de dados, sendo muito restrita, resultou em uma precisão insatisfatória no treinamento do modelo. A metodologia utilizada, por sua vez, contribuiu para essa deficiência, já que a abordagem adotada não conseguiu explorar todo o potencial do modelo, resultando em uma aplicação superficial e com limitações práticas. Esses fatores combinados destacam a importância de um conjunto de dados mais robusto e uma abordagem metodológica mais refinada para alcançar os resultados desejados.

---

### Problemas na Configuração do RAG

Além disso, identificamos um erro na configuração do processo de Recuperação de Informação (**RAG - Retrieval-Augmented Generation**). A indexação dos dados foi realizada de forma inadequada, pois as informações não foram corretamente categorizadas em classes e subclasses, ficando concentradas em um único conjunto. Essa falha prejudicou a capacidade do modelo de acessar informações relevantes e específicas durante a geração de respostas, comprometendo a precisão e a contextualização das respostas geradas.

---

### Avaliação de Modelos de Conversão de Fala em Texto

Quanto à conversão de fala em texto, inicialmente utilizamos o modelo **Whisper S2T** da OpenAI, que demonstrou excelente desempenho em tarefas de transcrição offline devido à sua arquitetura baseada em Transformers e sua capacidade de capturar dependências temporais em sequências de áudio. 

No entanto, percebemos que ele não atende adequadamente ao nosso objetivo, que é realizar conversões de áudio em texto em tempo real (*streaming*). O Whisper apresenta limitações nesse aspecto, já que sua arquitetura não foi projetada para tarefas de streaming, dependendo de um processamento mais lento e em lotes, o que introduz latência nas interações.

---

### Substituição pelo NVIDIA RIVA

Diante disso, decidimos substituir o Whisper pelo **NVIDIA RIVA**, um framework avançado que oferece suporte nativo para transcrição de áudio em tempo real. O RIVA se destaca por sua capacidade de streaming eficiente, oferecendo menor latência e maior precisão em aplicações voltadas para interações ao vivo. 

Sua arquitetura modular e otimizada para GPUs permite a escalabilidade do sistema, além de integrar facilmente outros serviços, como tradução em tempo real e análise de sentimento. Essas características fazem do RIVA uma escolha mais alinhada ao nosso propósito.

---

### Implementação de Voice Activity Detection (VAD)

Por fim, para aprimorar ainda mais o sistema, estamos considerando a implementação de um modelo dedicado de detecção de fala (**Voice Activity Detection - VAD**). Esse modelo seria responsável por identificar o início e o fim das falas de forma mais precisa, garantindo que apenas os segmentos relevantes sejam enviados para processamento. 

Essa abordagem reduzirá ainda mais a latência e permitirá uma interação mais fluida, além de diminuir a carga de processamento do sistema ao filtrar ruídos e pausas desnecessárias. O uso de um VAD também complementará a funcionalidade do RIVA, criando uma solução mais robusta e eficiente para nosso projeto.

---

## Referências

- [1] Hu, Y., Chen, C., Qin, C., Zhu, Q., Chng, E. S., and Li, R., “Listen Again and Choose the Right Answer: A New Paradigm for Automatic Speech Recognition with Large Language Models”, <i>arXiv e-prints</i>, Art. no. arXiv:2405.10025, 2024. doi:10.48550/arXiv.2405.10025.
- [2] Gan, Y., Li, C., Xie, J., Wen, L., Purver, M., and Poesio, M., “ClarQ-LLM: A Benchmark for Models Clarifying and Requesting Information in Task-Oriented Dialog”, <i>arXiv e-prints</i>, Art. no. arXiv:2409.06097, 2024. doi:10.48550/arXiv.2409.06097.
- [3] Leblanc LA, Geiger KB, Sautter RA, Sidener TM. Using the Natural Language Paradigm (NLP) to increase vocalizations of older adults with cognitive impairments. Res Dev Disabil. 2007 Jul-Sep;28(4):437-44. doi: 10.1016/j.ridd.2006.06.004. Epub 2006 Aug 4. PMID: 16889934.
- [4] Eskimez, S. E., “E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS”, <i>arXiv e-prints</i>, Art. no. arXiv:2406.18009, 2024. doi:10.48550/arXiv.2406.18009.
- [5] Castillo-Bolado, D., Davidson, J., Gray, F., and Rosa, M., “Beyond Prompts: Dynamic Conversational Benchmarking of Large Language Models”, <i>arXiv e-prints</i>, Art. no. arXiv:2409.20222, 2024. doi:10.48550/arXiv.2409.20222.
- GITHUB. WhisperS2T. Disponível em: https://github.com/shashikg/WhisperS2T. Acesso em: 21 nov. 2024.
- GITHUB. eSpeak NG. Disponível em: https://github.com/espeak-ng/espeak-ng. Acesso em: 21 nov. 2024.
- OLLAMA. Ollama: AI Models On Your Device. Disponível em: https://ollama.com/. Acesso em: 21 nov. 2024.
- UNSLOTH AI. Unsloth.ai. Disponível em: https://unsloth.ai/. Acesso em: 21 nov. 2024.
- GITHUB. Issue #3149: llama.cpp. Disponível em: https://github.com/ggerganov/llama.cpp/issues/3149. Acess- o em: 21 nov. 2024.
- GITHUB. Coqui TTS. Disponível em: https://github.com/coqui-ai/TTS. Acesso em: 21 nov. 2024.

---

## Apêndices

---

- [Notebook Desenvolvido](https://drive.google.com/drive/folders/1XW7sZDMX9NCREb7_wHeIBnXcTc2GEgZq)

- [Slides da Apresentação](https://www.canva.com/design/DAGYR8-bqYM/iDOyd7jqgCgaX8AnsDOAsQ/view?utm_content=DAGYR8-bqYM&utm_campaign=designshare&utm_medium=link&utm_source=editor)
