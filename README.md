# Atividade de Paralelização de LFA

Este repositório contém a atividade de paralelização relacionada à disciplina de Linguagens Formais e Autômatos (LFA).

## Projeto escolhido

O projeto utilizado como base foi o seguinte:

[Máquina de Turing Ponderada com Dijkstra](https://github.com/thalesvalente/teaching/tree/main/formal-languages-and-automata/3-projects/2024-2/p3/G4_MAQUINA_DE_TURING_PONDERADA_COM_DIJKSTRA)

## Observação importante

O tempo de execução da versão paralela está significativamente maior (aproximadamente 0.9 segundos) em comparação com a versão sequencial, que é quase instantânea.

Isso é um comportamento comum quando o paralelismo envolve overhead considerável, como comunicação entre processos e serialização dos dados, especialmente em tarefas pequenas ou rápidas, onde o custo da paralelização pode superar os ganhos de desempenho.

---


