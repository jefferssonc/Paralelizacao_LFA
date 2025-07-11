import heapq
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from multiprocessing import Pool, freeze_support
import time


class TuringMachine:
    """
    Representa uma máquina de Turing com estados, fita, transições,
    estado inicial e estados de aceitação.

    A máquina pode ser executada de forma sequencial ou paralela para
    buscar uma fita final aceita com custo mínimo.
    """

    def __init__(self, states, tape, transitions, start_state, accept_states):
        """
        Inicializa a máquina de Turing.

        Args:
            states (set): Conjunto de estados da máquina.
            tape (list): Lista inicial representando a fita.
            transitions (dict): Dicionário com transições no formato
                {(estado, símbolo): [(próximo_estado, símbolo_a_escrever, movimento, peso), ...]}.
            start_state (str): Estado inicial da máquina.
            accept_states (set): Conjunto de estados de aceitação.
        """
        self.states = states
        self.tape = list(tape) + ['_']
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states
        self.head = 0
        self.history = []

    def run_sequential(self):
        """
        Executa a máquina de Turing de forma sequencial para encontrar
        uma fita final aceita com custo mínimo.

        Utiliza uma fila de prioridade (heap) para explorar os estados.

        Returns:
            tuple or None: Retorna uma tupla contendo a fita final como string,
            o caminho percorrido como lista de transições, e o custo total.
            Retorna None se não encontrar estado de aceitação.
        """
        priority_queue = [(0, self.start_state, self.head, list(self.tape), [])]
        visited = {}

        while priority_queue:
            cost, state, head, tape, path = heapq.heappop(priority_queue)

            if (state, head) in visited and visited[(state, head)] <= cost:
                continue
            visited[(state, head)] = cost

            self.history.append((state, head, tape.copy(), path, cost))

            if state in self.accept_states:
                return ''.join(tape), path, cost

            current_symbol = tape[head] if 0 <= head < len(tape) else '_'
            transitions_list = self.transitions.get((state, current_symbol), [])

            for transition in transitions_list:
                next_state, write_symbol, move, weight = transition

                new_tape = tape.copy()
                new_tape[head] = write_symbol
                new_head = head + (1 if move == 'R' else -1)

                if new_head < 0:
                    new_tape.insert(0, '_')
                    new_head = 0
                elif new_head >= len(new_tape):
                    new_tape.append('_')

                heapq.heappush(priority_queue, (
                    cost + weight,
                    next_state,
                    new_head,
                    new_tape,
                    path + [(state, next_state, weight)]
                ))

        return None

    def _apply_transition(self, args):
        """
        Aplica uma transição da máquina a partir dos argumentos recebidos.

        Args:
            args (tuple): Tupla contendo (cost, state, head, tape, path, transition).

        Returns:
            tuple: Novo estado da fila de prioridade no formato
                (novo_custo, próximo_estado, nova_pos_cabeça, nova_fita, novo_caminho).
        """
        cost, state, head, tape, path, transition = args
        next_state, write_symbol, move, weight = transition

        new_tape = tape.copy()
        new_tape[head] = write_symbol
        new_head = head + (1 if move == 'R' else -1)

        if new_head < 0:
            new_tape.insert(0, '_')
            new_head = 0
        elif new_head >= len(new_tape):
            new_tape.append('_')

        new_cost = cost + weight
        new_path = path + [(state, next_state, weight)]

        return (new_cost, next_state, new_head, new_tape, new_path)

    def run_parallel(self, num_processes=4):
        """
        Executa a máquina de Turing utilizando multiprocessing para aplicar
        transições em paralelo.

        Args:
            num_processes (int): Número de processos paralelos a utilizar.

        Returns:
            tuple or None: Tupla contendo fita final, caminho percorrido e custo total
            caso encontre estado de aceitação. None caso contrário.
        """
        priority_queue = [(0, self.start_state, self.head, list(self.tape), [])]
        visited = {}

        with Pool(num_processes) as pool:
            while priority_queue:
                cost, state, head, tape, path = heapq.heappop(priority_queue)

                if (state, head) in visited and visited[(state, head)] <= cost:
                    continue
                visited[(state, head)] = cost

                self.history.append((state, head, tape.copy(), path, cost))

                if state in self.accept_states:
                    return ''.join(tape), path, cost

                current_symbol = tape[head] if 0 <= head < len(tape) else '_'
                transitions_list = self.transitions.get((state, current_symbol), [])

                args_list = [
                    (cost, state, head, tape, path, transition)
                    for transition in transitions_list
                ]

                results = pool.map(self._apply_transition, args_list)

                for result in results:
                    heapq.heappush(priority_queue, result)

        return None


def draw_graph(transitions, history, save_gif=False):
    """
    Desenha o grafo de transições da máquina e anima a execução
    passo a passo.

    Args:
        transitions (dict): Dicionário das transições da máquina.
        history (list): Histórico da execução contendo estados, cabeçote,
            fita, caminho e custo.
        save_gif (bool): Indica se a animação deve ser salva como GIF.

    Returns:
        None
    """
    G = nx.DiGraph()

    for (state, symbol), transitions_list in transitions.items():
        for next_state, write_symbol, move, weight in transitions_list:
            G.add_edge(state, next_state, label=f'{symbol}/{write_symbol}, {move}, {weight}')

    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame):
        ax.clear()
        state, head, tape, path, cost = history[frame]

        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                edge_color='gray', node_size=1000, font_size=10, ax=ax)

        if path:
            last_edge = path[-1]
            nx.draw_networkx_edges(G, pos, edgelist=[(last_edge[0], last_edge[1])],
                                   width=2.5, edge_color='red', ax=ax)

        labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=6, ax=ax)

        ax.set_title(f"Passo {frame+1}: Estado = {state}, Cabeça = {head}, Custo = {cost}\nFita: {''.join(tape)}")

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=1000, repeat=False)
    if save_gif:
        ani.save("turing_machine.gif", writer="pillow", fps=1)
    plt.show()


if __name__ == '__main__':
    """
    Ponto de entrada do programa. Executa as versões sequencial e paralela da
    máquina de Turing, imprime resultados e exibe animação da execução.
    """
    freeze_support()

    states = {'q0', 'q1', 'q2', 'q3', 'q4', 'q_accept'}
    tape = ['0', '0', '1', '1', '0']

    transitions = {
        ('q0', '0'): [('q1', '0', 'R', 1), ('q2', '1', 'R', 2)],
        ('q1', '0'): [('q2', '1', 'R', 3), ('q3', '0', 'L', 9)],
        ('q2', '1'): [('q3', '0', 'R', 6), ('q2', '0', 'R', 7)],
        ('q3', '1'): [('q4', '1', 'R', 1), ('q2', '1', 'L', 5)],
        ('q4', '0'): [('q_accept', '_', 'R', 6)]
    }

    start_state = 'q0'
    accept_states = {'q_accept'}

    print("\n==> Executando versão SEQUENCIAL:")
    machine_seq = TuringMachine(states, tape, transitions, start_state, accept_states)

    start_time = time.time()
    result_seq = machine_seq.run_sequential()
    end_time = time.time()

    if result_seq:
        print("Fita final:", result_seq[0])
        print("Caminho percorrido:", result_seq[1])
        print("Soma ponderada:", result_seq[2])
    else:
        print("Rejeitado")

    print(f"Tempo SEQUENCIAL: {end_time - start_time:.4f} segundos")

    print("\n==> Executando versão PARALELA:")
    machine_par = TuringMachine(states, tape, transitions, start_state, accept_states)

    start_time = time.time()
    result_par = machine_par.run_parallel(num_processes=4)
    end_time = time.time()

    if result_par:
        print("Fita final:", result_par[0])
        print("Caminho percorrido:", result_par[1])
        print("Soma ponderada:", result_par[2])
    else:
        print("Rejeitado")

    print(f"Tempo PARALELO: {end_time - start_time:.4f} segundos")

    draw_graph(transitions, machine_par.history, save_gif=False)
