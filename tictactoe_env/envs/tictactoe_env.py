import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

'''
Tab Game
[0,0,0,0,0,0,0,0,0]

action space {0,1,...,8}  una mossa per ogni casella

obs space min = [0,0,0,0,0,0,0,0,0] max = [2,2,2,2,2,2,2,2,2] . 
stati della tabella. 0 casella vuota, 1 casella con X, 2 casella con O

2 player 
1 per primo agente, 
2 per secondo agente.
Vengono scambiati in automatico
'''
class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.actual_player = 1 # Questa variabile puo assumere i soli valori 1 e 2
        self.invalid_move = 0
        self.valid_move = 0
        self.total_step = 0
        self.state = np.zeros(9)
        self.state = [0,0,0,0,0,0,0,0,0]
        self.action_space = spaces.Discrete(9)
        self.obs_min = np.full((9,), 0)
        self.obs_max = np.full((9,), 2)
        self.observation_space = spaces.Box(low = self.obs_min, high = self.obs_max, dtype = np.int8)
        self.debug_file_name = "TicTacToe.txt"
        self.report_file_name = "report.txt"
        self.report = open(self.report_file_name, "a+")
        self.debug = open(self.debug_file_name, "w+")
        self.game = 0
        self.tris_amount = {}

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.debug.write("\n---------------------------------------------------" + "\n")
        self.debug.write("\nActual Player " + str(self.get_actual_player()) + "\n")
        self.debug.write("\nStato Attuale " + str(self.get_state()) + "\n")
        done = False
        reward = 0
        self.increment_steps()

        # Controllo se la tabella di gioco e' piena in tal caso termino la partita senza reward in quanto sto considerando la prima mossa in eccesso
        # e nessuno dei due ha vinto
        # Se la funzione restituisce False vuol dire che posso ancora fare delle mosse in quanto ci sono caselle libere.
        if(not self.check_if_board_is_full()):
            #Se la casella è 0 vuol dire che e' vuota e che la mossa e' valida e occupo la casella
            if(self.get_state()[action] == 0):
                # Leggo lo stato
                state = self.get_state()
                # Lo aggiorno con la nuova mossa
                state[action] = self.get_actual_player()
                # Reimposto lo stato
                self.set_state(state)
                reward = 1
                self.debug.write("\nMossa valida.\n")
                self.debug.write("\nNuovo Stato " + str(self.get_state()) + " Reward " + str(reward) + "\n")
                self.increment_valid_move()

                # Dopo una mossa valida controllo se qualcuno ha fatto tris e aumento la ricompensa di 19 nel caso il tris l'ha fatto il giocatore che
                # sta giocando in questo momento e termino la partita.
                tris, player, pos = self.check_if_tris_is_performed()
                if (tris and player == self.get_actual_player()):
                    reward = reward + 19
                    self.increment_tris_amount(player)
                    self.debug.write("\nTris = " + str(tris) + " giocatore " + str(player) + " position " + str(pos) + " reward totale: " + str(reward) + "\n")
                    done = True

                # Cambio il giocatore in quanto la mossa e' stata valida e tocca al successivo
                self.change_player()
            else:
                # Non cambio il giocatore in quanto la mossa non e' valida per cui ritenta fino a quando non è valida
                self.increment_invalid_move()
                reward = -10
                self.debug.write("\nMossa non valida. Reward: " + str(reward) + " Tot. Mosse non valide " + str(self.get_invalid_move()) + " su totale passi " + str(self.get_total_steps()) + " ratio " + str(self.get_invalid_move()/self.get_total_steps()) +"\n")

        else:
            # Tabellone di gioco pieno. Il giocatore che avrebbe dovuto muovere inizia una nuova partita.
            # Se sono finito qui vuol dire che la partita e' finita in parita' in quanto controllando ad ogni mossa,
            # un eventuale tris si individua subito
            self.debug.write("\nTabellone di gioco pieno.\n")
            self.debug.write("\nPartita patta: " + str(self.get_state()) + " \n")
            reward = 0
            done = True

        self.debug.write("\nNext Player " + str(self.get_actual_player()) + "\n")
        self.debug.write("\n---------------------------------------------------" + "\n")
        self.debug.flush()
        return np.array(self.get_state()), reward, done, {}

    def reset(self):
        state = np.zeros(9)
        self.set_state(state)
        self.debug.write("\nAmbiente Resettato\n")
        self.debug.flush()
        #Se resetto l'ambiente vuol dire che devo iniziare una nuova partita. Ne porto il conto.
        self.increment_games()
        return np.array(self.get_state())

    def reset_all(self):
        #self.report.close()
        #self.report = open(self.debug_file_name, "a+")
        self.debug.close()
        self.debug = open(self.debug_file_name, "w+")
        # In questa funzione azzero contatori totali come numero di tris fatti, numero di partite giocate, numero di passi
        if(self.get_games() != 0):
            self.report.write("\n-----------------REPORT------------------" + "\n")
            self.report.write("\n" + "Episodi giocati " + str(self.get_total_steps()) +  "\n")
            self.report.write("\n" + "Numero partite " + str(self.get_games())  + "\n")
            self.report.write("\n" + "Numero Tris " + str(self.get_tris_amount()) + "\n")
            self.report.write("\n" + "Ratio Partite/Tris " + str(sum(self.get_tris_amount().values())/self.get_games()) + "\n")
            self.report.write("\n" + "Mosse non valide " + str(self.get_invalid_move()) + "\n")
            self.report.write("\n" + "Mosse valide " + str(self.get_valid_move()) + "\n")
            self.report.write("\n" + "Ratio Episodi/Mosse Non Valide " + str(self.get_invalid_move()/self.get_total_steps()) + "\n")
            self.report.write("\n" + "Ratio Episodi/Mosse Valide " + str(self.get_valid_move() / self.get_total_steps()) + "\n")
            self.report.write("\n-----------------END REPORT------------------" + "\n")
        self.report.flush()
        state = np.zeros(9)
        self.set_state(state)
        self.set_actual_player(1)
        self.set_total_steps(0)
        self.set_invalid_move(0)
        self.set_tris_amount({})
        self.set_game(0)
        self.set_valid_move(0)

    def increment_valid_move(self):
        self.valid_move += 1

    def set_valid_move(self, val):
        self.valid_move = val

    def get_valid_move(self):
        return self.valid_move

    def increment_tris_amount(self, player):
        if(self.get_tris_amount().get(player) == None):
            self.get_tris_amount().update({player : 1})
        else:
            actual_val = self.get_tris_amount().get(player)
            new_val = actual_val + 1
            self.get_tris_amount()[player] = new_val

    def set_tris_amount(self, val):
        self.tris_amount = val

    def get_tris_amount(self):
        return self.tris_amount

    def increment_games(self):
        self.game += 1

    def get_games(self):
        return self.game

    def set_game(self, val):
        self.game = val

    def increment_steps(self):
        self.total_step += 1

    def get_total_steps(self):
        return self.total_step

    def set_total_steps(self, val):
        self.total_step = val

    def set_invalid_move(self, val):
        self.invalid_move = val

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def set_obs_space(self, obs):
        self.observation_space = obs

    def get_obs_space(self):
        return self.observation_space

    def set_action_space(self, actions):
        self.action_space = actions

    def get_action_space(self):
        return self.action_space

    def set_actual_player(self, val):
        assert val == 1 or val == 2, "invalid player"
        self.actual_player = val

    def get_actual_player(self):
        return self.actual_player

    def increment_invalid_move(self):
        self.invalid_move = self.invalid_move + 1

    def get_invalid_move(self):
        return self.invalid_move

    def change_player(self):
        if(self.get_actual_player() == 1):
            self.set_actual_player(2)
        elif(self.get_actual_player() ==  2):
            self.set_actual_player(1)
        else:
            pass

    def check_if_board_is_full(self):
        board_full = True
        for val in self.state:
            if(val == 0):
                board_full = False
                break
        return board_full

    def check_if_tris_is_performed(self):
        tris = False
        player = 0
        pos = False
        # Check rows
        tris, player, pos = self.check_tris_for_rows()
        if(tris):
            return tris, player, pos

        # Check columns
        tris, player, pos = self.check_tris_for_columns()
        if(tris):
            return tris, player, pos

        # Check diags
        tris, player, pos = self.check_tris_for_diags()
        if(tris):
            return tris, player, pos

        return tris, player, pos

    def check_tris_for_rows(self):
        tris = False
        player = 0
        pos = None
        # Essendo che ad ogni mossa valida faccio il controllo, se qualcuno ha fatto un tris deve necessariamente essere
        # Il giocatore che sta giocando in questo momento
        actual_player = self.get_actual_player()
        if(self.state[0] == actual_player and self.state[1] == actual_player and self.state[2] == actual_player):
            tris = True
            player = actual_player
            pos = "first row"
        elif(self.state[3] == actual_player and self.state[4] == actual_player and self.state[5] == actual_player):
            tris = True
            player = actual_player
            pos = "second row"
        elif(self.state[6] == actual_player and self.state[7] == actual_player and self.state[8] == actual_player):
            tris = True
            player = actual_player
            pos = "third row"
        else:
            pass

        return tris, player, pos

    def check_tris_for_columns(self):
        tris = False
        player = 0
        pos = None
        actual_player = self.get_actual_player()
        if(self.state[0] == actual_player and self.state[3] == actual_player and self.state[6] == actual_player):
            tris = True
            player = actual_player
            pos = "first col"
        elif(self.state[1] == actual_player and self.state[4] == actual_player and self.state[7] == actual_player):
            tris = True
            player = actual_player
            pos = "second col"
        elif(self.state[2] == actual_player and self.state[5] == actual_player and self.state[8] == actual_player):
            tris = True
            player = actual_player
            pos = "third col"
        else:
            pass

        return tris, player, pos

    def check_tris_for_diags(self):
        tris = False
        player = 0
        pos = None
        actual_player = self.get_actual_player()
        if(self.state[0] == actual_player and self.state[4] == actual_player and self.state[8] == actual_player):
            tris = True
            player = actual_player
            pos = "l_to_r_diag"
        elif(self.state[2] == actual_player and self.state[4] == actual_player and self.state[6] == actual_player):
            tris = True
            player = actual_player
            pos = "r_to_l_diag"
        else:
            pass

        return tris, player, pos

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass