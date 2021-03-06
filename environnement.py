import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication

from MainWindow import MainWindow
from agent import Agent

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

TAUX_EVAPORATION = 0.001
DISTANCE_DIFFUSION = 2
DIMINUTION_INTENSITE = 1 / DISTANCE_DIFFUSION
INTENSITE_MAX = 1

class Environnement:
    taille = 50
    env = np.zeros((taille, taille))
    listeAgent = []
    listePosAgent = []
    liste_cluster = []
    liste_nb_ite = []
    liste_collaboration = []
    liste_robot_attente = []
    liste_pheromone_diff_zero = []
    liste_pheromone = np.zeros((taille, taille))
    index_agent_aide = -1

    def alea(self):
        x = random.randint(0, self.taille - 1)
        return x

    def init_env(self):
        ##Creation Objet A
        for i in range(200):
            vide = False
            x = self.alea()
            y = self.alea()
            while vide == False:
                x = self.alea()
                y = self.alea()
                if self.env[x][y] == 0:
                    vide = True
            self.env[x][y] = 1

        ##Creation Objet B
        for i in range(200):
            vide = False
            x = self.alea()
            y = self.alea()
            while vide == False:
                x = self.alea()
                y = self.alea()
                if self.env[x][y] == 0:
                    vide = True
            self.env[x][y] = 2

        ##Creation Objet C
        for i in range(50):
            vide = False
            x = self.alea()
            y = self.alea()
            while vide == False:
                x = self.alea()
                y = self.alea()
                if self.env[x][y] == 0:
                    vide = True
            self.env[x][y] = 3

        # Creation agent

        for i in range(20):
            x = self.alea()
            y = self.alea()
            agent = Agent(i)
            self.listeAgent.append(agent)
            self.listePosAgent.append([x, y])
            self.liste_robot_attente.append(False)

    def liste_coord(self, p):
        liste = []
        for i in range(self.taille):
            for j in range(self.taille):
                if self.env[i][j] == p:
                    liste.append([j, self.taille - i, self.env[i][j]])
        return liste

    def selecting_nb_cluster(self, X):
        range_n_clusters = [k for k in range(2, 20)]
        avg_init = 0
        nb_cluster = 0
        liste_sil = []
        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            liste_sil.append(silhouette_avg)
            if avg_init <= silhouette_avg:
                nb_cluster = n_clusters
                avg_init = silhouette_avg
            """print(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg)"""
            # filter rows of original data
        # print("Le nombre de cluster : " + str(nb_cluster))
        """Y = np.array(liste_sil)
        X = np.array(range_n_clusters)
        plt.plot(X, Y)
        plt.title("Evolution silhouette")
        plt.xlabel("valeur de k")
        plt.ylabel("Silhouette")
        plt.show()"""
        return nb_cluster

    def deplace(self, nb_iteration):
        cmpt = 0

        while cmpt <= nb_iteration:
            # on choisit l'agent auquel on donne la parole
            choix = random.randint(0, 19)
            agent = self.listeAgent[choix]
            pos_agent_x = self.listePosAgent[choix][0]
            pos_agent_y = self.listePosAgent[choix][1]

            self.liste_pheromone = self.liste_pheromone * (1-TAUX_EVAPORATION)

            #On regarde si ce robot est associ?? ?? un autre robot
            suiveur = -1
            placegagent = 0
            for k in range(len(self.liste_collaboration)):
                if self.liste_collaboration[k][0] == choix:
                    suiveur = self.liste_collaboration[k][1]
                    placegagent = 0
                if self.liste_collaboration[k][1] == choix:
                    suiveur = self.liste_collaboration[k][0]
                    placegagent =1

            #On construit la liste des position pr??sente autour de l'agent ainsi que les ph??romone
            listePosautour, listeFeromoneAutour = self.position_phero_autour(pos_agent_x, pos_agent_y)

            #On regarde si un robot est en attente (s'il appel ?? l'aide)
            Robot = None
            robot_en_attente = False
            for k in range (len(self.liste_robot_attente)):
                if self.listePosAgent[k][0] == pos_agent_x and self.listePosAgent[k][1] == pos_agent_y and choix!=k and self.liste_robot_attente[k]==True:
                    Robot = k
                    robot_en_attente = True

            #L'agent fait sa boucle perception/action
            enAttente, Collab, arrete, newPosition, act = agent.perception_action(self.env[pos_agent_x][pos_agent_y],
                                                                                  self.listePosAgent[choix],
                                                                                  listeFeromoneAutour,
                                                                                  listePosautour, robot_en_attente)

            #Si l'agent rentre en collaboration
            if Collab == True:
                self.listeAgent[Robot].tenir = 3
                self.listeAgent[Robot].appel = 0
                self.liste_robot_attente[Robot] = False
                self.liste_robot_attente[choix] = False
                self.liste_collaboration.append([Robot, agent.id])
                for k in range(len(listePosautour)):
                    self.liste_pheromone[listePosautour[k][0]][listePosautour[k][1]] = 0
                    self.liste_pheromone[pos_agent_x][pos_agent_y] = 0
                    # TODO : Cas ou d'autre agents sont autour
            #Si l'agent rentre en attente
            if enAttente == True:

                self.liste_robot_attente[choix] = True
                phero = INTENSITE_MAX
                for k in range (DISTANCE_DIFFUSION):
                    for l in range (DISTANCE_DIFFUSION):
                        for i in range (max(k,l)):
                            phero = phero - phero/DISTANCE_DIFFUSION
                        if pos_agent_x + k < self.taille-1 and pos_agent_y + l < self.taille-1:
                            if self.liste_pheromone[k+pos_agent_x][l+pos_agent_y] == 0:
                                self.liste_pheromone[k+pos_agent_x][l+pos_agent_y] = phero
                        if pos_agent_x - k >=0  and pos_agent_y + l < self.taille - 1:
                            if self.liste_pheromone[pos_agent_x - k][pos_agent_y+l] == 0:
                                self.liste_pheromone[pos_agent_x-k][pos_agent_y+l] = phero
                        if pos_agent_x + k < self.taille-1 and pos_agent_y - l >=0:
                            if self.liste_pheromone[pos_agent_x+k][pos_agent_y-l] == 0:
                                self.liste_pheromone[pos_agent_x+k][pos_agent_y-l] = phero
                        if pos_agent_x - k >= 0 and pos_agent_y - l >= 0:
                            if self.liste_pheromone[pos_agent_x-k][pos_agent_y-l] == 0:
                                self.liste_pheromone[pos_agent_x-k][pos_agent_y-l] = phero
                        else :
                            if pos_agent_x + k < self.taille - 1 and pos_agent_y + l < self.taille - 1:
                                self.liste_pheromone[pos_agent_x+k][pos_agent_y+l] = self.liste_pheromone[pos_agent_x+k][pos_agent_y+l] + (1-self.liste_pheromone[pos_agent_x+k][l+pos_agent_y])*phero
                            if pos_agent_x - k >= 0 and pos_agent_y - l >= 0:
                                self.liste_pheromone[pos_agent_x-k][pos_agent_y-l] = self.liste_pheromone[pos_agent_x-k][pos_agent_y-l] + (1-self.liste_pheromone[pos_agent_x-k][pos_agent_y-l])*phero
                            if pos_agent_x + k < self.taille-1  and pos_agent_y - l >= 0:
                                self.liste_pheromone[pos_agent_x + k][pos_agent_y - l] = self.liste_pheromone[pos_agent_x + k][pos_agent_y - l] + (1 - self.liste_pheromone[pos_agent_x + k][pos_agent_y - l]) * phero
                            if pos_agent_x - k >= 0 and pos_agent_y + l < self.taille -1:
                                self.liste_pheromone[pos_agent_x - k][pos_agent_y + l] = self.liste_pheromone[pos_agent_x - k][pos_agent_y + l] + (1 - self.liste_pheromone[pos_agent_x - k][pos_agent_y + l]) * phero

                """for k in range(len(listePosautour)):
                    if listeFeromoneAutour[k] == 0:
                        self.liste_pheromone[listePosautour[k][0]][listePosautour[k][1]] = IN
                        ##Question???
                self.liste_pheromone[pos_agent_x][pos_agent_y] = 1"""

            #Si le robot arrete d'attendre car aucun robot n'est venu l'aider
            if arrete == True:
                self.liste_robot_attente[choix] = False
                """for k in range(len(listePosautour)):
                    if listeFeromoneAutour[k] == 0:
                        self.liste_pheromone[listePosautour[k][0]][listePosautour[k][1]] = 0
                        self.liste_pheromone[pos_agent_x][pos_agent_y] = 0"""
            #Si l'agent pose un objet
            if act > 0:
                self.env[pos_agent_x][pos_agent_y] = act
            #Si l'agent prend l'objet courant
            if act == -2:
                self.env[pos_agent_x][pos_agent_y] = 0
            """if suivi == 1:
                self.listePosAgent[choix] = newPosition
                self.listePosAgent[Robot] = newPosition
            else:"""
            #On d??place l'agent
            self.listePosAgent[choix] = newPosition
            #On deplace eventuellement l'agent suiveur
            if suiveur != -1:
                if act ==3:
                    self.listeAgent[suiveur].tenir = 0
                    if placegagent == 0:
                        del self.liste_collaboration[self.liste_collaboration.index([choix, suiveur])]
                    else :
                        del self.liste_collaboration[self.liste_collaboration.index([suiveur, choix])]
                else :
                    self.listePosAgent[suiveur] = newPosition

            if cmpt %1000000==0:

                nb_1 = self.selecting_nb_cluster(self.liste_coord(1))
                nb_2 = self.selecting_nb_cluster(self.liste_coord(2))
                nb_3 = self.selecting_nb_cluster(self.liste_coord(3))
                self.liste_cluster.append(nb_2 + nb_1 +nb_3)
                self.liste_nb_ite.append(cmpt)
            #print(cmpt)
            if cmpt == 1000000 or cmpt == 2000000 or cmpt == 3000000 or cmpt % 5000000 == 0:

                app2 = QApplication.instance()
                print("Affichage de la grille pour " + str(cmpt) + " it??rations !")
                if not app2:  # sinon on cr??e une instance de QApplication
                    app2 = QApplication(sys.argv)

                # cr??ation d'une fen??tre avec QWidget dont on place la r??f??rence dans fen
                fen2 = MainWindow(self.env)

                # la fen??tre est rendue visible
                fen2.show()

                # ex??cution de l'application, l'ex??cution permet de g??rer les ??v??nements
                app2.exec_()
                print("Calculs en cours ...")
            cmpt += 1
        Y = np.array(self.liste_cluster)
        X = np.array(self.liste_nb_ite)
        plt.plot(X, Y)
        plt.title("Evolution du nombre de clusters")
        plt.xlabel("Nombre d'it??rations")
        plt.ylabel("Nombre de clusters")
        plt.show()

        app2 = QApplication.instance()
        if not app2:  # sinon on cr??e une instance de QApplication
            app2 = QApplication(sys.argv)

            # cr??ation d'une fen??tre avec QWidget dont on place la r??f??rence dans fen
        fen2 = MainWindow(self.env)

        # la fen??tre est rendue visible
        fen2.show()

        # ex??cution de l'application, l'ex??cution permet de g??rer les ??v??nements
        app2.exec_()


    def position_phero_autour (self, pos_agent_x, pos_agent_y):
        listePosautour = []
        listeFeromoneAutour = []
        if pos_agent_x - 1 >= 0:
            listePosautour.append([pos_agent_x - 1, pos_agent_y])
            listeFeromoneAutour.append(self.liste_pheromone[pos_agent_x - 1][pos_agent_y])
        if pos_agent_x - 1 >= 0 and pos_agent_y + 1 < self.taille:
            listePosautour.append([pos_agent_x - 1, pos_agent_y + 1])
            listeFeromoneAutour.append(self.liste_pheromone[pos_agent_x - 1][pos_agent_y + 1])
        if pos_agent_y + 1 < self.taille:
            listePosautour.append([pos_agent_x, pos_agent_y + 1])
            listeFeromoneAutour.append(self.liste_pheromone[pos_agent_x][pos_agent_y + 1])
        if pos_agent_x + 1 < self.taille and pos_agent_y + 1 < self.taille:
            listePosautour.append([pos_agent_x + 1, pos_agent_y + 1])
            listeFeromoneAutour.append(self.liste_pheromone[pos_agent_x + 1][pos_agent_y + 1])
        if pos_agent_x + 1 < self.taille:
            listePosautour.append([pos_agent_x + 1, pos_agent_y])
            listeFeromoneAutour.append(self.liste_pheromone[pos_agent_x + 1][pos_agent_y])
        if pos_agent_x + 1 < self.taille and pos_agent_y - 1 >= 0:
            listePosautour.append([pos_agent_x + 1, pos_agent_y - 1])
            listeFeromoneAutour.append(self.liste_pheromone[pos_agent_x + 1][pos_agent_y - 1])
        if pos_agent_y - 1 >= 0:
            listePosautour.append([pos_agent_x, pos_agent_y - 1])
            listeFeromoneAutour.append(self.liste_pheromone[pos_agent_x][pos_agent_y - 1])
        if pos_agent_x - 1 >= 0 and pos_agent_y - 1 >= 0:
            listePosautour.append([pos_agent_x - 1, pos_agent_y - 1])
            listeFeromoneAutour.append(self.liste_pheromone[pos_agent_x - 1][pos_agent_y - 1])
        return listePosautour, listeFeromoneAutour