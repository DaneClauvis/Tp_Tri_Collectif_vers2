import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication

from MainWindow import MainWindow
from agent import Agent

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans


class Environnement:
    taille = 50
    env = np.zeros((taille, taille))
    listeAgent = []
    listePosAgent = []
    liste_cluster = []
    liste_nb_ite = []
    liste_collaboration = []
    liste_robot_attente = []
    liste_pheromone = np.zeros((taille, taille))
    accept = False
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
        for i in range(1):
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

            Robot = None
            robot_en_attente = False
            for robot in self.liste_robot_attente:
                if self.listePosAgent[robot][0] == pos_agent_x and self.listePosAgent[robot][
                    1] == pos_agent_y and robot != choix:
                    Robot = robot
                    robot_en_attente = True
            enAttente, Collab, arrete, newPosition, act = agent.perception_action(self.env[pos_agent_x][pos_agent_y],
                                                                                  self.listePosAgent[choix],
                                                                                  self.taille, listeFeromoneAutour,
                                                                                  listePosautour, robot_en_attente)
            if Collab == True:
                del self.liste_robot_attente[self.liste_robot_attente.index(choix)]
                self.liste_collaboration.append([Robot, agent.id])
                for k in range(len(listePosautour)):
                    self.liste_pheromone[listePosautour[k][0]][listePosautour[k][1]] = 0
                    self.liste_pheromone[pos_agent_x][pos_agent_y] = 0
                    # TODO : Cas ou d'autre agent son autour
            if enAttente == True:
                self.liste_robot_attente.append(choix)
                print("JAJOUTE")
                for k in range(len(listePosautour)):
                    if listeFeromoneAutour[k] == 0:
                        self.liste_pheromone[listePosautour[k][0]][listePosautour[k][1]] = 1
                        self.liste_pheromone[pos_agent_x][pos_agent_y] = 1
            if arrete == True:
                del self.liste_robot_attente[self.liste_robot_attente.index(choix)]
                for k in range(len(listePosautour)):
                    if listeFeromoneAutour[k] == 0:
                        self.liste_pheromone[listePosautour[k][0]][listePosautour[k][1]] = 0
                        self.liste_pheromone[pos_agent_x][pos_agent_y] = 0
            if act > 0:
                self.env[pos_agent_x][pos_agent_y] = act
            if act == -2:
                self.env[pos_agent_x][pos_agent_y] = 0

            self.listePosAgent[choix] = newPosition

            # print(array)
            # affichageAgent(listeAgent, listePosAgent)
            print(cmpt)
            """if cmpt %500000==0:

                nb_1 = self.selecting_nb_cluster(self.liste_coord(1))
                nb_2 = self.selecting_nb_cluster(self.liste_coord(2))
                self.liste_cluster.append(nb_2 + nb_1)
                self.liste_nb_ite.append(cmpt)"""
            if cmpt == 1000000 or cmpt == 2000000 or cmpt == 3000000 or cmpt % 5000000 == 0:

                app2 = QApplication.instance()
                print("Affichage de la grille pour " + str(cmpt) + " itérations !")
                if not app2:  # sinon on crée une instance de QApplication
                    app2 = QApplication(sys.argv)

                # création d'une fenêtre avec QWidget dont on place la référence dans fen
                fen2 = MainWindow(self.env)

                # la fenêtre est rendue visible
                fen2.show()

                # exécution de l'application, l'exécution permet de gérer les événements
                app2.exec_()
                print("Calculs en cours ...")
            cmpt += 1
        Y = np.array(self.liste_cluster)
        X = np.array(self.liste_nb_ite)
        plt.plot(X, Y)
        plt.title("Evolution du nombre de clusters")
        plt.xlabel("Nombre d'itérations")
        plt.ylabel("Nombre de clusters")
        plt.show()

        app2 = QApplication.instance()
        if not app2:  # sinon on crée une instance de QApplication
            app2 = QApplication(sys.argv)

            # création d'une fenêtre avec QWidget dont on place la référence dans fen
        fen2 = MainWindow(self.env)

        # la fenêtre est rendue visible
        fen2.show()

        # exécution de l'application, l'exécution permet de gérer les événements
        app2.exec_()
