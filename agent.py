import random
from random import randint


def update_memoire(actual, memoire, appel):
    if len(memoire) > 10:
        del memoire[0]
    if appel == 0:
        memoire.append(actual)
    else:
        memoire.append(0)
    return memoire


def prop_Agents(liste, objet):
    f = 0
    f_c = 0
    if objet == 1:
        for i in range(len(liste)):
            if liste[i] == objet:
                f = f + 1
            if liste[i] == 2:
                f_c = f_c + 1
            if liste[i] == 3:
                f_c = f_c + 1
    elif objet == 2:
        for i in range(len(liste)):
            if liste[i] == objet:
                f = f + 1
            if liste[i] == 1:
                f_c = f_c + 1
            if liste[i] == 3:
                f_c = f_c + 1
    elif objet == 3:
        for i in range(len(liste)):
            if liste[i] == objet:
                f = f + 1
            if liste[i] == 1:
                f_c = f_c + 1
            if liste[i] == 2:
                f_c = f_c + 1
    f = (f + f_c * 0) / len(liste)
    return f


class Agent:
    def __init__(self, id):
        self.id = id
        self.tenir = 0
        self.memoire = []
        self.pprise = -1
        self.pdepot = -1
        self.change = 0  # Pris ou déposé
        self.pas = 1
        self.feromone = -1
        self.debut_diffusion = 0
        self.fin_diffusion = 5
        self.appel = 0

    def prob(self, actual, memoire, tenir):
        pprise = -1
        pdepot = -1
        if actual == 1:
            if tenir == 0:
                f1 = prop_Agents(memoire, 1)
                pprise = (0.1 / (0.1 + f1)) ** 2
                return pprise
            else:
                return pprise
        elif actual == 2:
            if tenir == 0:
                f2 = prop_Agents(memoire, 2)
                pprise = (0.1 / (0.1 + f2)) ** 2
                return pprise
            else:
                return pprise
        elif actual == 3:
            if tenir == 0:
                f3 = prop_Agents(memoire, 3)
                pprise = (0.1 / (0.1 + f3)) ** 2
                return pprise
            else:
                return pprise
        else:  # actual != 0.9 and actual != 0.8 and actual != 0.7:
            if tenir == 0:
                return pdepot
            elif tenir == 1:
                f1 = prop_Agents(memoire, 1)
                pdepot = (f1 / (0.3 + f1)) ** 2
                return pdepot
            elif tenir == 2:
                f2 = prop_Agents(memoire, 2)
                pdepot = (f2 / (0.3 + f2)) ** 2
                return pdepot
            else:
                f3 = prop_Agents(memoire, 3)
                pdepot = (f3 / (0.3 + f3)) ** 2
                return pdepot

    def perception_action(self, actual, pos, taille_grille, feromoneAutour, pos_accessible, robot):
        self.memoire = update_memoire(actual, self.memoire, self.appel)

        if self.tenir == 0:
            # Si il appel à l'aide, il continue si ça fait moins de 10 tours
            if self.appel > 0 and self.appel < 3:
                self.appel = self.appel + 1
                # print("JATEND")
                return False, False, False, pos, -1  # En attente, Collaboration acceptée ou non, position, si on prend ou pas
            if self.appel >= 3:
                self.appel = 0
                return False, False, True, pos_accessible[randint(0, len(pos_accessible) - 1)], -1

            # Si la case sur laquelle on se trouve vaut 3, on appelle à l'aide si aucun autre robot n'appelle à l'aide
            elif actual == 3 and robot == False:
                self.appel = 1
                return True, False, False, pos, -1

            # Si un robot appelle déjà à l'aide sur la case, il est sur le point de fusionner
            elif robot == True:
                self.tenir = 3
                return False, True, False, pos_accessible[randint(0, len(pos_accessible) - 1)], -2

            # On prend un objet avec une probabilité pprise
            prend = -1
            if actual != 0:
                pprise = self.prob(actual, self.memoire, self.tenir)
                r = random.uniform(0, 1)

                if r < pprise:
                    self.tenir = actual
                    prend = -2

            # Sinon on se déplace vers la phéromone la plus importante

            litse_pheromone = feromoneAutour
            pheromone_max = max(litse_pheromone)
            while pheromone_max != 0 :
                # Creation de la liste contenant les index avec le pheromone max
                liste = []
                for k in range(len(feromoneAutour)):
                    if feromoneAutour[k] == pheromone_max:
                        liste.append(k)
                del litse_pheromone[litse_pheromone.index(pheromone_max)]
                r = random.uniform(0, 1)
                if r < pheromone_max:
                    ra = randint(0, len(liste) - 1)
                    return False, False, False, pos_accessible[liste[ra]], prend
                if len(litse_pheromone) !=0 :
                    pheromone_max = max(litse_pheromone)
                else :
                    pheromone_max = 0
                # On crée une liste contenant les index

            return False, False, False, pos_accessible[randint(0, len(pos_accessible) - 1)], prend
        else:
            depot = -1
            pdepot = self.prob(actual, self.memoire, self.tenir)
            if actual == 0:
                r = random.uniform(0, 1)
                if r < pdepot:
                    depot = self.tenir
                    self.tenir = 0

            return False, False, False, pos_accessible[randint(0, len(pos_accessible) - 1)], depot
