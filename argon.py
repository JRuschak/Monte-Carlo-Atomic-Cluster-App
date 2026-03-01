import random, copy, math
import matplotlib.pyplot as plt



class argon:

    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z

def initialize_cluster(numatoms):
    atomlist = []
    for i in range(numatoms):
        temp_atom = argon(random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1))
        atomlist.append(temp_atom)
    return atomlist

def translational_move(atomlist,t,stepsize):
    for i in range(len(atomlist)):
        original_coords = atomlist[i]
        original_energy = calc_oneatom_energy(i,atomlist,original_coords)
        new_coords = argon(original_coords.x+stepsize*random.uniform(-1,1),original_coords.y+stepsize*random.uniform(-1,1),original_coords.z+stepsize*random.uniform(-1,1))
        new_energy = calc_oneatom_energy(i,atomlist,new_coords)
        delta = new_energy - original_energy
        if delta < 0:
            atomlist[i] = new_coords
        elif random.random() < math.e**(-delta/t):
            atomlist[i] = new_coords
    return atomlist

def COM(atomlist):
    xavg = 0
    yavg = 0
    zavg = 0
    length = len(atomlist)
    for i in range(length):
        xavg+=atomlist[i].x/length
        yavg+=atomlist[i].y/length
        zavg+=atomlist[i].z/length
    for i in range(length):
        atomlist[i].x -= xavg
        atomlist[i].y -= yavg
        atomlist[i].z -= zavg


def calc_energy(atomlist):
    atomlist2 = copy.deepcopy(atomlist) # used for the j loop 
    total_energy = 0
    for i in range(len(atomlist)):
        atomlist2.pop(0) #removes the first of the array every loop to prevent comparing the same atom
        for j in range(len(atomlist2)):
            r = calc_dist(atomlist[i],atomlist2[j])
            single_energy = calc_single_energy(r,121,3.4)
            total_energy += single_energy
    return total_energy


def calc_dist(atom1,atom2):
    distance = ((atom1.x-atom2.x)**2+(atom1.y-atom2.y)**2+(atom1.z-atom2.z)**2)**(1/2)
    return distance

def calc_single_energy(r,epsilon,sigma):
    sr6 = (sigma/r)**6
    return 4*epsilon*((sr6)**2-sr6)

def calc_oneatom_energy(singleindex,atomlist,atom):
    atomlist2 = copy.deepcopy(atomlist)
    atomlist2.pop(singleindex)
    sum_energy = 0
    for i in range(len(atomlist2)):
        r = calc_dist(atom,atomlist2[i])
        sum_energy += calc_single_energy(r,121,3.4)
    return sum_energy
    
def print_coords(atomlist):
    for i in range(len(atomlist)):
        print(f'x:{atomlist[i].x},y:{atomlist[i].y},z:{atomlist[i].z}')

def graph(atomlist):
    x=[]
    y=[]
    z=[]
    for i in range(len(atomlist)):
        x.append(atomlist[i].x)
        y.append(atomlist[i].y)
        z.append(atomlist[i].z)
    return [x,y,z]
