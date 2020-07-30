"""
Gradient descent implementation in 2d
"""

import math
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

# Creating data tests using gaussian potential (512x512 grid)
potential = np.zeros((1024, 1024), float)

minima = [[100, 500], [900, 500]]  # list of gaussian minima
maxima = [[100, 250], [700, 200], [400, 400], [700, 500], [700, 700], [300, 650]]  # list of gaussian maxima
eff_r = 30

# Gradient_Descent_functions

sigma_x = 2.35482 * eff_r
sigma_y = 2.35482 * eff_r
depth = 10000000.0 / (2 * math.pi * sigma_x ** 2)

"""
Parameters
"""
spr_k = 1
ts = 0.1
iterate = 3000
image_n = 30
mass = 1

"""
Gauss function is the potential energy function used
In this case it is read from a file and transferred into [data] array
[coord] array is the energy of a point in [data] array
In 2d case [coord] is an array [x,y]
If using gaussian potential, set this as a function that take input of point [x,y] and produce a value of energy for that point
"""


def gauss(coord):  # Gauss functions
    gaussoid = 0
    for i in range(len(minima)):
        expo = 0
        for k in range(len(coord)):
            expo += ((coord[k] - minima[i][k]) ** 2) / (2 * sigma_x ** 2)
        gaussoid += -1 * depth * np.exp(-1 * (expo))
    for j in range(len(maxima)):
        expo = 0
        for k in range(len(coord)):
            expo += ((coord[k] - maxima[j][k]) ** 2) / (2 * sigma_x ** 2)
        gaussoid += 1 * depth * np.exp(-1 * (expo))
    return gaussoid


"""
This is a simplistic calculation for the numerical derivative
$grad_gauss function calculates the derivative by varying the input a little
using centralised difference
n1 is the index of varied variables

for a coord [x,y] $grad_gauss([x,y],0) gives the derivative with respect to
variable x
"""


def grad_gauss(coord, n1):
    h = 0.000001
    der = []
    conf_up = deepcopy(coord)
    conf_down = deepcopy(coord)
    conf_up[n1] += 0.5 * h
    conf_down[n1] -= 0.5 * h
    return (gauss(conf_up) - gauss(conf_down)) / h


"""
This is just a function to fill the array potential for plotting
"""
# prepare_data
for i in range(len(potential)):
    for j in range(len(potential[0])):
        potential[j, i] = gauss([i, j])

# 2d Gradient Descent procedures
"""
This is a gradient descent function to find the closest local
minima with a given starting guess_coord array [x,y]
iteration is maximum number of iteration test done if it does not converge
based on accuracy value
step_size is initial guess size (which will be updated based on first derivative and second derivative)
"""

iteration = 1000000
step_size = 1
accuracy = 1e-4


def find_minima(guess_coord, iteration, step_size, accuracy):
    flag = False
    init_guess = deepcopy(guess_coord)
    for i in range(iteration):
        mem_guess = deepcopy(init_guess)
        dist = 0
        for j in range(len(init_guess)):
            init_guess[j] -= step_size * grad_gauss(init_guess, j)
            dist += (init_guess[j] - mem_guess[j]) ** 2
        if math.sqrt(dist) <= accuracy:
            flag = True
            break
        dx = 0
        ddx = 0
        for j in range(len(init_guess)):
            dx += ((init_guess[j] - mem_guess[j]) * (grad_gauss(init_guess, j) - grad_gauss(mem_guess, j)))
            ddx += ((grad_gauss(init_guess, j) - grad_gauss(mem_guess, j)) * (
            (grad_gauss(init_guess, j) - grad_gauss(mem_guess, j))))
        if ddx != 0:
            step_size = min(abs(dx / ddx), 10)
        else:
            step_size = 1
    if flag == True:
        return init_guess
    else:
        return print("doesn't converge to a minima")


"""
Reverse of the gradient descent, needed to find the multipaths 
"""


def find_maxima(guess_coord, iteration, step_size, accuracy):
    flag = False
    init_guess = deepcopy(guess_coord)
    for i in range(iteration):
        mem_guess = deepcopy(init_guess)
        dist = 0
        for j in range(len(init_guess)):
            init_guess[j] += step_size * grad_gauss(init_guess, j)
            dist += (init_guess[j] - mem_guess[j]) ** 2
        if math.sqrt(dist) <= accuracy:
            flag = True
            break
        dx = 0
        ddx = 0
        for j in range(len(init_guess)):
            dx += ((init_guess[j] - mem_guess[j]) * (grad_gauss(init_guess, j) - grad_gauss(mem_guess, j)))
            ddx += ((grad_gauss(init_guess, j) - grad_gauss(mem_guess, j)) * (
            (grad_gauss(init_guess, j) - grad_gauss(mem_guess, j))))
        if ddx != 0:
            step_size = min(abs(dx / ddx), 10)
        else:
            step_size = 1
    if flag == True:
        return init_guess
    else:
        return print("doesn't converge to a maxima")


# NEB Procedure

"""
Simple distance calculator between two points
"""


def distance(pA, pB):
    dist = 0
    for i in range(len(pA)):
        dist += (pA[i] - pB[i]) ** 2
    dist = math.sqrt(dist)
    return dist


"""
This is a simple pathway maker to create initial path for NEB for a set of two minima
conforma, and conformb are array [x,y,z,...] representing the two minima
image is the number of interval between minima
image=20 will produce 20 interval (21 images counting both minima)
"""


def image_spacer(conforma, conformb, image):  # linear spacer
    pathway = np.zeros((image + 1, len(conforma)))
    for n in range(image + 1):
        for i in range(len(conforma)):
            pathway[n, i] = (conforma[i] + n * (conformb[i] - conforma[i]) / image)
    return pathway


"""
This is the tangent of specific image in the pathway
[pathway] is the array representing the whole pathway at the moment
n1 is the index of a specific image in the pathway
"""


def tangent(n1, pathway):
    updiff = np.zeros((len(pathway[n1])), 'float')
    uppot = np.zeros((len(pathway[n1])), 'float')
    downdiff = np.zeros((len(pathway[n1])), 'float')
    downpot = np.zeros((len(pathway[n1])), 'float')
    for i in range(len(pathway[n1])):
        downdiff[i] = pathway[n1][i] - pathway[n1 - 1][i]
        updiff[i] = pathway[n1 + 1][i] - pathway[n1][i]
    if gauss(pathway[n1 + 1]) > gauss(pathway[n1]) and gauss(pathway[n1]) > gauss(pathway[n1 - 1]):
        dist = 0
        for m in range(len(updiff)):
            dist += updiff[m] * updiff[m]
        for l in range(len(updiff)):
            updiff[l] = updiff[l] / math.sqrt(dist)
        return updiff
    elif gauss(pathway[n1 + 1]) < gauss(pathway[n1]) and gauss(pathway[n1]) < gauss(pathway[n1 - 1]):
        dist = 0
        for m in range(len(downdiff)):
            dist += downdiff[m] * downdiff[m]
        for l in range(len(downdiff)):
            downdiff[l] = downdiff[l] / math.sqrt(dist)
        return downdiff
    else:
        difftan = gauss(pathway[n1 + 1]) - gauss(pathway[n1 - 1])
        diffright = abs(gauss(pathway[n1 + 1]) - gauss(pathway[n1]))
        diffleft = abs(gauss(pathway[n1 - 1]) - gauss(pathway[n1]))
        if difftan >= 0:
            dist = 0
            # for l in range(len(updiff)):
            #    updiff[l]=updiff[l]*max(diffright,diffleft)+downdiff[l]*min(diffright,diffleft)
            for m in range(len(updiff)):
                dist += updiff[m] * updiff[m]
            for l in range(len(updiff)):
                updiff[l] = updiff[l] / math.sqrt(dist)
            return updiff
        if difftan < 0:
            dist = 0
            # for l in range(len(updiff)):
            #    updiff[l]=updiff[l]*min(diffright,diffleft)+downdiff[l]*max(diffright,diffleft)
            dist = 0
            for m in range(len(downdiff)):
                dist += downdiff[m] * downdiff[m]
            for l in range(len(downdiff)):
                downdiff[l] = downdiff[l] / math.sqrt(dist)
            return downdiff


"""
Initialising the pathway to be minimised using NEB
"""

guess1 = [110, 510]
guess2 = [910, 510]

minima1 = find_minima(guess1, 10000, 10, 1e-8)
minima2 = find_minima(guess2, 10000, 10, 1e-8)
pathway1 = image_spacer(minima1, minima2, image_n)

grad_tan = np.zeros((len(pathway1), len(pathway1[0])), 'float')
grad_perp = np.zeros((len(pathway1), len(pathway1[0])), 'float')
spring_f = np.zeros((len(pathway1), len(pathway1[0])), 'float')
grad_unit = []
spring_perp = np.zeros((len(pathway1), len(pathway1[0])), 'float')
force_n = np.zeros((len(pathway1), len(pathway1[0]),), 'float')
pathmem = []
pathtemp = []

forcerms = []

"""
The function to call to minimise the pathway
The input is a path_route array
for example in 2d case with 3 image
path_route=[[x1,y1],[x2,y2],[x3,y3]]
"""


def neb(path_route):
    pathway = deepcopy(path_route)

    # calculating grad for pathway, calculating g//, g+
    # grad_tan=np.zeros((len(pathway),len(pathway[0])),'float')
    # grad_perp=np.zeros((len(pathway),len(pathway[0])),'float')
    # spring_f=np.zeros((len(pathway),len(pathway[0])),'float')
    # grad_unit=[]
    # spring_perp=np.zeros((len(pathway),len(pathway[0])),'float')
    # force_n=np.zeros((len(pathway),len(pathway[0]),),'float')
    # pathmem=[]
    # pathtemp=[]

    vvel = []

    # verlet starter

    # Calculate Gradient at each point
    # tangent component of true gradient
    for i in range(1, len(pathway) - 1):
        for j in range(len(pathway[i])):
            grad_tan[i, j] = grad_gauss(pathway[i], j) * tangent(i, pathway)[j]
        tans = 0
        for j in range(len(pathway[i])):
            tans += grad_tan[i, j]
        for j in range(len(pathway[i])):
            grad_tan[i, j] = tans * tangent(i, pathway)[j]

        for j in range(len(pathway[i])):
            grad_perp[i, j] = grad_gauss(pathway[i], j) - grad_tan[i, j]

        upspr = np.zeros((len(pathway[i])), 'float')
        dwspr = np.zeros((len(pathway[i])), 'float')
        for j in range(len(pathway[i])):
            upspr[j] = pathway[i + 1, j] - pathway[i, j]
            dwspr[j] = pathway[i, j] - pathway[i - 1, j]
        spruceup = 0
        sprucedw = 0
        for j in range(len(pathway[i])):
            spruceup += upspr[j] * upspr[j]
            sprucedw += dwspr[j] * dwspr[j]
        for j in range(len(pathway[i])):
            spring_f[i, j] = spr_k * (math.sqrt(spruceup) - math.sqrt(sprucedw)) * tangent(i, pathway)[j]

    # Quenched_Verlet_half_step
    for i in range(1, len(pathway) - 1):
        for j in range(len(pathway[i])):
            force_n[i, j] = spring_f[i, j] - grad_perp[i, j]
    pathmem = deepcopy(pathway)
    # update position
    for i in range(1, len(pathway) - 1):
        for j in range(len(pathway[i])):
            pathway[i, j] += ts * (ts / (2 * mass)) * (force_n[i, j])

    for time in range(0, iterate, 1):
        # updating forces (gradient)
        # tangent component of true gradient
        for i in range(1, len(pathway) - 1):
            for j in range(len(pathway[i])):
                grad_tan[i, j] = grad_gauss(pathway[i], j) * tangent(i, pathway)[j]
            tans = 0
            for j in range(len(pathway[i])):
                tans += grad_tan[i, j]
            for j in range(len(pathway[i])):
                grad_tan[i, j] = tans * tangent(i, pathway)[j]

            for j in range(len(pathway[i])):
                grad_perp[i, j] = grad_gauss(pathway[i], j) - grad_tan[i, j]

            upspr = np.zeros((len(pathway[i])), 'float')
            dwspr = np.zeros((len(pathway[i])), 'float')
            for j in range(len(pathway[i])):
                upspr[j] = pathway[i + 1, j] - pathway[i, j]
                dwspr[j] = pathway[i, j] - pathway[i - 1, j]
            spruceup = 0
            sprucedw = 0
            for j in range(len(pathway[i])):
                spruceup += upspr[j] * upspr[j]
                sprucedw += dwspr[j] * dwspr[j]
            for j in range(len(pathway[i])):
                spring_f[i, j] = spr_k * (math.sqrt(spruceup) - math.sqrt(sprucedw)) * tangent(i, pathway)[j]

        # force-calc and moving step

        for i in range(1, len(pathway) - 1):
            for j in range(len(pathway[i])):
                force_n[i, j] = spring_f[i, j] - grad_perp[i, j]

        for i in range(1, len(pathway) - 1):
            pathtemp = deepcopy(pathway)
            for j in range(len(pathway[i])):
                pathway[i, j] = 2 * pathway[i, j] - pathmem[i, j] + (force_n[i, j]) * ts / mass
            pathmem = deepcopy(pathtemp)
        forcet = 0
        for i in range(len(force_n)):
            for j in range(len(force_n[i])):
                forcet += force_n[i, j] * force_n[i, j]

        forcerms.append(math.sqrt(forcet) / len(pathway))

        if time > 10:
            breaker = forcerms[-20:]
            if abs(sum(breaker) / len(breaker)) < 0.00001:
                break

    return pathway


"""
Saving the path $path_name into a text file $text_name
"""


def path_totxt(path_name, text_name):
    with open(text_name, 'w') as f:
        for i in range(len(path_name)):
            f.write(str(path_name[i][0]) + ' ' + str(path_name[i][1]) + ' \n')


"""
Plotting the path $path_coord into a .png file named $base_name with text title $title
and also save the coordinate of the paths into .txt file
"""


def subplt(path_coord, image_title):
    images = len(path_coord) - 1
    fig = plt.figure(figsize=(8, 16))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    ax0 = plt.subplot(gs[0])

    ax0.margins(x=180, y=180)
    img = ax0.imshow(potential, origin="lower")
    fig.colorbar(img, orientation='vertical', use_gridspec=True)
    nimage = int(images / 2)
    if path_coord[nimage][0] > 0 and path_coord[nimage][0] < 1024:
        if path_coord[nimage][1] > 0 and path_coord[nimage][1] < 1024:
            for j in range(images):
                plt.arrow(path_coord[j][0], path_coord[j][1], (path_coord[j + 1][0] - path_coord[j][0]),
                          (path_coord[j + 1][1] - path_coord[j][1]), color='r')
                if j % 5 == 0:
                    plt.plot(path_coord[j][0], path_coord[j][1], '--o', color='b')
                elif j == (images - 1):
                    plt.plot(path_coord[j][0], path_coord[j][1], '--o', color='b')
    # for i in range(len(max_coord)):
    #    plt.plot(max_coord[i][0],max_coord[i][1],'ko')
    x = np.arange(-100, 1024, 1)
    y = np.arange(-100, 1024, 1)
    Z = potential
    X, Y = np.meshgrid(x, y)
    plt.xlim(-100, 1024)
    plt.ylim(-100, 1024)
    plt.contour(Z, 10)
    plt.title(image_title)
    ax1 = plt.subplot(gs[1])
    # plt.subplot(2,1,2)
    dist_axis = 0
    for j in range(images):
        clip_color = 'r'
        if j % 5 == 0:
            clip_color = 'b'
        plt.arrow(dist_axis, gauss(path_coord[j]), distance(path_coord[j], path_coord[j + 1]),
                  (gauss(path_coord[j + 1]) - gauss(path_coord[j])), color='r')
        plt.plot(dist_axis, gauss(path_coord[j]), '--o', linestyle='dashed', color=clip_color)
        dist_axis += distance(path_coord[j], path_coord[j + 1])
    # fig.tight_layout()
    fig.savefig(image_title + '.png', dpi=200)
    path_totxt(path_coord, image_title + '.txt')
    plt.close('all')
    # plt.show()


def unique_max(max_list):
    temp_list = [[0, 0]]
    for i in range(len(max_list)):
        flag = False
        for j in range(len(temp_list)):
            adds = 0
            for k in range(len(max_list[0])):
                adds += abs(max_list[i][k] - temp_list[j][k])
            if adds < 0.1:
                break
            if j == (len(temp_list) - 1):
                flag = True
        if flag:
            temp_list.append([max_list[i][0], max_list[i][1]])
    temp_list = temp_list[2:-1]
    return temp_list


def maxima_diff(maxlist1, maxlist2):
    temp_list = []
    for i in range(len(maxlist1)):
        flag = True
        for j in range(len(maxlist2)):
            adds = 0
            for k in range(len(maxlist2[0])):
                adds += abs(maxlist1[i][k] - maxlist2[j][k])
            if adds < 0.1:
                flag = False
        if flag:
            temp_list.append([maxlist1[i][0], maxlist1[i][1]])
    return temp_list


def plot_path(path_coord):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.margins(x=180, y=180)
    img = ax.imshow(potential, origin="lower")
    fig.colorbar(img, orientation='vertical', fraction=.1)
    for i in range(len(path_coord)):
        plt.plot(path_coord[i][0], path_coord[i][1], 'ko')

    x = np.arange(-100, 1024, 1)
    y = np.arange(-100, 1024, 1)
    Z = potential
    X, Y = np.meshgrid(x, y)
    plt.xlim(-100, 1024)
    plt.ylim(-100, 1024)
    plt.contour(Z, 10)
    plt.title('plot_test')

    plt.show()


def plot_frms(frms):
    for i in range(0, len(frms), 10):
        plt.plot(i, frms[i], 'o-')
    plt.xlabel('step')
    plt.ylabel('Force_RMS')
    plt.ylim(-0.1, 1)
    plt.title('pathwise Force RMS')

    plt.show()


"""
A function used for multipath calculation
This function takes a pathway, and a given maxima, then create an initial path
that goes from one minima in the pathway to the other minima in the pathway but
passing around the given maxima
Used for multipaths search
"""


def maxima_refine(path_coord, maxima_coord):
    mdist = deepcopy(maxima_coord)
    path_mean = np.mean(path_coord, axis=0)
    mdist[1] += 0.5 * (mdist[1] - path_mean[1])
    print(str(mdist[0]) + ' ' + str(mdist[1]))
    ldist = 0
    rdist = 0
    for i in range(len(mdist)):
        ldist += (mdist[i] - path_coord[0][i]) ** 2
        rdist += (mdist[i] - path_coord[-1][i]) ** 2
    ldist = math.sqrt(ldist)
    rdist = math.sqrt(rdist)
    ldist = math.ceil(ldist * len(path_coord) / (ldist + rdist))
    rdist = int(len(path_coord) - ldist) - 1
    first_half = image_spacer(path_coord[0], mdist, ldist)
    second_half = image_spacer(mdist, path_coord[-1], rdist)
    return np.concatenate((first_half[:-1], second_half))


# call the initialised path (pathway1)
# Store and plot the result using txt title 'result'
pathway = neb(pathway1)
subplt(pathway, 'result')

"""
#Only activate if needing multipaths search
#maxima nudging
max_recalc_plus = deepcopy(pathway)
max_recalc_minus = deepcopy(pathway)

for i in range(1,len(pathway)-1):
    max_recalc_plus[i,1]+=random.uniform(-10,10)
    #max_recalc_minus[i,1]-=random.uniform(5,20)

for i in range(1,len(pathway)-1):
    max_recalc_plus[i]=find_maxima(max_recalc_plus[i],10000,1,1e-8)
    #max_recalc_minus[i]=find_maxima(max_recalc_minus[i],10000,1,1e-8)

unique_m_plus = unique_max(max_recalc_plus)
#unique_m_minus = unique_max(max_recalc_minus)

multipaths = deepcopy(pathway)

for i in range(len(unique_m_plus)):
    new_path = maxima_refine(pathway,unique_m_plus[i])
    calc_new = neb(new_path)
    multipaths = np.concatenate((multipaths,calc_new))
    
         
        
    
subplt(multipaths,'multis')
"""
