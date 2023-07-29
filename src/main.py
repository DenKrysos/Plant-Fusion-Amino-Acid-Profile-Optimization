#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

'''
Created on 2023-06-30

@author: Dennis Krummacker

What it does:
    It finds the optimal proportional combination of plant-based proteins to resemble various reference Amino Acid Profile as close as possible.

Involved Mathematics:
    Vector-Fitting Optimization/Minimzation Technique (Least Squares Regression) together with Linear Algebra.

Didactic Note:
    Just in case some novice programmer looks through this script:
    I did some stuff, considered unclean. Mainly all the globals.
    That works only because everything here is nicely sequential, without any concurrency whatsoever.
    As soon as any kind of parallelism or whatever source of concurrent access is introduced, this would have to be substantially reconsidered.
    And just in general: As soon as a project grows bigger than this little toy script here or you deal with multiple src-Files, chances are high, global variables give you headaches, to say the least.
    I beg your pardon, for hacking this script quick 'n with a lack of elegancy from Software-Development's point-of-view. ;oP
'''

import sys
import os
import json
import shutil

from itertools import product
from functools import reduce
import operator

import numpy as np
from scipy.optimize import minimize, basinhopping
import matplotlib.pyplot as plt
import matplotlib.patches as patches




################################################################
# Versioning


def set_Version_Description():
    # VERSION_DevStage="beta"
    # VERSION_ReleaseTrack="Preview"
    # VERSION_ReleaseTrackMAJOR=1
    VERSION_MAJOR=1
    VERSION_MINOR=0
    VERSION_PATCH=0
    VERSION_TWEAK=0
    if 0<VERSION_TWEAK:
        VERSION_NUMBER="%s.%s.%s_%s"%(VERSION_MAJOR,VERSION_MINOR,VERSION_PATCH,VERSION_TWEAK)
    else:
        VERSION_NUMBER="%s.%s.%s"%(VERSION_MAJOR,VERSION_MINOR,VERSION_PATCH)
    
    VERSION_DESCRIPTION="%s"%(VERSION_NUMBER)
    
    try: VERSION_DevStage
    except NameError: VERSION_DevStage=None
    if not VERSION_DevStage is None and 0<len(VERSION_DevStage):
        VERSION_DESCRIPTION="%s-%s"%(VERSION_DevStage,VERSION_DESCRIPTION)
    
    try: VERSION_ReleaseTrack
    except NameError: VERSION_ReleaseTrack=None
    try: VERSION_ReleaseTrackMAJOR
    except NameError: VERSION_ReleaseTrackMAJOR=None
    if not VERSION_ReleaseTrack is None and 0<len(VERSION_ReleaseTrack) and not VERSION_ReleaseTrackMAJOR is None and 0<VERSION_ReleaseTrackMAJOR:
        VERSION_ReleaseTRACKDescription=f"{VERSION_ReleaseTrack}-{VERSION_ReleaseTrackMAJOR}"
    else:
        VERSION_ReleaseTRACKDescription=None
    
    if not VERSION_ReleaseTRACKDescription is None:
        VERSION_DESCRIPTION=f"{VERSION_DESCRIPTION}:{VERSION_ReleaseTRACKDescription}"
    return VERSION_DESCRIPTION


################################################################
# Just some Python Gimmicks

properDirHierarchy=False

srcSubDir="src"
outSubDir="out"
outSubDirLatest="latest"
plotSubDir="plots"
plotInputAnalSubDir="InputAnalysis"

progPath=None
progPathExe=None
progPathOut=None
progPathOutLatest=None
progPathOutLatestPlot=None
progPathOutLatestPlotInAnal=None
progPathOutIngSeq=None
progPathOutIngSeqPlot=None
progPathOutIngSeqPlotInAnal=None

def set_ProgramPath(mainFile__file__):
    global properDirHierarchy
    global progPath
    global progPathExe
    global progPathOut
    global progPathOutLatest
    global progPathOutLatestPlot
    global progPathOutLatestPlotInAnal
    global srcSubDir
    global outSubDir
    global outSubDirLatest
    global plotSubDir
    global plotInputAnalSubDir
    progPath=os.path.realpath(mainFile__file__)
    progPath=os.path.dirname(progPath)
    head,tail=os.path.split(progPath)
    if not tail==srcSubDir:
        properDirHierarchy=False
        return
    else:
        properDirHierarchy=True
        progPathExe=progPath
        progPath=head
        progPathOut=os.path.join(progPath,outSubDir)
        progPathOutLatest=os.path.join(progPathOut,outSubDirLatest)
        progPathOutLatestPlot=os.path.join(progPathOutLatest,plotSubDir)
        progPathOutLatestPlotInAnal=os.path.join(progPathOutLatestPlot,plotInputAnalSubDir)

def set_output_paths(enabledIngredients):
    global properDirHierarchy
    if not properDirHierarchy:
        return
    global progPath
    global progPathOut
    global progPathOutIngSeq
    global progPathOutIngSeqPlot
    global progPathOutIngSeqPlotInAnal
    global plotSubDir
    global plotInputAnalSubDir
    enabSubDir=""
    for enabled in enabledIngredients:
        if enabled:
            enabSubDir+="1"
        else:
            enabSubDir+="0"
    progPathOutIngSeq=os.path.join(progPathOut,enabSubDir)
    progPathOutIngSeqPlot=os.path.join(progPathOutIngSeq,plotSubDir)
    progPathOutIngSeqPlot=os.path.join(progPathOutIngSeqPlot,plotInputAnalSubDir)


def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)
def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

#Create all missing subdirectories
def _assure_Dirs_exist(fullPath):
    os.makedirs(fullPath, exist_ok=True)
#Only creates the last directory
def _assure_Dir_exists(fullPath):
    try:
        os.mkdir(fullPath)
    except FileExistsError:
        pass
    except OSError:
        print("Creation of the Directory failed: \"%s\"."%fullPath)
    except:
        print("Creation of the Directory failed: \"%s\"."%fullPath)
    #else:
        #print("Successfully created the directory %s " % path)

def _del_dir_recursive(fullPath):
    try:
        shutil.rmtree(fullPath)
    except FileNotFoundError:
        pass
    except:
        print("Deletion of the Directory failed: \"%s\"."%fullPath)

def _cpy_dir_recursive(dstDir,srcDir):
    shutil.copytree(srcDir,dstDir)

def _create_outDir(subDir):
    oPath=os.path.join(progPath,subDir)
    _assure_Dir_exists(oPath)
    return oPath

#write and create
def wrcr_file(path,fName):
    _assure_Dir_exists(path)
    file_full=os.path.join(path,fName)
    try:
        f=open(file_full, mode='w', encoding = 'utf-8')
    except FileNotFoundError:
        print("Creation of File failed: \"%s\"" % file_full)
        return
    except:
        print("Creation of File failed: \"%s\"" % file_full)
        return
    #finally:
        #f.close()
    return f

def clean_resultLatest():
    _del_dir_recursive(progPathOutLatest)
    return

def dump_resultOut():
    global save_result
    global properDirHierarchy
    if not save_result:
        return
    elif not properDirHierarchy:
        printDirectoryHierarchyNote()
    global resOut
    global progPathOutLatest
    fName_terminal_std="Output_Legend_stdOut.txt"
    fName_terminal_result="Output_Results_perProfile.txt"
    fName_json="In_Out_Data.json"
    #
    # fPath=_create_outDir(outSubDir)
    _assure_Dirs_exist(progPathOutLatest)
    #
    file=wrcr_file(progPathOutLatest,fName_terminal_std)
    file.write(resOut['Terminal']['Std'])
    file.close()
    #
    file=wrcr_file(progPathOutLatest,fName_terminal_result)
    for profResult in resOut['Terminal']['Result']:
        file.write(profResult)
    file.close()
    #
    file=wrcr_file(progPathOutLatest,fName_json)
    json.dump(resOut['json'],file,sort_keys=False,indent=2)
    file.close()

def save_plot(fPath,fName,fig):
    global save_result
    if not save_result:
        return
    _assure_Dirs_exist(fPath)
    fStore=os.path.join(fPath,f'{fName}.pdf')
    printOut_std(f"Stored Plot under {fStore}.")
    fig.savefig(fStore)

# Generally, results are stored to 'Latest'. This copies files over to the Ingredient-Cfg-Specific Dir.
def duplicate_Out_fromLatest():
    global save_result
    if not save_result:
        return
    global progPathOutLatest
    global progPathOutIngSeq
    _del_dir_recursive(progPathOutIngSeq)
    _cpy_dir_recursive(progPathOutIngSeq,progPathOutLatest)

def printOut_std(outStr,end='\n'):
    global resOut
    resOut['Terminal']['Std']+=outStr+end
    print(outStr,end=end)
def printOut_res(outStr,end='\n'):
    global resOut
    resOut['Terminal']['Result'][resOut['curProfileIdx']]+=outStr+end
    print(outStr,end=end)


def set_json(key,val):#key as list
    setInDict(resOut,key,val)

def set_json_perProf(key,val):#key as list
    fullKeys=['json','Out',resOut['curProfile']]+key
    setInDict(resOut,fullKeys,val)
    #resOut['json']['Out'][resOut['curProfile']][key]=val


def distribute_capacity(capacity, num_entries, fraction_size=None):
    """
    It is a Generator Function (that operates recursively).
    It generates every possible Permutation for Distributing a given 'Capacity' in Chunks of 'fraction_size' over a list of length 'num_entries'.
    With each iteration over the Generator you will get the next Permutation.
    So this is gentle on memory consumption as it does not generate all permutations at once, but only ever returns the next.
    - If no 'fraction_size' is passed, this is internally set to evenly split the 'capacity' in 'num_entries' pieces.
      - In many use-cases, this might make the most sense, because with every fraction-size that has no multiple to equal capacity, always an unused headroom is left
    - But nonetheless, any 'fraction_size' can be freely passed explicitly
    Usage-Example:
        capacity=1
        num_entries=4
        permutations_generator = distribute_capacity(capacity, num_entries, fraction_size)
        for permutation in permutations_generator:
            print(permutation)
        print("\nEnd")
    """
    if fraction_size is None:
        fraction_size=capacity/num_entries
    #Have to normalize step_size and capacity for the loop.
    loop_lim=int(capacity/fraction_size)
    if 0>=num_entries:
        yield []
    else:
        for i in range(loop_lim,-1,-1):
            sublist=distribute_capacity(capacity-i*fraction_size,num_entries-1,fraction_size)
            for permutation in sublist:
                yield [i*fraction_size]+permutation

def num_permutations(capacity, num_entries):
    num=0
    permutations_generator = distribute_capacity(capacity, num_entries)
    for permutation in permutations_generator:
        num+=1
    return num



################################################################
# Some Set-Up for the actual Script

resOut={
    'curProfile':"",
    'curProfileIdx':0,
    'Terminal':{
        'Std':"",
        'Result':[],
    },
    'json':{
        'In':{
            'Target-Profiles':{},
            'Ingredient-Candidates':{}
        },
        'In-Analysis':{
        },
        'Out':{}
    }
}

AAindices=[
    "Leucine*‡ (Leu, L)",
    "Isoleucine*‡ (Ile, I)",
    "Valine*‡ (Val, V)",
    "Lysine* (Lys, K)",
    "Threonine* (Thr, T)",
    "Tryptophan* (Trp, W)",
    "Methionine*+Cysteine† (Met+Cys, M+C)",
    "Phenylalanine*+Tyrosine† (Phe+Tyr, F+Y)",
    "Glutamine† (Gln, Q)",
    "Arginine† (Arg, R)",
    "Histidine (His, H)",
    "Remaining Non-Essentials"
]
AAindices_abbr=[
    "Leu",
    "Ile",
    "Val",
    "Lys",
    "Thr",
    "Trp",
    "M+C",#"Met+Cys",
    "F+Y",#Phe+Tyr",
    "Gln",
    "Arg",
    "His",
    "RNE"
]
AAindices_Remaining=[
    "Alanine (Ala, A)",
    "Aspartic Acid (Asp, D)",
    "Glycine (Gly, G)",
    "Proline (Pro, P)",
    "Serine (Ser, S)",
    "Asparagine (Asn, N)",
    "Glutamic Acid (Glu, E)"
]
AAidxLegend=[
    "* EAA (Essential Amino Acid): Cannot be synthesized by the Human Body:\n\t-> Isoleucin, Leucin, Lysin, Methionin, Phenylalanin, Threonin, Tryptophan, Valin.",
    "‡ BCAA (Branched-Chain Amino Acids): Belong to EAAs. Are not metabolized by the liver, but are directly delivered to utilizing tissue (like muscles) after being reabsorbed by the gut:\n\t-> Leucin, Isoleucin, Valin.",
    "† Conditionally Essential AA: Can be synthesized by Human Body to some extend. But situations may occur, where the demand exceeds its production capacity, such as during intense physical activity, injury, illness, or stress:\n\t-> Glutamine, Arginine, Cysteine, Tyronine"
]

################################################################
# The actual Script

save_result=False
draw_Plot_Switch=True
plot_InputAnalysis=False


def persistentStorage_turnOn():
    global save_result
    global properDirHierarchy
    if properDirHierarchy:
        save_result=True
def persistentStorage_turnOff():
    global save_result
    save_result=False

def plotting_turnOn():
    global draw_Plot_Switch
    draw_Plot_Switch=True
def plotting_turnOff():
    global draw_Plot_Switch
    draw_Plot_Switch=False

def plotting_inputAnalysis_turnOn():
    global plot_InputAnalysis
    plot_InputAnalysis=True
def plotting_inputAnalysis_turnOff():
    global plot_InputAnalysis
    plot_InputAnalysis=False

def create_resultPlot(argTitle,argBar1Name,argVals1,argBar2Name,argVals2,eucNorm,cosSim,quotScore):
    bar_width=0.3
    ### build a rectangle in axes coords
    left, width = .05, .5
    bottom, height = .25, .5
    right=left+width
    top=bottom+height
    #
    ### Create an array of indices for x-axis ticks (grouped bars)
    x_indexes_group1=np.arange(len(AAindices_abbr))
    x_indexes_group2=x_indexes_group1+bar_width
    #
    #
    ### Create the figure
    ### Set the figure size
    # fig,ax=plt.subplots(figsize=(3, 2), constrained_layout=True)
    fig=plt.figure(
        # figsize=(10, 6),
        # figsize=[3,2],
        tight_layout={'pad':0}
    )
    # ax=fig.add_axes([0.08, 0.09, 0.91, 0.86])
    ax=fig.add_subplot(111)
    # axes coordinates: (0, 0) is bottom left and (1, 1) is upper right
    # p=patches.Rectangle(
    #     (left, bottom), width, height,
    #     fill=False, transform=ax.transAxes, clip_on=False
    # )
    # ax.add_patch(p)
    ### Add Data to Plot
    ax.bar(x_indexes_group1, argVals1, width=bar_width, label=f'{argBar1Name}', color='orange')
    ax.bar(x_indexes_group2, argVals2, width=bar_width, label=f'{argBar2Name}', color='blue')
    ### Config the labels for x and y axes, and chart title
    ax.set_xlabel('Amino Acids')
    ax.set_ylabel('\'g/100 g\' aka percent')
    ax.set_title(argTitle)
    ### Set the x-axis tick positions and labels
    ax.set_xticks(x_indexes_group1 + bar_width / 2, AAindices_abbr)
    ### Add a legend
    ax.legend()
    ### Add Text, showing Evaluation Data
    ax.text(
        0.03, 0.76,#0.83
        f"Euclidean-δ: {eucNorm}\nCos-Sim: {cosSim}\nQuotient-Score: {quotScore}", 
        ha='left', va='top',
        transform=ax.transAxes,
        #verticalalignment='top',
        fontsize=11, color='black',
        bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round,pad=0.5')
    )
    return fig

def plot_resultComparison(targetProfiles,targetVecs,plants,coefficientsCollection,ResultVectors,eucNorms,cosSims,quotScores):
    global progPathOutLatestPlot
    for i in range(len(targetProfiles)):
        targetName=targetProfiles[i]
        targetVec=targetVecs[i]
        cs=coefficientsCollection[i]
        vResult=ResultVectors[i]
        eucNorm=eucNorms[i]
        cosSim=cosSims[i]
        quotScore=quotScores[i]
        BlendRatios=""
        lines=1
        for i in range(len(cs)):
            c=cs[i]
            plant=plants[i]
            if not 0.0==c:
                def concatBlendName(oldName):
                    if not len(oldName)==0:
                        oldName+="+"
                    oldName+=f"{c}*{plant}"
                    return oldName
                #Ehem, yeah, the proper way to do this would actually have been to check only and precisely the last line and not the len of the entire string. But, that effort is not really worth it for this use-case here, where it's not very likely to exceed 3 or even 2 lines...
                lenTest=concatBlendName(BlendRatios)
                if (lines*50+(lines-1))<len(lenTest):
                    BlendRatios+="\n"
                    lines+=1
                BlendRatios=concatBlendName(BlendRatios)
        #
        fig=create_resultPlot('Target Profile vs. Optimized Blend',
            targetName,targetVec,
            BlendRatios,vResult,
            eucNorm,cosSim,quotScore
        )
        #
        save_plot(progPathOutLatestPlot,targetName,fig)
        ### plt.show() done outside of here, because it has to be the last thing done
    return

def plot_show():
    ### Show the plot(s) - Last thing to do, because of GUI-Loop
    plt.show(block=True)


    #(vectors.T @ coefficients)# different notation for a linear-combination
def euclideanNormDistance(vecTrgt,vecRes):
    return np.linalg.norm(vecTrgt-vecRes)

def cosineSimilarity(vecTrgt,vecRes):
    magnitude=np.linalg.norm(vecTrgt)*np.linalg.norm(vecRes)
    sim=np.dot(vecTrgt,vecRes)/magnitude
    return sim

def quotientSimilarityScore(vecTrgt,vecRes):
    simScore=0
    for i in range(len(vecTrgt)-1):# -1 to exclude the Non-Essentials
        simScore+=vecRes[i]/vecTrgt[i]
    simScore=simScore/len(vecTrgt)
    return simScore

def fit_vector(vectors,target):
    num_vec=len(vectors[0])
    #
    ### Define the objective function for optimization
    def objective_euclideanNorm(coefficients):
        linear_combination=np.dot(vectors,coefficients)
        dist=euclideanNormDistance(target,linear_combination)
        return dist
    def objective_cosineSimilarity(coefficients):
        linear_combination=np.dot(vectors,coefficients)
        ### Invert to turn the maximization into a minimization
        sim=-1.0*cosineSimilarity(target,linear_combination)
        return sim
    def objective_quotientScore(coefficients):
        linear_combination=np.dot(vectors,coefficients)
        ### Invert to maximize (towards 1 in normalized case)
        score=-1.0*quotientSimilarityScore(target,linear_combination)
        return score
    def objective(coefficients):
        return objective_euclideanNorm(coefficients)
    #
    ### Constraints
    ### ## The sum of coefficients should be 1 (equality constraint)
    def constraint_sum(coefficients):
        return np.sum(coefficients)-1.0
    ### ## Each entry of the result must be greater or equal to corresponding coordinate of target (except the NEAAs)
    ### ## ## Formulate as 'inequality constraint': These must be formulated so that the allowed values are non-negative
    def constraint_overshooting_algebra(coefficients,vectorsIn,target):
        ### Create sublist to exclude last entry
        linComb=np.dot(vectorsIn,coefficients)
        linComb=linComb[0:len(linComb)-1]
        ### Formulate it so that the 'bad values' result in negative, then normalize the positive to '0'.
        diff=linComb-target[0:len(target)-1]
        for i in range(len(diff)):
            if 0<diff[i]:
                diff[i]=0
        return np.sum(diff)
    def constraint_overshooting_loop(coefficients,vectorsIn,target):
        ### Check each entry except the last
        linComb=np.dot(vectorsIn,coefficients)
        for i in range(len(target)-1):
            if linComb[i]<target[i]:
                return target[i]-linComb[i]
        ### If all constraints are satisfied, return 0
        return 0
    def constraint_overshooting(coefficients,vectorsIn,target):
        return constraint_overshooting_algebra(coefficients,vectorsIn,target)
    ### ## Defining Constraints for minimize function
    #   #    Add the second constraint to the constraints-list below to add the constraint
    #   #    But be cautious: With that, better also remove the Non-Essential Amino-Acids from all profiles to allow everything making sense
    activeConstraints=[
        {'type': 'eq', 'fun': constraint_sum}
        # ,
        # {'type': 'ineq', 'fun': constraint_overshooting, 'args': (vectors,target)}
    ]
    #
    ### Define the range for the coefficients as Bounds: 0 <= coefficients <= 1
    ### ## For the Sum-Constraint it is actually indifferent what the upper bound is. The total sum assures that no single value goes beyond 1.0. Here, only the lower bound is relevant to avoid negative numbers
    ### ## For the Overshooting-Constaint on the other hand, an upper bound of 1 can be too restrictive.
    ### ## -> Hence, here in the code, I just set the upper bound to a higher value. It makes no difference for the Sum-Constraint, but gives the Overshooting more freedom.
    bounds=[(0, 10)]*num_vec
    #
    ### Make one very first run
    ### ## Initialize coefficients with equal weights
    initial_guess=np.ones(num_vec)/num_vec
    resultGlobal=basinhopping(
        objective,
        initial_guess,
        minimizer_kwargs={
            'method':'SLSQP',#'nelder-mead' 'BFGS'
            'bounds':bounds,
            'constraints':activeConstraints
        },
        stepsize=0.02,
        niter=100,
        niter_success=10
    )
    ### Execute multiple times in loop to start with different initial-guesses
    capacity=1
    num_entries=num_vec
    permutations_generator=distribute_capacity(capacity, num_entries)
    ### Only every xth Permutation shall actually be computed
    ### ## Having 7 AA Profiles defined results already in 3432 permutations. Only using every 100th still computes about 30 basinhops per target Profile.
    num_perms=num_permutations(capacity, num_entries)
    perms_toCalc=10# Set it to num_perms to calculate all
    permCount=0
    for permutation in permutations_generator:
        permCount+=1
        if not 0==permCount % int(num_perms/perms_toCalc):
            continue
        initial_guess=np.array(permutation)
        ### Solve the quadratic program
        #result=minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=activeConstraints)
        result=basinhopping(
            objective,
            initial_guess,
            minimizer_kwargs={
                'method':'SLSQP',#'nelder-mead' 'BFGS'
                'bounds':bounds,
                'constraints':activeConstraints
            },
            stepsize=0.02,
            niter=100,
            niter_success=10
        )
        print(f"Termination: {result.message}")
        if resultGlobal is None:
            resultGlobal=result
        else:
            if resultGlobal.fun>result.fun:
                print(f"replaced {resultGlobal.fun} with {result.fun}")
                print(f"{np.round(resultGlobal.x,2)} -> \n{np.round(result.x,2)}")
                print(f"Init: {initial_guess}")
                resultGlobal=result
    #
    ### Retrieve the optimal coefficients
    coefficients=resultGlobal.x
    #
    return coefficients



def optimize_forTarget(profileName,targetVec,plants,datasets):
    resOut['json']['Out'][profileName]={}
    printOut_res(f"====================================\n-- Target Profile: {profileName}\n------------------------------\n")
    #
    datlen=len(targetVec)
    datnum=len(datasets)
    #
    matrix=np.empty((datlen,0))
    for vec in datasets:
        vec=vec.reshape(datlen,1)
        matrix=np.hstack((matrix,vec))
    # print("Dataset Matrix:\n",matrix)
    # print("")
    #
    coefficients=fit_vector(matrix,targetVec)
    set_json(["Optimal Coefficients"],coefficients.tolist())
    printOut_res(f"Optimal coefficients: {coefficients}")
    printOut_res("")
    #
    rounded_coef=[]
    roundedSum=0
    for toRound in coefficients:
        rounded=round(toRound,2)
        rounded_coef.append(rounded)
        roundedSum+=rounded
    #
    set_json_perProf(["Rounded Coefficients"],rounded_coef)
    printOut_res(f"Rounded: (Sum: {roundedSum})")
    printOut_res("Rounded coefficients: [")
    for i in range(len(datasets)):
        name=plants[i]
        c=rounded_coef[i]
        printOut_res(f"  {c}\t{name}")
    printOut_res("]")
    #
    ### Kept some other variants for calculating the stuff for the archive
    def resulting_linearCombination_v1():
        vResult=np.zeros(datlen)
        for i in range(0,len(datasets),1):
            vec=datasets[i]
            coe=rounded_coef[i]
            vResult=vResult+coe*vec
        return vResult
    def resulting_linearCombination_v2():
        vecC=np.array(rounded_coef)
        # vecC=vecC.reshape(datnum,1)
        vResult=np.dot(matrix,vecC)
        return vResult
    def resulting_linearCombination():
        return resulting_linearCombination_v2()
    #
    vResult=resulting_linearCombination()
    #
    return rounded_coef,vResult


def evaluate_result(targetVec,datasets,coefs,vResult):
    AASum=0
    for aacid in vResult:
        AASum+=aacid
    AASum=round(AASum,1)
    printOut_res("")
    printOut_res("Result (with rounded coefficients):")
    printOut_res(f"\tSum: {AASum}")
    printOut_res(f"{vResult}")
    vResult_converted=vResult.tolist()
    for i in range(len(vResult_converted)):
        vResult_converted[i]=round(vResult_converted[i],3)
    set_json_perProf(["Resulting Profile"],vResult_converted)
    #
    vDiff=targetVec-vResult
    printOut_res("")
    printOut_res(f"Delta to Target:\n{vDiff}")
    vDiff_converted=vDiff.tolist()
    for i in range(len(vDiff_converted)):
        vDiff_converted[i]=round(vDiff_converted[i],3)
    set_json_perProf(["Delta to Target"],vDiff_converted)
    #
    eucNorm=euclideanNormDistance(targetVec,vResult)
    set_json_perProf(["Euclidean Norm"],{})
    set_json_perProf(["Euclidean Norm","val"],eucNorm)
    eucNorm=round(eucNorm,2)
    set_json_perProf(["Euclidean Norm","rounded"],eucNorm)
    printOut_res(f"Euclidian-Norm (Length of Delta-Vector): {eucNorm}")
    #
    cosSim=cosineSimilarity(targetVec,vResult)
    set_json_perProf(["Cosinus Similarity"],{})
    set_json_perProf(["Cosinus Similarity","val"],cosSim)
    cosSim=round(cosSim,2)
    set_json_perProf(["Cosinus Similarity","rounded"],cosSim)
    printOut_res(f"Cosine Similarity: {cosSim}")
    #
    quotScore=quotientSimilarityScore(targetVec,vResult)
    set_json_perProf(["Quotient Score"],{})
    set_json_perProf(["Quotient Score","val"],quotScore)
    quotScore=round(quotScore,2)
    set_json_perProf(["Quotient Score","rounded"],quotScore)
    printOut_res(f"Quotient Score: {quotScore}")
    #
    printOut_res("\n")
    #
    return eucNorm,cosSim,quotScore



def compute_forAll_targetProf(targetProfiles,targetVecs,plants_toUse,datasets_toUse):
    coefficientsCollection=[]
    ResultVectors=[]
    eucNorms=[]
    cosSims=[]
    quotScores=[]
    for i in range(len(targetVecs)):
        targetVec=targetVecs[i]
        profileName=targetProfiles[i]
        #
        resOut['curProfileIdx']=i
        resOut['curProfile']=profileName
        #
        coefficients,vResult=optimize_forTarget(profileName,targetVec,plants_toUse,datasets_toUse)
        eucNorm,cosSim,quotScore=evaluate_result(targetVec,datasets_toUse,coefficients,vResult)
        #
        coefficientsCollection.append(coefficients)
        ResultVectors.append(vResult)
        eucNorms.append(eucNorm)
        cosSims.append(cosSim)
        quotScores.append(quotScore)
    #
    return coefficientsCollection,ResultVectors,eucNorms,cosSims,quotScores


 


def analyze_Input(targetProfiles,targetVecs,plants,datasets,enabledIngredients):
    global plot_InputAnalysis
    global progPathOutLatestPlotInAnal
    plants_active=[]
    datasets_active=[]
    for i in range(len(plants)):
        plant=plants[i]
        datset=datasets[i]
        enabled=enabledIngredients[i]
        if enabled:
            plants_active.append(plant)
            datasets_active.append(datset)
    printOut_std("Input Analysis:")
    printOut_std("\t(Directly comparing the ingredients with the target profiles)")
    for i in range(len(targetProfiles)):
        trgtName=targetProfiles[i]
        trgt=targetVecs[i]
        set_json(['json','In-Analysis',trgtName],{})
        printOut_std(f"{trgtName}:")
        for j in range(len(plants_active)):
            plantName=plants_active[j]
            plant=datasets_active[j]
            printOut_std(f"   {plantName}:[")
            set_json(['json','In-Analysis',trgtName,plantName],{})
            eucNorm=euclideanNormDistance(trgt,plant)
            eucNorm=round(eucNorm,2)
            cosSim=cosineSimilarity(trgt,plant)
            cosSim=round(cosSim,2)
            quotScore=quotientSimilarityScore(trgt,plant)
            quotScore=round(quotScore,2)
            printOut_std(f"      Euclidean Norm: {eucNorm}")
            printOut_std(f"      Cosinus Similarity: {cosSim}")
            printOut_std(f"      Quotient Score: {quotScore}")
            set_json(['json','In-Analysis',trgtName,plantName,'Euclidean Norm'],eucNorm)
            set_json(['json','In-Analysis',trgtName,plantName,'Cosinus Similarity'],cosSim)
            set_json(['json','In-Analysis',trgtName,plantName,'Quotient Score'],quotScore)
            printOut_std(f"   ]")
            #
            if plot_InputAnalysis and save_result:
                plotName=f"{trgtName} vs {plantName}"
                fig=create_resultPlot('Target Profile vs. Ingredient',
                    trgtName,trgt,
                    plantName,plant,
                    eucNorm,cosSim,quotScore
                )
                save_plot(progPathOutLatestPlotInAnal,plotName,fig)
                plt.close(fig)
    #
    printOut_std("\n#######################################")
    printOut_std("#######################################")
    printOut_std("#######################################\n")
    #
    return plants_active,datasets_active





def printDirectoryHierarchyNote():
    print("Note on Directory Structure:")
    print("- In case you want the results to be stored in files, make sure you have the main.py script inside a folder called \"src\", which in turn lies inside a dedicated parent directory.")
    print("  - (It is indifferent how the parent directory is called, but it is recommended that it is solely dedicated to this script, i.e. it keeps nothing aside this script within its src-dir.)")
    print("- I.e. have something like \"[...]/Amino-Acid Profile Resembling Optimization/src/main.py\".")
    print("- The script then creates a directory \"out\" parallel to \"src\", in which the results are stored.")
    print("- The creation of the out-dir and the storage of results in only performed in case you comply with this prerequisite to assure that it doesn't mess with your file system.")
def printUsage():
    global outSubDirLatest
    print("\nUsage:")
    print("- This Script is able to store the computed Results to persistent Storage.")
    print("  - You can configur whether this shall be done in the main() of the Script with \"persistentStorage_turnOn() / ..._turnOff()\".")
    print("  - This also controls whether or not the Plots are stored.")
    print("- It can Plot the Results to a bar digram. Also this can be turned on/off; with \"plotting_turnOn() / ..._turnOff()\".")
    print("- The results are stored in two ways:")
    print(f"  - The results from the current run are stored in the sub-directory \"{outSubDirLatest}\".")
    print("  - The results are duplicated stored depending on the configured Ingredients to include.")
    print(f"    - For that, a directory parallel to \"{outSubDirLatest}\" is created.")
    print("    - This dir's name is a sequence of '0' & '1' which represent the enabled/disabled ingredients.")
    print("    - That way you can keep the results of distinct runs with different ingredient configurations, without them being overwritten.")
    print("")
    printDirectoryHierarchyNote()
    
def printLegend():
    VERSION_DESCRIPTION=set_Version_Description()
    printOut_std(f"PFAPO - Plant Fusion AA Profile Optimization (v. »{VERSION_DESCRIPTION}«)")
    printOut_std("\n#######################################\n")
    printOut_std("Indices in the Datasets for the Amino Acids are:[")
    for idx in AAindices:
        printOut_std(f"  {idx}")
    printOut_std("]")
    for prLine in AAidxLegend:
        printOut_std(f"{prLine}")
    printOut_std("\n#######################################")
    printOut_std("#######################################")
    printOut_std("#######################################\n")

def print_enabled_Ingredients(plants,enabledIngredients):
    printOut_std("Following Ingredients were in-/excluded for Computation:")
    for i in range(len(plants)):
        plant=plants[i]
        enab=enabledIngredients[i]
        printOut_std(f"{plant}: {enab}")
    printOut_std("\n#######################################")
    printOut_std("#######################################")
    printOut_std("#######################################\n")
        


    # add_Target(
    #     "Egg & Meat",
    #     [8.3, 4.9, 5.9, 7.9, 4.2, 1.25, 4.45, 8.6, 14.15, 6.65, 2.8, 30.9],
    # )
def set_TargetProfiles():
    global resOut
    def add_Target(name,profile):
        targetProfiles.append(name)
        targetVecs.append(np.array(profile))
        resOut['json']['In']['Target-Profiles'][name]=profile
        resOut['Terminal']['Result'].append("")
        printOut_std(f"{name}=[",end='')
        AAcid=profile[0]
        printOut_std(f"{AAcid}",end='')
        for i in range(1,len(profile)):
            AAcid=profile[i]
            printOut_std(f", {AAcid}",end='')
        printOut_std("]")
    targetProfiles=[]
    targetVecs=[]
    #
    printOut_std("The Target Profiles to resemble:")
    #
    add_Target(
        "WHO",
        [5.9, 3, 3.9, 4.5, 2.3, 0.6, 2.2, 3, 10, 6.5, 1.5, 56.6],
    )
    add_Target(
        "Whole-Body Protein Tissue",
        [7.5, 3.5, 4.9, 7.3, 4.2, 1.2, 3.5, 7.3, 14, 6.5, 2.7, 37.4],
    )
    add_Target(
        "Human Skeletal Muscle",
        [6.3, 3.4, 4.3, 6.6, 2.9, 1.3, 1.7, 5.8, 13.1, 4.4, 2.8, 47.4],
    )
    add_Target(
        "Egg (Chicken)",
        [8.6, 5.3, 6.8, 7.3, 4.4, 1.3, 5.2, 9.4, 13.3, 6.5, 2.4, 29.5],
    )
    add_Target(
        "Meat, Beef",
        [8, 4.5, 5, 8.5, 4, 1.2, 3.7, 7.8, 15, 6.8, 3.2, 32.3],
    )
    add_Target(
        "Meet, Chicken, Breast",
        [7.5, 5.3, 5, 8.5, 4.2, 1.3, 3.6, 7.6, 15, 6.8, 3.7, 31.5],
    )
    add_Target(
        "Function-oriented",
        [9, 4.5, 5, 7.3, 4.5, 1.2, 5, 8, 15, 6.5, 2.7, 31.3],
    )
    add_Target(
        "Digestion-oriented",
        [8.1, 5, 6.4, 6.8, 4.1, 1.2, 4.9, 8.8, 12.5, 6.1, 2.2, 34],
    )
    #
    printOut_std("\n#######################################")
    printOut_std("#######################################")
    printOut_std("#######################################\n")
    #
    return targetProfiles,targetVecs

ingredientIdx={}
def set_IngredientDatasets():
    '''
    You can use the "enabledIngredients" to DISABLE certain Ingredients for calculation.
    For example you might perform multiple full runs in succession, but with different Ingredients enabled.
    Then just alter the entries in this list, like flipping a '1' to '0' for a certain plant to take it out.
    '''
    global resOut
    def add_Ingredient(name,AAprofile):
        plants.append(name)
        ingredientIdx[plants[len(plants)-1]]=len(plants)-1
        enabledIngredients.append(True)
        datasets.append(np.array(AAprofile))
        set_json(['json','In','Ingredient-Candidates',name],{})
        set_json(['json','In','Ingredient-Candidates',name,'enabled'],True)
        set_json(['json','In','Ingredient-Candidates',name,'AA-Profile'],AAprofile)
        printOut_std(f"{name}=[",end='')
        AAcid=AAprofile[0]
        printOut_std(f"{AAcid}",end='')
        for i in range(1,len(AAprofile)):
            AAcid=AAprofile[i]
            printOut_std(f", {AAcid}",end='')
        printOut_std("]")
    global ingredientIdx
    plants=[]
    enabledIngredients=[]
    datasets=[]
    #
    printOut_std("The Ingredient-Candidates to mix proportionally:")
    ###############################
    add_Ingredient(
        'Pea (Yellow)',
        [8.5, 4.5, 5.2, 8.1, 3.7, 0.8, 2.2, 9.1, 17.1, 8, 3.1, 29.7]
    )
    add_Ingredient(
        'Hemp',
        [6.6, 4.1, 5.3, 3.8, 3.7, 1.2, 4.5, 8, 18, 12.4, 2.8, 29.6]
    )
    add_Ingredient(
        'Soy',
        [7.9, 4.4, 4.6, 6.5, 3.8, 1.1, 2.2, 8.8, 19.5, 7.6, 2.9, 30.7]
    )
    add_Ingredient(
        'Rice',
        [8.3, 4.3, 5.9, 3.6, 3.7, 1.3, 4, 9.7, 19.6, 7.8, 2.5, 29.4]
    )
    add_Ingredient(
        'Lupine',
        [7.5, 3.9, 3.8, 6.6, 3.8, 0.7, 1.5, 7.8, 24.6, 10.5, 2.9, 26.4]
    )
    add_Ingredient(
        'Pumpkin Seed',
        [7.2, 3.8, 4.7, 3.7, 3, 1.7, 2.8, 8.4, 18.5, 16, 2.3, 27.7]
    )
    add_Ingredient(
        'Quinoa',
        [7.3, 4.2, 5, 6.1, 3.2, 1, 4, 7.9, 16.4, 10.4, 3.1, 31.4]
    )
    add_Ingredient(
        'Potato (Yellow)',
        [5.5, 3.5, 5.4, 5.9, 3.8, 0.7, 2.7, 8.6, 17.2, 5.3, 1.9, 39.5]
    )
    add_Ingredient(
        'Potato (Protein Concentrate)',
        [9.7, 5.6, 6.4, 7.3, 5.8, 1.5, 3.8, 11.3, 10.5, 5, 2.2, 30.9]
    )
    add_Ingredient(
        'Fava',
        [8, 4.3, 4.8, 6.8, 3.8, 1, 2.2, 7.9, 18.2, 9.9, 2.7, 30.3]
    )
    add_Ingredient(
        'Mushrooms (Agaricus)',
        [6.3, 4.1, 4.7, 10.4, 4.7, 2.3, 2.2, 6.2, 17.6, 5.1, 2.8, 33.6]
    )
    ###############################
    printOut_std("\n#######################################")
    printOut_std("#######################################")
    printOut_std("#######################################\n")
    return plants,datasets,enabledIngredients


#------------------------------------------------------------------------------------------

def debug_print_Output():
    global resOut
    print("\n==== Debug, Terminal-Output-Std ====")
    print(resOut['Terminal']['Std'])
    for trgtOut in resOut['Terminal']['Result']:
        print("\n==== Debug, Terminal-Output ====")
        print(trgtOut)
    print("\n==== Debug, Output-JSON ====")
    print(resOut['json'])
    print("\n==== ==== ==== ==== ==== ====\n")
    return

def main():
    def disableIngredient(plantString):
        enabledIngredients[ingredientIdx[plantString]]=False
        resOut['json']['In']['Ingredient-Candidates'][plantString]['enabled']=False
    def enableIngredient(plantString):
        enabledIngredients[ingredientIdx[plantString]]=True
        resOut['json']['In']['Ingredient-Candidates'][plantString]['enabled']=True
    #
    err=0
    global ingredientIdx
    global resOut
    #
    ### Whether or not the Results shall be drawn on a Plot
    plotting_turnOff()
    plotting_turnOn()
    ### Whether or not all Results (As Text, JSON and Plot) shall be persistently stored to disk
    persistentStorage_turnOff()
    persistentStorage_turnOn()
    ### Whether or not the Analysis of the Input (Ingredients vs. Target) shall be stored as Plots
    plotting_inputAnalysis_turnOff()
    # plotting_inputAnalysis_turnOn()
    #
    printLegend()
    #
    targetProfiles,targetVecs=set_TargetProfiles()
    plants,datasets,enabledIngredients=set_IngredientDatasets()
    #
    disableIngredient('Hemp')
    enableIngredient('Hemp')
    disableIngredient('Quinoa')
    disableIngredient('Potato (Yellow)')
    disableIngredient('Potato (Protein Concentrate)')
    disableIngredient('Fava')
    disableIngredient('Mushrooms (Agaricus)')
    print_enabled_Ingredients(plants,enabledIngredients)
    #
    set_output_paths(enabledIngredients)
    clean_resultLatest()
    #
    activePlants,activeDatasets=analyze_Input(targetProfiles,targetVecs,plants,datasets,enabledIngredients)
    #
    coefficientsCollection,ResultVectors,eucNorms,cosSims,quotScores=compute_forAll_targetProf(targetProfiles,targetVecs,activePlants,activeDatasets)
    #
    dump_resultOut()
    if draw_Plot_Switch:
        plot_resultComparison(targetProfiles,targetVecs,activePlants,coefficientsCollection,ResultVectors,eucNorms,cosSims,quotScores)
    duplicate_Out_fromLatest()
    #
    ### Show the plot(s) - Last thing to do, because of GUI-Loop
    plot_show()
    #
    return err


if __name__ == '__main__':
    set_ProgramPath(__file__)
    # - - - - - - - - -
    main()