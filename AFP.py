"""
Advanced Fitting Program
To provide the best fitting of Energy Loss Function (ELF)
Using machine learning to automatically to identify and fit the curve
fitting the first model (negative oscillator added) 0823
NUAA ZLH
2022/5/17 -
"""

import numpy as np
import scipy as sci
from scipy import integrate
from scipy import interpolate
from scipy.misc import derivative
from scipy.signal import find_peaks
from scipy.signal import find_peaks_cwt
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt
from lmfit import Parameters, Minimizer, report_fit, minimize, printfuncs
from scipy import constants

# constants
e = 1.60217663410 * 10 ** -19
NA = 6.022E23
me = 9.1093837015E-31
pv = 8.854187817E-12  # Vacuum Dielectric Constant  (C2/(N*m2))
h_bar = 6.582119514 * 10 ** (-16)  # Reduced Planck Constant eV/s

# ELF_data = 'oELF/oELF_for_fit3.txt'
# f = open(ELF_data,'r')
#
# iniE = []
# iniELF = []
# for line in f:
#     string = line.split()
#     iniE.append(string[0])
#     iniELF.append(string[1])
# arr_E = np.asarray(iniE).astype(np.float64)
# arr_ELF = np.asarray(iniELF).astype(np.float64)
# f.close()

# elem = 'Au'
def ReadELF(elem):
    ELFname = 'oELF/oELF_'+elem+'.dat'
    f = open(ELFname, 'r')
    iniE = []
    iniELF = []
    im_ep = []
    re_ep = []
    for num,line in enumerate(f):
        if num == 0:
            continue
        string = line.split()
        iniE.append(string[0])
        re_ep.append(string[3])
        im_ep.append(string[4])
        iniELF.append(string[5])
    arr_E = np.asarray(iniE).astype(np.float64)
    arr_imep = np.asarray(im_ep).astype(np.float64)
    arr_reep = np.asarray(re_ep).astype(np.float64)
    arr_ELF = np.asarray(iniELF).astype(np.float64)
    f.close()
    return arr_E, arr_imep, arr_reep, arr_ELF

# iniE = []
# iniELF = []
# for line in f:
#     string = line.split()
#     iniE.append(string[0])
#     iniELF.append(string[1])
# arr_E = np.asarray(iniE).astype(np.float64)
# arr_ELF = np.asarray(iniELF).astype(np.float64)
# f.close()


def ReadData(elem):
    basicData = 'oELF/BasicInfo.dat'
    f = open(basicData,'r')
    found = False
    for num,line in enumerate(f):
        string = line.split()
        if string[0] == elem:
            Z = float(string[1])
            M = float(string[2])
            density = float(string[3])
            found = True
    if found == False:
        print("Element "+elem+ " is not found")
        return 0
    return Z,M,density


def ReadParameters(filename,numofpara):
    f = open(filename, 'r')
    ai = []
    wpi = []
    gammai = []
    Eth = []
    delta = []
    for num,line in enumerate(f):
        if num == 0:
            continue
        str = line.split()
        ai.append(str[0])
        wpi.append(str[1])
        gammai.append(str[2])
        if numofpara >= 4:
            Eth.append(str[3])
        if numofpara >= 5:
            delta.append(str[4])
    arr_ai = np.asarray(ai).astype(np.float64)
    arr_wpi = np.asarray(wpi).astype(np.float64)
    arr_gammai = np.asarray(gammai).astype(np.float64)
    if numofpara == 3:
        return arr_ai, arr_wpi, arr_gammai
    if numofpara == 4:
        arr_Eth = np.asarray(Eth).astype(np.float64)
        return arr_ai, arr_wpi, arr_gammai, arr_Eth
    if numofpara >= 5:
        arr_Eth = np.asarray(Eth).astype(np.float64)
        arr_delta = np.asarray(delta).astype(np.float64)
        return arr_ai, arr_wpi, arr_gammai, arr_Eth, arr_delta



def BindingEnergy(elem):
    Bind_data = 'oELF/BindErg.dat'
    fb = open(Bind_data, 'r')
    ini_bderg = []
    num_find = 0
    found = False
    finding = False
    for num,line in enumerate(fb):
        string = line.split()
        if string[0] == elem:
            # print("found!")
            num_find = num
            # print(num_find)
            finding = True
            # print(finding)
        if num == num_find+2 and finding:
            ini_bderg = string[0:-1]
            finding = False
            found = True
    if found == False:
        print("No found for element!")
        return 0
    Bind_erg = np.asarray(ini_bderg).astype(np.float64)*1000.0
    fb.close()
    return Bind_erg


def PlasmonPeak(elem):
    PlasmonData = 'oELF/PlasmonPeak.dat'
    fp = open(PlasmonData, 'r')
    pl_erg = []
    pl_width = []
    found = False
    for num, line in enumerate(fp):
        string = line.split()
        if string[0] == elem:
            if len(string) >= 4:
                pl_string = string[3].split(';')
                if len(pl_string)>1:
                    for i in range(len(pl_string)):
                        pl_erg.append(pl_string[i])
                else:
                    pl_erg.append(pl_string[0])
            if len(string) >= 5:
                pl_width.append(string[4])
            found = True
    if found == False:
        print("No found for element!")
        return 0
    plasmonErg = np.asarray(pl_erg).astype(np.float64)
    plasmonWidth = np.asarray(pl_width).astype(np.float64)
    return plasmonErg,plasmonWidth


# peaks,_ = find_peaks(arr_ELF, height = 0)
# plt.loglog(arr_E,arr_ELF)
# # for i in Bind_erg:
# plt.vlines(Bind_erg,0,1)
# # plt.loglog(arr_E[peaks], arr_ELF[peaks], "x")
# plt.show()

# The most simple model but require a lot of oscillators even for the negative oscillators
# @jit
def Drude(ai,omegaP,gamma,w):
    # X = ai * omegaP ** 2 * gamma * w
    X = ai * gamma * w
    Y = (omegaP ** 2 - w ** 2) ** 2 + (gamma * w) ** 2
    return X/Y


def DrudeRe(ai,omegaP,gamma,w):
    X = ai * (omegaP**2 - w**2)
    Y = (omegaP**2 - w**2)**2 + (gamma*w)**2
    return X/Y


def switching(E_Eth):
    return 1/(1+np.exp(E_Eth))


# For a threhold drude function fitting
def DrudeEth(ai, omegaP, gamma, Eth, w):
    X = ai * gamma * w * np.heaviside(w-Eth,1)
    Y = (omegaP ** 2 - w ** 2) ** 2 + (gamma * w) ** 2
    return X/Y


def DrudeReEth(ai, omegaP, gamma, Eth,w):
    X = ai * (omegaP ** 2 - w ** 2) * np.heaviside(w-Eth,1)
    Y = (omegaP ** 2 - w ** 2) ** 2 + (gamma * w) ** 2
    return X/Y


def DrudeSigmoid(ai, omegaP, gamma, Eth, w):
    X = ai * gamma * w * 1/(1+np.exp(Eth - w))
    Y = (omegaP ** 2 - w ** 2) ** 2 + (gamma * w) ** 2
    return X/Y


def DrudeReSigmoid(ai, omegaP, gamma, Eth, w):
    X = ai * (omegaP ** 2 - w ** 2) * 1/(1+np.exp(Eth - w))
    Y = (omegaP ** 2 - w ** 2) ** 2 + (gamma * w) ** 2
    return X / Y


# For a exponetial threhold drude function fitting
# def DrudeEthexp(ai,omegaP,gamma,Eth,w):
#     subtract = Drude(ai,omegaP,gamma,Eth)*np.exp(Eth-w)
#     X = ai * gamma * w
#     Y = (omegaP ** 2 - w ** 2) ** 2 + (gamma * w) ** 2
#     return (X/Y - subtract)* np.heaviside(w-Eth,1)


#Derivative Drude function
# def DerivativeDrude(ai,omegaP,gamma,w):
#     X = 2 * ai * gamma**3 * w**3
#     Y = ((omegaP ** 2 - w ** 2) ** 2 + (gamma * w) ** 2)**2
#     return X / Y

# def DerivativeDrudeEthexp(ai,omegaP,gamma,Eth,w):
#     X = 2 * ai * gamma ** 3 * w ** 3
#     Y = ((omegaP ** 2 - w ** 2) ** 2 + (gamma * w) ** 2) ** 2
#     subtract = DerivativeDrude(ai, omegaP, gamma, Eth) * np.exp(Eth - w)
#     return (X / Y - subtract) * np.heaviside(w-Eth,1)

# Generate initial guess
def GenerateIni(Erg,ELF,mode = 0):
    # func = interpolate.interp1d(Erg, ELF, kind='linear')
    # new_E = np.logspace(-1, 6, 1000)
    # new_E = np.logspace(np.log10(0.126), np.log10(1e5),1000)
    # new_ELF = func(new_E)
    # new_ELF = log_interp(new_E,Erg,ELF)
    peaks,_ = find_peaks(ELF,height = 0.0)
    # plasmonErg,_ = PlasmonPeak(elemName)

    # pre_omegaP = Bind_erg
    pre_omegaP = Erg[peaks]
    # pre_omegaP = np.concatenate((new_E[peaks],Bind_erg))
    pre_omegaP.sort()
    print(pre_omegaP)
    # print(pre_omegaP)
    # plt.loglog(arr_E, arr_ELF)
    # plt.vlines(pre_omegaP, 0, 1.2)
    # plt.show()
    # peak_values = func(pre_omegaP)
    peak_values = log_interp(pre_omegaP, Erg, ELF)
    # pre_gamma = np.ones(np.size(pre_omegaP))
    # pre_gamma = peak_values
    pre_gamma = pre_omegaP/2.0
    pre_A = 0.7*peak_values/Drude(1.0,pre_omegaP, pre_gamma,pre_omegaP)

    # test f-sum rule first
    # test initial guess
    plt.figure()
    pre_ELF = np.zeros((np.size(pre_omegaP)+1,np.size(Erg)))
    for i in range(np.size(pre_omegaP)):
        pre_ELF[i,:] = Drude(pre_A[i],pre_omegaP[i],pre_gamma[i],Erg)
        pre_ELF[-1,:] += pre_ELF[i,:]

    print(len(pre_A))
    plt.loglog(Erg, ELF)
    for i in range(np.size(pre_omegaP)+1):
        plt.loglog(Erg, pre_ELF[i,:])
    plt.show()
    while True:
        new_values = input('如果你希望增加新的振子，请输入振子能量，只输入end即终止循环输入,如果需要删除元素，输入delete + 该元素的下标（从0开始）')
        if 'end' in new_values:
            break
        if 'delete' in new_values:
            line = new_values.split()
            tag = list(map(int, line[1:]))
            print('delete', tag)
            # for i in tag:
            pre_omegaP = np.delete(pre_omegaP,tag, axis = 0)
            pre_A = np.delete(pre_A, tag, axis = 0)
            pre_gamma = np.delete(pre_gamma, tag, axis = 0)
        else:
            new_omegaP = np.asarray(new_values.split()).astype(np.float64)
            pre_omegaP = np.concatenate((pre_omegaP,new_omegaP))
            pre_omegaP.sort()
            # peak_values = func(pre_omegaP)
            peak_values = log_interp(pre_omegaP, Erg, ELF)
            # pre_gamma = np.ones(np.size(pre_omegaP))
            # pre_gamma = peak_values
            pre_gamma = pre_omegaP/2.0
            pre_A = 0.7*peak_values / Drude(1.0, pre_omegaP, pre_gamma, pre_omegaP)
        plt.figure()
        pre_ELF = np.zeros((np.size(pre_omegaP), np.size(new_E)))
        for i in range(np.size(pre_omegaP)):
            pre_ELF[i, :] = Drude(pre_A[i], pre_omegaP[i], pre_gamma[i], new_E)
        print(len(pre_A))
        plt.loglog(new_E, new_ELF)
        for i in range(np.size(pre_omegaP)):
            plt.loglog(new_E, pre_ELF[i, :])
        plt.show()
    pre_Eth = np.zeros(len(pre_omegaP))
    if mode == 1:
        print("需要手动给拟合峰指定阈值,给定阈值的会以Sigmoid函数在Eth处衰减")
        print("未指定阈值的峰则没有衰减函数")
        while True:
            pre_ELF = np.zeros((np.size(pre_omegaP), np.size(new_E)))
            for i in range(np.size(pre_omegaP)):
                if pre_Eth[i] < 1e-5:
                    pre_ELF[i, :] = Drude(pre_A[i], pre_omegaP[i], pre_gamma[i], new_E)
                else:
                    pre_ELF[i,:] = DrudeSigmoid(pre_A[i], pre_omegaP[i], pre_gamma[i], pre_Eth[i], new_E)
            print(len(pre_ELF))
            plt.loglog(new_E, new_ELF)
            for i in range(np.size(pre_omegaP)):
                plt.loglog(new_E, pre_ELF[i, :])
            plt.show()
            line = input("峰编号+空格+阈值，end即终止循环")
            if 'end' in line:
                break
            string = line.split()
            tag = int(string[0])
            pre_Eth[tag] = float(string[1])
    if mode == 2:
        print("需要手动给拟合峰指定阈值,给定阈值的会以阶跃函数在Eth处衰减")
        while True:
            line = input("峰编号+空格+阈值，end即终止循环")
            pre_ELF = np.zeros((np.size(pre_omegaP), np.size(new_E)))
            for i in range(np.size(pre_omegaP)):
                if pre_Eth < 1e-5:
                    pre_ELF[i, :] = Drude(pre_A[i], pre_omegaP[i], pre_gamma[i], new_E)
                else:
                    pre_ELF[i,:] = DrudeSigmoid(pre_A[i], pre_omegaP[i], pre_gamma[i], pre_Eth, new_E)
            print(len(pre_ELF))
            plt.loglog(new_E, new_ELF)
            for i in range(np.size(pre_omegaP)):
                plt.loglog(new_E, pre_ELF[i, :])
            plt.show()
            if 'end' in line:
                break
            string = line.split()
            tag = int(string[0])
            pre_Eth[tag] = float(string[1])
    # f = open("Initial Guess step 1.txt",'w')
    if mode == 0:
        return pre_A, pre_omegaP, pre_gamma
    else:
        return pre_A, pre_omegaP, pre_gamma, pre_Eth



# GenerateIni(arr_E,arr_ELF,'Au')


def f_sumrule(Erg,ELF,Emin, Emax, ro, M):
    Ergmin = Erg[0]
    Ergmax = Erg[-1]
    N = 1000
    logdiv = np.linspace(np.log10(Emin), np.log10(Emax), N)
    Ee = 10 ** logdiv
    ro = ro * 1e3 # g/cm3 -> kg/m3
    M = M/1e3 # g/mol -> kg/mols
    """
    if Emin < 0.01 and Emax > 1e6:
        Emin = 0.01
        Emax = 1e6
        Ee = np.logspace(-2, 6, N)
    else:
        logdiv = np.linspace(np.log10(Emin),np.log10(Emax),N)
        Ee = 10 ** logdiv
    """

    # ro = 19.32E3  # 金的密度 kg/m3
    # M = 196.9665523000E-3  # 金的相对原子质量 kg/mol
    # func = interpolate.interp1d(Erg, ELF, kind='linear')
    omegaP2 = NA * ro / M * e ** 2 / (me * pv) * h_bar ** 2
    # print(omegaP2)
    result = np.zeros(N)
    for i in range(N-1):
        # print(i)
        #Emin = 0.005
        Emaxi = Ee[i+1]
        num = 1000
        W = np.zeros(num)
        W[1:] = np.logspace(np.log10(Emin),np.log10(Emaxi),num-1)
        # ELFinterps = func(W)*W*2/(np.pi*omegaP2)
        ELFinterps = np.zeros(num)
        ELFinterps[1:] = log_interp(W[1:],Erg,ELF)*W[1:]*2/(np.pi*omegaP2)
        #result[i] = np.trapz(ELFinterp,W)
        result[i+1] = integrate.trapz(ELFinterps,W)
        #print([Ee,result])
    result[0] = 0.0
    plt.figure()
    plt.semilogx(Ee,result)
    plt.show()
    return result[-1]

def P_f_sumrule(A,omega,gamma,Emin,Emax,ro,M):
    ro = ro * 1e3  # g/cm3 -> kg/m3
    M = M / 1e3  # g/mol -> kg/mols
    omegaP2 = NA * ro / M * e ** 2 / (me * pv) * h_bar ** 2
    N = 1000
    Ee = np.logspace(np.log10(Emin),np.log10(Emax),N)
    integrand = np.zeros(N)
    for i in range(N):
        ELF = Drude(A,omega,gamma,Ee[i])
        integrand[i] = ELF*Ee[i]*2/(np.pi*omegaP2)
    result = integrate.simps(integrand,Ee)
    return result


def plotELF(Erg, ELF, ELF_re,Emin, Emax, type):
    plt.figure()
    if type == 'loglog':
        plt.loglog(Erg, ELF, label='Im[1/epsilon]')
        plt.loglog(Erg, ELF_re, label='Re[1/epsilon]')
    if type == 'linear':
        plt.plot(Erg, ELF, label='Re[1/epsilon]')
        plt.plot(Erg, ELF_re, label='Re[1/epsilon]')
    if type == 'semilogx':
        plt.semilogx(Erg, ELF, label='Re[1/epsilon]')
        plt.semilogx(Erg, ELF_re, label='Re[1/epsilon]')
    plt.xlim(Emin,Emax)
    plt.legend()
    plt.show()


def ps_sumrule(Erg,ELF):
    Emin = min(Erg)+0.000001
    # print(Emin)
    Emax = max(Erg)-0.000001
    # print(Emax)
    N = 1000
    logdiv = np.linspace(np.log10(Emin), np.log10(Emax), N)
    Ee = 10 ** logdiv
    func = interpolate.interp1d(Erg, ELF, kind='linear')
    result = np.zeros(N)
    for i in range(N-1):
        # print(i)
        #Emin = 0.005
        Emaxi = Ee[i+1]
        logdiv = np.linspace(np.log(Emin), np.log(Emaxi), 10000)
        W = np.exp(logdiv)
        ELFinterp = func(W) / W * 2 / np.pi
        # result[i] = np.trapz(ELFinterp,W)
        result[i+1] = integrate.simps(ELFinterp, W)
        # print([Ee,result])
    result[0] = 0.0
    plt.figure()
    plt.semilogx(Ee, result)
    plt.show()
    print('peff =', result[len(result)-1])
    return result[-1]


def MeanIoniPotential(E,Erg,ELF):
    maxW = np.log10(E)
    W = np.logspace(-1,maxW,1000)
    integrand1 = np.zeros(1000)
    integrand2 = np.zeros(1000)
    if E < 1000:
        func = interpolate.interp1d(Erg, ELF, kind='cubic')
    if E > 1000:
        func = interpolate.interp1d(Erg, ELF, kind='linear')
    for i in range(len(W)):
        integrand2[i] = func(W[i]) * W[i]
        integrand1[i] = integrand2[i] * np.log(W[i])

    integral1 = integrate.simps(integrand1,W)
    integral2 = integrate.simps(integrand2,W)

    return np.exp(integral1/integral2)


def ModelN(parameters, w, data, re_ELF):
    result1 = np.zeros(len(w))
    result2 = np.ones(len(w))
    n = int(len(parameters) / 3)
    an = np.zeros(n)
    omegaPn = np.zeros(n)
    gamman = np.zeros(n)
    for i in range(n):
        str1 = 'A' + str(i)
        str2 = 'OmegaP' + str(i)
        str3 = 'Gamma' + str(i)
        an[i] = parameters[str1].value
        omegaPn[i] = parameters[str2].value
        gamman[i] = parameters[str3].value
    for i in range(n):
        result1 = result1 + Drude(an[i]*omegaPn[i]**2, omegaPn[i], gamman[i], w)
        result2 = result2 - DrudeRe(an[i]*omegaPn[i]**2, omegaPn[i], gamman[i], w)
    return abs(result1 - data) + abs(result2-re_ELF)


def Model0(parameters, w, data):
    result = np.zeros(len(w))
    n = int(len(parameters) / 3)
    an = np.zeros(n)
    omegaPn = np.zeros(n)
    gamman = np.zeros(n)
    for i in range(n):
        str1 = 'A' + str(i)
        str2 = 'OmegaP' + str(i)
        str3 = 'Gamma' + str(i)
        an[i] = parameters[str1].value
        omegaPn[i] = parameters[str2].value
        gamman[i] = parameters[str3].value
    for i in range(n):
        result = result + Drude(an[i]*omegaPn[i]**2, omegaPn[i], gamman[i], w)
    return abs(result - data)


def Model1(parameters, w, data, re_ELF):
    result1 = np.zeros(len(w))
    result2 = np.ones(len(w))
    n = int(len(parameters)/4)
    an = np.zeros(n)
    omegaPn = np.zeros(n)
    gamman = np.zeros(n)
    Ethn = np.zeros(n)
    for i in range(n):
        str1 = 'A' + str(i)
        str2 = 'OmegaP' + str(i)
        str3 = 'Gamma' + str(i)
        str4 = 'Eth' + str(i)
        an[i] = parameters[str1].value
        omegaPn[i] = parameters[str2].value
        gamman[i] = parameters[str3].value
        Ethn[i] = parameters[str4].value
    for i in range(n):
        if Ethn[i]<1e-5:
            result1 = result1 + Drude(an[i], omegaPn[i], gamman[i], w)
            result2 = result2 - DrudeRe(an[i] * omegaPn[i] ** 2, omegaPn[i], gamman[i], w)
        else:
            result1 = result1 + DrudeSigmoid(an[i], omegaPn[i], gamman[i],Ethn[i], w)
            result2 = result2 - DrudeReSigmoid(an[i] * omegaPn[i] ** 2, omegaPn[i], gamman[i], Ethn[i], w)
    return abs(result1 - data) + abs(result2-re_ELF)


def Model2(parameters, w, data):
    result = np.zeros(len(w))
    n = int(len(parameters)/4)
    an = np.zeros(n)
    omegaPn = np.zeros(n)
    gamman = np.zeros(n)
    Ethn = np.zeros(n)
    for i in range(n):
        str1 = 'A' + str(i)
        str2 = 'OmegaP' + str(i)
        str3 = 'Gamma' + str(i)
        str4 = 'Eth' + str(i)
        an[i] = parameters[str1].value
        omegaPn[i] = parameters[str2].value
        gamman[i] = parameters[str3].value
        Ethn[i] = parameters[str4].value
    for i in range(n):
        result = result + DrudeEth(an[i], omegaPn[i], gamman[i],Ethn[i], w)

    #return (result - data)/data
    return result - data


def para_0(arr_a, arr_wP, arr_gamma):
    # 如果初始值未给也可以通过所含的振子个数来默认赋值为1作为振子
    parameter = Parameters()
    N = len(arr_a)
    for i in range(N):
        str1 = 'A'+str(i)
        str2 = 'OmegaP'+str(i)
        str3 = 'Gamma'+str(i)
        parameter.add_many((str1, arr_a[i], True, 0.0, None),
                            (str2, arr_wP[i], True, 0.0, None),
                            (str3, arr_gamma[i], True, 0.0, None))
    return parameter


def para_1(arr_a, a_min, a_max, arr_wP, wP_min, wP_max, arr_gamma, gamma_min, gamma_max):
    # 如果初始值未给也可以通过所含的振子个数来默认赋值为1作为振子
    parameter = Parameters()
    N = len(arr_a)
    for i in range(N):
        str1 = 'A'+str(i)
        str2 = 'OmegaP'+str(i)
        str3 = 'Gamma'+str(i)
        parameter.add_many((str1, arr_a[i], True, a_min[i], a_max[i]),
                            (str2, arr_wP[i], True, wP_min[i], wP_max[i]),
                            (str3, arr_gamma[i], True, gamma_min[i], gamma_max[i]))
    return parameter


def para_2(arr_a,  a_min, a_max, arr_wP, wP_min, wP_max, arr_gamma,gamma_min, gamma_max, arr_Eth, Eth_min, Eth_max):
    # 如果初始值未给也可以通过所含的振子个数来默认赋值为1作为振子
    parameter = Parameters()
    N = len(arr_a)
    for i in range(N):
        str1 = 'A'+str(i)
        str2 = 'OmegaP'+str(i)
        str3 = 'Gamma'+str(i)
        str4 = 'Eth' + str(i)
        parameter.add_many((str1, arr_a[i], True, a_min[i], a_max[i]),
                            (str2, arr_wP[i], True, wP_min[i], wP_max[i]),
                            (str3, arr_gamma[i], True, gamma_min[i], gamma_max[i]),
                           (str4, arr_Eth[i], True, Eth_min[i], Eth_max[i]))
    return parameter


def getfitpara(para):
    n = int(len(para) / 4)
    arr_A = np.zeros(n)
    arr_Gamma = np.zeros(n)
    arr_Omega = np.zeros(n)
    arr_Eth = np.zeros(n)
    # arr_ETH = np.zeros(n)
    # arr_DELTA = np.zeros(n)
    # E = np.logspace(np.log10(Emin),np.log10(Emax),N)
    for i in range(n):
        str1 = 'A' + str(i)
        str2 = 'OmegaP' + str(i)
        str3 = 'Gamma' + str(i)
        str4 = 'Eth' + str(i)
        # str4 = 'Eth' + str(i)
        # str5 = 'Delta' + str(i)
        arr_A[i] = para[str1].value
        arr_Omega[i] = para[str2].value
        arr_Gamma[i] = para[str3].value
        arr_Eth[i] = para[str4].value
        # arr_ETH[i] = para[str4].value
        # arr_DELTA[i] = para[str5].value
    return arr_A, arr_Omega, arr_Gamma, arr_Eth


def log_interp(zz, xx, yy):
    logz = np.log10(zz)
    logx = np.log10(xx)
    logy = np.log10(yy)
    return np.power(10.0, np.interp(logz, logx, logy))


def test0917():
    Z, M, density = ReadData('Au')
    arr_E, arr_imep, arr_reep, arr_ELF = ReadELF('Au')
    ELF_s_0 = np.zeros(len(arr_E))
    for i in range(len(arr_imep)):
        ELF_s_0[i] = arr_imep[i] / ((arr_reep[i] + 1.0) ** 2 + arr_imep[i] ** 2)
    # arr_ai,arr_wpi,arr_yi = ReadParameters('oELF/oELF_para_Au.dat', 3)
    arr_ai, arr_wpi, arr_yi = ReadParameters('oELF/oELF_para_Au.dat', 3)
    arr_ELF_fit = np.zeros(len(arr_E))
    arr_re_ELF = np.ones(len(arr_E))
    arr_re = np.zeros(len(arr_E))
    arr_im = np.zeros(len(arr_E))
    ELF_s_1 = np.zeros(len(arr_E))
    for i, erg in enumerate(arr_E):
        for j in range(len(arr_ai)):
            # if arr_delta[j] < 1e-6:
            #     coef = np.heaviside(erg - arr_Eth[j], 1)
            # else:
            #     coef = 1 / (1 + np.exp(-arr_delta[j] * (erg - arr_Eth[j])))
            arr_ELF_fit[i] += Drude(arr_ai[j] * arr_wpi[j] ** 2, arr_wpi[j], arr_yi[j], erg)
            arr_re_ELF[i] -= DrudeRe(arr_ai[j] * arr_wpi[j] ** 2, arr_wpi[j], arr_yi[j], erg)
        arr_re[i] = arr_re_ELF[i] / (arr_ELF_fit[i] ** 2 + arr_re_ELF[i] ** 2)
        arr_im[i] = arr_ELF_fit[i] / (arr_ELF_fit[i] ** 2 + arr_re_ELF[i] ** 2)
        ELF_s_1[i] = arr_im[i] / ((arr_re[i] + 1.0) ** 2 + arr_im[i] ** 2)
    plt.figure()
    # plt.plot(arr_E, re_ELF, label = 'Re[1/epsilon] database')
    # plt.plot(arr_E, arr_ELF, label = 'Im[1/epsilon] database')
    # plt.plot(arr_E, arr_ELF_fit, label = 'Im[1/epsilon] fitted')
    # plt.plot(arr_E, arr_re_ELF, label='Re[1/epsilon] fitted')
    # plt.semilogx(arr_E, arr_imep, label = 'Im[epsilon]')
    # plt.semilogx(arr_E, arr_reep, label='Re[epsilon]')
    # plt.semilogx(arr_E, arr_im, label = 'Im[epsilon] fitted')
    # plt.semilogx(arr_E, arr_re, label ='Re[epsilon] fitted')
    plt.plot(arr_E, ELF_s_0, label = 'Surface loss function database')
    plt.plot(arr_E, ELF_s_1, label='Surface loss function fitted')
    plt.xlim(0.1, 100)
    plt.legend()
    plt.show()



def plotFittedELF(arr_E, arr_ELF,arr_ELF_re, a_fit, omega_fit, gamma_fit, Eth_fit):
    plt.figure(1)
    plt.plot(arr_E, arr_ELF, label='Im[1/epsilon] exp')
    plt.plot(arr_E,arr_ELF_re, label = 'Re[1/epsilon] exp')
    # plt.xlim(0.1, 1000000)
    # plt.ylim(1e-12, 1.2)

    finalELF = np.zeros(len(arr_E))
    finalReELF = np.ones(len(arr_E))
    oscillators = np.zeros((len(a_fit),len(arr_E)))

    for i in range(len(a_fit)):
        # oscillators[i,:] = Drude(a_fit[i]*omega_fit[i]**2, omega_fit[i], gamma_fit[i], arr_E)
        oscillators[i, :] = DrudeSigmoid(a_fit[i] * omega_fit[i] ** 2, omega_fit[i], gamma_fit[i], Eth_fit[i], arr_E)
        finalELF += oscillators[i,:]
        # finalReELF -= DrudeRe(a_fit[i]*omega_fit[i]**2, omega_fit[i], gamma_fit[i], arr_E)
        finalReELF -= DrudeReSigmoid(a_fit[i] * omega_fit[i] ** 2, omega_fit[i], gamma_fit[i], Eth_fit[i], arr_E)
    plt.plot(arr_E, finalELF, 'g--', label='Im[1/epsilon] fit')
    plt.plot(arr_E, finalReELF, '.', label = 'Re[1/epsilon] fit')
    plt.legend()

    plt.figure(2)
    for i in range(len(a_fit)):
        plt.loglog(arr_E, oscillators[i, :], label='oscillator:' + str(i))
    plt.loglog(arr_E, arr_ELF, label='exp')
    plt.loglog(arr_E, finalELF, 'g--', label='fit')
    # plt.xlim(0.1, 1000000)
    # plt.ylim(1e-12, 1.2)
    plt.legend()
    plt.show()


# main function
if __name__ == "__main__":
    # 分段拟合法

    # Read material data
    # Au(79) 1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 4f14 5s2 5p6 5d10 6s1
    # 80724.9 14352.8 13733.6 11918.7 3424.9 3147.8 2743.0 2291.1 2205.7
    # 758.8 643.7 545.4 352.0 333.9 87.8 84.1 107.8 71.7 58.7 2.5 2.5

    # Ag(47) 1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s1
    # 25514.0 3805.8 3523.7 3351.1 | 717.5 602.4 571.4 372.8 366.7 95.2 62.6 55.9 3.3 3.3

    # Cu(29) 1s2 2s2 2p6 3s2 3p6 3d10 4s1
    # 8978.9 1096.1 951.0 931.1 119.8 73.6 73.6 1.6 1.6

    # Pt(78) 1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 4f14 5s2 5p6 5d9 6s1
    # 78394.8 13880.5 13272.6 11563.8 3297.6 3027.0 2645.3 2201.5 2121.1
    # 724.0 607.6 519.1 330.7 313.8 74.5 71.1 101.7 65.3 51.0 2.1 2.1

    matName = 'Au'
    Z,M,density = ReadData(matName)
    Bind_erg = BindingEnergy(matName)
    arr_E, arr_imep, arr_reep, arr_ELF = ReadELF(matName)
    # arr_ai, arr_wpi, arr_yi = ReadParameters('oELF/oELF_para_Au.dat', 3)
    arr_ELF_re = arr_reep/(arr_imep**2 + arr_reep**2) # Re[1/epsilon] to be fitted
    plt.figure()
    plt.loglog(arr_E, arr_ELF)
    plt.show()


    # test input data
    plotELF(arr_E,arr_ELF,arr_ELF_re, 0.125, 1000, 'semilogx')

    # Two values (ELF and Re[1/]) are needed to be fitted
    Emin = min(arr_E) # eV
    Emax = max(arr_E) # eV
    f_sum = f_sumrule(arr_E, arr_ELF, Emin, Emax, density,M)
    print("f-sum rule: ",f_sum)
    ps_sum = ps_sumrule(arr_E,arr_ELF)
    print("ps-sun rule: ",ps_sum)

    # Generate initial guess
    # ini_A, ini_omegaP, ini_gamma = GenerateIni(arr_E, arr_ELF)



    # parameters for fitting
    # two steps fitting
    # step 1
    # in step 1 both Im[1/epsilon] and Re[1/epsilon] should be fitted
    # for calculations of surface plasmon
    divide = 150.0  # divide the ELF between outer shell electrons and inner shell electrons
    new_E = np.logspace(np.log10(arr_E[0]), np.log10(divide), 1000)
    new_ELF = log_interp(new_E, arr_E, arr_ELF)
    ini_A, ini_omegaP, ini_gamma, ini_Eth = GenerateIni(new_E, new_ELF, mode=1)
    ini_A = ini_A / ini_omegaP ** 2 # ai = Ai/omegaP**2
    print("第一步拟合开始，能量损失函数能量范围为 ",arr_E[0],' eV 到 ',divide, ' eV')
    while True:
        A_min = np.ones(len(ini_A))*1e-5
        omega_min = ini_omegaP * 0.9
        gamma_min = np.zeros(len(ini_A))
        Eth_min = ini_Eth * 0.8
        A_max = np.ones(len(ini_A))
        omega_max = ini_omegaP * 1.1
        gamma_max = np.ones(len(ini_A)) * 50.0
        Eth_max = ini_Eth * 1.2+1e-6
        # plot sum rule
        # partial_fsum = np.zeros(len(ini_A))
        # for i in range(len(ini_A)):
        #     partial_fsum[i] = P_f_sumrule(ini_A[i],ini_omegaP[i],ini_gamma[i],Emin,Emax,density,M)
        # print("f-sum rule of initial guess: ",sum(partial_fsum))
        # plt.figure()
        # plt.loglog(ini_omegaP, partial_fsum)
        # plt.show()

        new_ELF_re = np.interp(new_E, arr_E, arr_ELF_re)
        # paras = para_1(ini_A, A_min, A_max, ini_omegaP, omega_min, omega_max, ini_gamma, gamma_min, gamma_max)
        paras = para_2(ini_A, A_min, A_max, ini_omegaP, omega_min, omega_max,
                       ini_gamma, gamma_min, gamma_max,ini_Eth, Eth_min, Eth_max)
        # methodname = 'leastsq'
        methodname = 'dual_annealing'
        # methodname = 'cg'
        results = minimize(Model1, paras, args=(new_E, new_ELF, new_ELF_re), method=methodname)
        report_fit(results)
        # final = results.residual + new_ELF
        a_fit, omega_fit, gamma_fit, Eth_fit = getfitpara(results.params)
        plotFittedELF(new_E, new_ELF, new_ELF_re, a_fit, omega_fit, gamma_fit,Eth_fit)
        ini_A = a_fit
        ini_omegaP = omega_fit
        ini_gamma = gamma_fit
        ini_Eth = Eth_fit
        shutnow = input('请输入是否终止循环？Y/N')
        if shutnow is 'Y' or shutnow is 'y':
            break

    print('拟合后参数如下')
    print('A Omega Gamma')
    f = open('oELF/step1.txt', 'w')
    for i in range(len(ini_A)):
        print(ini_A[i], ' ', ini_omegaP[i], ' ', ini_gamma[i])
        f.write(str(ini_A[i])+' '+str(ini_omegaP[i])+' '+str(ini_gamma[i])+'\n')
    f.close()

    # plot Re[1/epsilon] again


    #step 2
    """
    """




# peaks_i = argrelextrema(arr_ELF, np.greater)
# plt.loglog(arr_E,arr_ELF)
# plt.loglog(arr_E[peaks_i], arr_ELF[peaks_i], marker = "x")
#
#
# valley_i = argrelextrema(arr_ELF, np.less)
# plt.loglog(arr_E[valley_i], arr_ELF[valley_i], marker = "*")

# plt.show()


# erg = np.logspace(np.log10(arr_E[0]+1e-6),np.log10(arr_E[-1]-1e-6),1000)
# # print(erg)
# arr_diff_ELF = np.zeros(1000)
# for i in range(len(erg)):
#     arr_diff_ELF[i] = derivative(ELFfunc, erg[i], dx=1e-8)
# arr_plus = []
# arr_minus = []
# arr_log_diffELF = np.zeros(1000)
# for i in range(len(erg)):
#     if arr_diff_ELF[i] > 0.0:
#         arr_log_diffELF[i] = np.log10(arr_diff_ELF[i])
#         arr_plus.append(arr_log_diffELF[i])
#     elif arr_diff_ELF[i] < 0.0:
#         arr_log_diffELF[i] = np.log10(-arr_diff_ELF[i])
#         arr_minus.append(arr_log_diffELF[i])
#
# n_max_plus = abs(min(arr_plus))
# print(n_max_plus)
# n_max_minus = abs(min(arr_minus))
# print(n_max_minus)
# for i in range(len(erg)):
#     if arr_diff_ELF[i] > 0.0:
#         arr_log_diffELF[i] = n_max_plus + arr_log_diffELF[i]
#     if arr_diff_ELF[i] < 0.0:
#         arr_log_diffELF[i] = -(n_max_minus + arr_log_diffELF[i])
#
#
# plt.figure()
# # plt.semilogx(erg,arr_diff_ELF)
# plt.semilogx(erg,arr_log_diffELF)
# plt.xlabel('Energy(eV)')
# plt.ylabel('Differential ELF')
# plt.show()
