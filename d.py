import scipy.interpolate
import pandas as pd
from pandas import *
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from numpy import linalg as LA

# read the data file
xls = pd.ExcelFile('/Users/zebinxu/Desktop/data APM466.xlsx')
ab1 = pd.read_excel(xls, '1.2')
ab2 = pd.read_excel(xls, '1.3')
ab3 = pd.read_excel(xls, '1.6')
ab4 = pd.read_excel(xls, '1.7')
ab5 = pd.read_excel(xls, '1.8')
ab6 = pd.read_excel(xls, '1.9')
ab7 = pd.read_excel(xls, '1.10')
ab8 = pd.read_excel(xls, '1.13')
ab9 = pd.read_excel(xls, '1.14')
ab10 = pd.read_excel(xls, '1.15')
ab = [ab1, ab2, ab3, ab4, ab5, ab6, ab7, ab8, ab9, ab10]

def ttm(q):
    curr_date = []
    curr_date.append((x.columns.values)[0])
    q['time to maturity'] = [(maturity - current_date).days in q['maturity']]
def ytm(q):
    yr = []

    timer = []
    curr_date = []
    current_date.append((q.columns.values)[0])
    for i, bond in q.iterrows():
        ttm = bond['time to maturity']
        timer.append(ttm / 365)
        y = int(ttm / 182)
        initial = (ttm % 182) / 365
        time = np.asarray([2 * initial + n for n in range(0, y + 1)])
        coupon = bond['coupon'] * 100
        dirty_price = bond['close price'] + coupon * ((182 - ttm % 182) / 365)
        payment = np.asarray([coupon / 2] * y + [coupon / 2 + 100])
        ytm_func = lambda y: np.dot(payment, (1 + y / 2) ** (-time)) - dirty_price
        ytm = optimize.fsolve(ytm_func, .04)
        yr.append(ytm)
    return timer, yr

plt.xlabel('time-to-maturity')
plt.ylabel('yield-to-maturity')
plt.title('5-year yield curve')
labels = ['Jan 2', 'Jan 3', 'Jan 6', 'Jan 7', 'Jan 8' 'Jan 9', 'Jan 10', 'Jan 13', 'Jan 14', 'Jan 15']
i = 0
for a in ab:
    ttm(a)
    plt.plot(ytm(a)[0], ytm(a)[1], label=labels[i])
    i = i + 1
plt.legend()
plt.savefig('/Users/zebinxu/Desktop/br.png')
plt.show()


def spot(x):
    sp = np.empty([1, 10])
    timer = []
    coupons = []
    dirty_price = []
    for i, bond in x.iterrows():
        ttm = bond['time to maturity']
        timer.append(ttm / 365)
        coupon = (bond['coupon']) * 100/2
        coupons.append(coupon)
        dirty_price.append(bond['close price'] + coupon * (0.5 - (ttm % 182) / 365))

    for i in range[1, 10]:
        pmt = np.asarray([coupons[i]] * i + [coupons[i] + 100])
        spot_func = lambda y: np.dot(pmt[:-1], np.exp(-(np.multiply(sp[0, :i], timer[:i])))) + pmt[i] * np.exp(-y * timer[i]) - \
                                  dirty_price[i]
        sp[0, i] = optimize.fsolve(spot_func, .04)
    return timer, sp


labels = ['Jan 2','Jan 3','Jan 6','Jan 7','Jan 8',
         'Jan 9','Jan 10','Jan 13','Jan 14','Jan 15']
plt.xlabel('time to maturity')
plt.ylabel('spot rate')
plt.title('5-year spot curve')
i = 0
for a in ab:
    ttm(a)
    plt.plot(spot(a)[0], spot(a)[1].squeeze(), label = labels[i])
    i = i+1
plt.legend()
plt.savefig()
plt.show()
def forward(x):
    s = (np.asarray(spot(d)[0]),np.asarray(spot(d)[1]).squeeze())
    f = [(s[1][3] * 2 - s[1][1] * 1) / (2 - 1), (s[1][5] * 3 - s[1][1]* 1) / (3 - 1), (s[1][7] * 4 - s[1][1] * 1) / (4 - 1)\
        ,(s[1][9] * 5 - s[1][1] * 1) / (5 - 1)]
    return f

plt.xlabel('year to year')
plt.ylabel('forward rate')
labels = ['Jan 2', 'Jan 3', 'Jan 6', 'Jan 7', 'Jan 8',
          'Jan 9', 'Jan 10', 'Jan 13', 'Jan 14', 'Jan 15']
plt.title('1-year forward curve')
i = 0
for d in ab:
    plt.plot(['1yr-1yr','1yr-2yr','1yr-3yr','1yr-4yr'], forward(d), label = labels[i])
    i = i+1
plt.legend()
plt.savefig('/Users/zebinxu/Desktop/br.png')
plt.show()

log = np.empty([5,9])
y = np.empty([5,10])
for i in range(len()):
    ttm(ab[i])
    y[0,i] = ytm(ab[i])[1][1]
    y[1,i] = ytm(ab[i])[1][3]
    y[2,i] = ytm(ab[i])[1][5]
    y[3,i] = ytm(ab[i])[1][7]
    y[4,i] = ytm(ab[i])[1][9]
for i in range(0, 9):
    log[0, i] = np.log(y[0,i+1]/y[0,i])
    log[1, i] = np.log(y[1,i+1]/y[1,i])
    log[2, i] = np.log(y[2,i+1]/y[2,i])
    log[3, i] = np.log(y[3,i+1]/y[3,i])
    log[4, i] = np.log(y[4,i+1]/y[4,i])

print(np.cov(log))
forwa = np.empty([4,10])
for i in range(len(ab)):
    forwa[0,i] = forward(ab[i])[0]
    forwa[1,i] = forward(ab[i])[1]
    forwa[2,i] = forward(ab[i])[2]
    forwa[3,i] = forward(ab[i])[3]
print(np.cov(forwa))
m, n = LA.eig(np.cov(log))
print(m)
print(n)
b,bb = LA.eig(np.cov(forwa))
print(b)
print(bb)

