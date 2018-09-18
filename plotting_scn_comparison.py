# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 18:32:56 2018

@author: tilseb
"""

# =============================================================================
# Imports
# =============================================================================

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Make directories
# =============================================================================

plot_path = 'plots/'

if not os.path.isdir(plot_path):
        os.makedirs(plot_path)


# =============================================================================
# Import Scenario Results
# =============================================================================

results_path = 'model_output/selected/'

all_files = glob.glob(results_path + '/*.csv')

l = []
for f in all_files:
    df = pd.read_csv(f, encoding='utf8')
    scn = f.split('_', 2)[2].split('.')[0]
    df['Scenario'] = scn
    df.rename(columns={'Unnamed: 0': 'year', 'Unnamed: 1': 'ship_type'},
              inplace=True)
    #df['rate'] = R
    #df.rate = scn.rsplit('_', 1)[1]
    df.set_index('Scenario', inplace=True)
    l.append(df)
res = pd.concat(l)


# =============================================================================
# Import dictionaries
# =============================================================================

path = 'dictionaries'

dict_fuel = pd.read_csv(path + '/dict_fuel.csv', index_col=0)
dict_fuel = dict_fuel.sort_values(by='name')
dict_ship = pd.read_csv(path + '/dict_ship.csv', index_col=0)
dict_ship = dict_ship.sort_values(by='name')
dict_result = pd.read_csv(path + '/dict_result.csv', index_col=0)
dict_result = dict_result.sort_index()
dict_scn = pd.read_csv(path + '/dict_scn.csv', index_col=0)
dict_scn = dict_scn.sort_values(by='name')


# =============================================================================
# Global plot specifications
# =============================================================================

plt.rcParams.update({'font.size': 30})
plt.rc('legend', fontsize=30)
plt.rcParams['xtick.major.pad'] = '5'
fs = (12, 8)
bw = 0.5


# =============================================================================
# System Costs
# =============================================================================

ccc = ['CF', 'CI', 'CS']
c = res[ccc]
c = c.groupby(c.index).sum()
c = c.rename(columns=dict_result.name.to_dict())
c = c.rename(index=dict_scn.name.to_dict())
c = c.sort_index(axis=1)
clr = list(dict_result.loc[dict_result.name.isin(c.columns)].color)
c.plot(kind='bar', stacked=True, figsize=fs, color=clr)
plt.hlines(c.loc['BAU'].sum(), -.5, len(c.index) + 1, lw=2.5,
           label='BAU, total', linestyles='--')
ymax = int(c.sum(axis=1).max() * 1.1)
plt.ylim(0, ymax)
plt.ylabel('$10^9 €_{2016}$')
plt.yticks(np.arange(0, ymax, 5))
plt.xlabel('Scenarios')
plt.xticks(rotation=90)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.50))
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)

# Safe figure
plt.savefig(plot_path + '/scn_costs.pdf',
            bbox_inches='tight',
            compression=None,
            transparent=True)

# =============================================================================
# Emissions
# =============================================================================

eee = ['EC', 'EM']
e = res[eee]
e = e.groupby(e.index).sum()
e /= 10**3
e = e.rename(columns=dict_result.name.to_dict())
e = e.rename(index=dict_scn.name.to_dict())
e = e.sort_index(axis=1, ascending=False)
clr = list(dict_result.loc[dict_result.name.isin(e.columns)].color)
e.plot(kind='bar', stacked=True, figsize=fs, color=clr, width=bw)
plt.hlines(e.loc['REF'].sum(), -bw, len(e.index) + bw, lw=2.5,
           label='Emission Budget', linestyles='--')
plt.xlabel('Scenarios')
plt.xticks(rotation=90)
plt.ylabel('$Mt_{CO2e}$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.50))
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)

# Safe figure
plt.savefig(plot_path + '/scn_emissions.pdf',
            bbox_inches='tight',
            compression=None,
            transparent=True)


# =============================================================================
# Fuels
# =============================================================================


f = res.groupby([res.index, 'ship_type'])['fa'].sum()
f = f[f > 1]
f = f.unstack()
f = f.rename(columns=dict_ship.name.to_dict())
f = f.rename(index=dict_scn.name.to_dict())
f = f.sort_index(axis=1)
clr = list(dict_ship.loc[dict_ship.name.isin(f.columns)].color)
f.plot(kind='bar', width=bw, stacked=True, figsize=fs, color=clr)
plt.hlines(f.loc['REF'].sum(), -bw, len(f.index) + bw, lw=2.5,
           label='REF, total', linestyles='--')
plt.ylabel('$PJ_{fuel}$')
plt.xlabel('Scenarios')
plt.xticks(rotation=90)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.50))
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)

# Safe figure
plt.savefig(plot_path + '/scn_fuels.pdf',
            bbox_inches='tight',
            compression=None,
            transparent=True)
