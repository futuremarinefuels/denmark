# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 15:21:08 2018

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

R = ['40', '30', '20', '10', '0', '-10', '-20', '-30', '-40', '-50',
     '-60', '-70', '-80', '-90', '-99']
REG = ['rs', 'bau', 'rs_mp', 'snox', 'imo', 'tdvar']

# =============================================================================
# Make directories
# =============================================================================

if not os.path.isdir('plots'):
    os.makedirs('plots')

if not os.path.isdir('tables'):
    os.makedirs('tables')


# =============================================================================
# Import Scenario Results
# =============================================================================

res = []
for r in R:
    results_path = 'model_output/' + r
    all_files = glob.glob(results_path + '/*.csv')
    l = []
    for f in all_files:
        df = pd.read_csv(f, encoding='utf8')
        scn = f.split('_', 2)[2].split('.')[0]
        df['Scenario'] = scn
        df.rename(columns={'Unnamed: 0': 'year', 'Unnamed: 1': 'ship_type'},
                  inplace=True)
        df['rate'] = r
        df.set_index('Scenario', inplace=True)
        l.append(df)
    l = pd.concat(l)
    res.append(l)
res = pd.concat(res)


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
dict_scnfuel = pd.read_csv(path + '/dict_scn2fuel.csv', index_col=0)
dict_scnfuel = dict_scnfuel.fuel.to_dict()


# =============================================================================
# Global plot specifications
# =============================================================================

plt.rcParams.update({'font.size': 30})
plt.rc('legend', fontsize=30)
plt.rcParams['xtick.major.pad'] = '5'
fs = (12, 8)
bw = 0.75


res['fuel'] = res.ship_type.str.split('_').str[1]

# Cost Rates
cr = res.loc[~res.index.isin(REG), ['fuel', 'fa', 'rate']]
fa_scnrate = cr.groupby([cr.index, 'rate'])['fa'].sum()
cr['scnfuel'] = cr.index
cr.scnfuel.replace(dict_scnfuel, inplace=True)
cr = cr[cr.fuel == cr.scnfuel]
cr = cr.groupby([cr.index, 'rate']).sum()
cr['fa_share'] = cr.fa * 100 / fa_scnrate
del cr['fa']
cr = cr.unstack().round(1)
cr.loc['rate'] = cr.columns.levels[1]
cr.loc['rate'] = cr.loc['rate'].astype(int)
cr = cr.T.sort_values('rate', ascending=False).T
cr = cr.drop('rate')
cr.to_csv('tables/fuels_per_rate.csv', encoding='utf8')

# Plot cr
cr = cr.T
cr.index = cr.index.droplevel(level=0)
cr = cr.rename(columns=dict_scn.name.to_dict())
cr = cr.sort_index(axis=1)
dict_scn = dict_scn.sort_values(by=['name'])
c = list(dict_scn.loc[dict_scn.name.isin(cr.columns)].color)
style = ['-', '-', '-', '-', '-', '-.', '-', ':', '-']
cr.plot(figsize=fs, lw=3, color=c, style=style)
plt.axvline(4, label='REF', color='black', linestyle='--', lw=3)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.margins(0, 0)
plt.grid(axis='x', which='major')
plt.tight_layout()
plt.ylabel('Fuel Share [%]')
plt.ylim(0, cr.max().max() * 1.1)
plt.xticks(np.arange(0, len(cr.index), 1), cr.index)
plt.xlabel('Cost rate [%]')
plt.subplots_adjust(bottom=0.1)
plt.savefig('plots/fuels_per_rate.pdf',
            bbox_inches='tight',
            compression=None,
            transparent=True)


# Regulations
rr = res.loc[res.index.isin(REG), ['fuel', 'fa', 'rate']]
rr = rr.groupby([rr.index, 'fuel']).sum().unstack().T
rr *= 100 / rr.sum(axis=0)
rr.index = rr.index.set_levels(['fa_share'], level=0)
rr = rr.round(1)
rr = rr.T.sort_values(('fa_share', 'hfo'), ascending=False).T
rr.index = rr.index.droplevel(level=0)
rr.to_csv('tables/fuels_per_regs.csv', encoding='utf8')

# Plot rr
rr = rr.T
rr = rr.rename(index=dict_scn.name.to_dict())
rr = rr.rename(columns=dict_fuel.name.to_dict())
rr = rr.sort_index(axis=1)
rr = rr.loc[:, (rr != 0).any(axis=0)]
dict_fuel = dict_fuel.sort_values(by=['name'])
c = list(dict_fuel.loc[dict_fuel.name.isin(rr.columns)].color)
rr.plot(figsize=fs, width=bw, color=c, kind='bar', stacked=True)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12))
plt.margins(0, 0)
plt.ylabel('Fuel Share [%]')
plt.xlabel('Scenarios')
plt.xticks(rotation=0)
plt.subplots_adjust(bottom=0.1)
plt.savefig('plots/fuels_per_regs.pdf',
            bbox_inches='tight',
            compression=None,
            transparent=True)
