# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:27:30 2018

@author: tilseb
"""

# =============================================================================
# Imports
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rate = 0.0
R = str(int(rate*100))
scn = 'rs'

# =============================================================================
# Import scenario data
# =============================================================================

path = 'model_input/' + R + '/' + scn

st_df = pd.read_csv(path + '/ship_data.csv', encoding='utf8', index_col=0)
ec_df = pd.read_csv(path + '/ef-co2-w2p.csv', encoding='utf8', index_col=0)
em_df = pd.read_csv(path + '/ef-ch4-w2p.csv', encoding='utf8', index_col=0)
cf_df = pd.read_csv(path + '/cf.csv', encoding='utf8', index_col=0)
ci_df = pd.read_csv(path + '/ci.csv', encoding='utf8', index_col=0)
cs_df = pd.read_csv(path + '/cs.csv', encoding='utf8', index_col=0)
ts_df = pd.read_csv(path + '/ts.csv', encoding='utf8', index_col=0)
td_df = pd.read_csv(path + '/td.csv', encoding='utf8', index_col=0)
ba_df = pd.read_csv(path + '/ba.csv', encoding='utf8', index_col=0)


# =============================================================================
# Import reference data
# =============================================================================

path = 'model_input/' + R + '/' + 'rs'

st_rs = pd.read_csv(path + '/ship_data.csv', encoding='utf8', index_col=0)
ec_rs = pd.read_csv(path + '/ef-co2-w2p.csv', encoding='utf8', index_col=0)
em_rs = pd.read_csv(path + '/ef-ch4-w2p.csv', encoding='utf8', index_col=0)
cf_rs = pd.read_csv(path + '/cf.csv', encoding='utf8', index_col=0)
ci_rs = pd.read_csv(path + '/ci.csv', encoding='utf8', index_col=0)
cs_rs = pd.read_csv(path + '/cs.csv', encoding='utf8', index_col=0)
ts_rs = pd.read_csv(path + '/ts.csv', encoding='utf8', index_col=0)
td_rs = pd.read_csv(path + '/td.csv', encoding='utf8', index_col=0)
ba_rs = pd.read_csv(path + '/ba.csv', encoding='utf8', index_col=0)


# =============================================================================
# Import dictionaries
# =============================================================================

path = 'dictionaries'

dict_fuel = pd.read_csv(path + '/dict_fuel.csv', index_col=0)
dict_fuel = dict_fuel.sort_index()
dict_ship = pd.read_csv(path + '/dict_ship.csv', index_col=0)
dict_ship = dict_ship.sort_index()
dict_result = pd.read_csv(path + '/dict_result.csv', index_col=0)
dict_result = dict_result.sort_index()


# =============================================================================
# Make directories
# =============================================================================

plot_path = 'plots/' + R + '/' + scn

if not os.path.isdir(plot_path):
    os.makedirs(plot_path)


# =============================================================================
# Global plot specifications
# =============================================================================

plt.rcParams.update({'font.size': 40})
plt.rc('legend', fontsize=30)
plt.rcParams['xtick.major.pad'] = '5'
fs = (12, 8)


# =============================================================================
# Plot parameters
# =============================================================================

cf_df = cf_df.loc[:2050] * 1000
ci_df = ci_df.loc[:2050] * 1000
cs_df = cs_df.loc[:2050] * 1000

cf_rs = cf_rs.loc[:2050] * 1000
ci_rs = ci_rs.loc[:2050] * 1000
cs_rs = cs_rs.loc[:2050] * 1000


def func(y):
    return (lambda x: np.asarray(x) * np.asarray(y))


ci_df = ci_df.apply(func(st_df.li_yr), axis=1)
cs_df = cs_df.apply(func(st_df.ls_yr), axis=1)

ci_rs = ci_rs.apply(func(st_rs.li_yr), axis=1)
cs_rs = cs_rs.apply(func(st_rs.ls_yr), axis=1)

ymax = 0
for a in [cf_df, ci_df]:
    if a.max().max() > ymax:
        ymax = a.max().max()
    ymax *= 1.01


#
# Fuel costs
#


rs = cf_rs
rs.columns = rs.columns.str.split('_').str[1]
rs = rs.groupby(rs.columns, axis=1).mean()
rs = rs.rename(columns=dict_fuel.name.to_dict())
rs = rs.sort_index(axis=1)
dict_fuelrs = dict_fuel.sort_values(by=['name'])
crs = list(dict_fuelrs.loc[dict_fuelrs.name.isin(rs.columns)].color)

cf = cf_df
cf.columns = cf.columns.str.split('_').str[1]
cf = cf.groupby(cf.columns, axis=1).mean()
cf = cf.rename(columns=dict_fuel.name.to_dict())
cf = cf.sort_index(axis=1)
dict_fuel = dict_fuel.sort_values(by=['name'])
c = list(dict_fuel.loc[dict_fuel.name.isin(cf.columns)].color)


ax = rs.plot(lw=2, figsize=fs, linestyle='--', color=crs)
cf.plot(ax=ax, lw=5, color=c)

handles = ax.get_legend_handles_labels()[0][len(cf.columns):]
labels = ax.get_legend_handles_labels()[1][len(cf.columns):]

plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.20))
plt.margins(0, 0)
plt.ylim(0, ymax)
plt.grid(axis='x', which='major')
plt.tight_layout()
plt.xlabel('Years')
plt.ylabel('$€_{2016}$/$GJ_{fuel}$')
plt.subplots_adjust(bottom=0.1)
plt.savefig(plot_path +
            '/param_cf_r%s.pdf' % R,
            bbox_inches='tight',
            compression=None,
            transparent=True)


#
# Infrastructure costs multiplied with the lifetime (otherwise only annuity)
#

rs = ci_rs
rs.columns = rs.columns.str.split('_').str[1]
rs = rs.groupby(rs.columns, axis=1).mean()
rs = rs.rename(columns=dict_fuel.name.to_dict())
rs = rs.sort_index(axis=1)
dict_fuelrs = dict_fuel.sort_values(by=['name'])
crs = list(dict_fuelrs.loc[dict_fuelrs.name.isin(rs.columns)].color)

ci = ci_df
ci.columns = ci.columns.str.split('_').str[1]
ci = ci.groupby(ci.columns, axis=1).mean()
ci = ci.rename(columns=dict_fuel.name.to_dict())
ci = ci.sort_index(axis=1)
dict_fuel = dict_fuel.sort_values(by=['name'])
c = list(dict_fuel.loc[dict_fuel.name.isin(ci.columns)].color)

ax = rs.plot(lw=2, figsize=fs, linestyle='--', color=crs)
ci.plot(ax=ax, lw=5, color=c)

handles = ax.get_legend_handles_labels()[0][len(ci.columns):]
labels = ax.get_legend_handles_labels()[1][len(ci.columns):]

plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.20))
plt.margins(0, 0)
plt.ylim(0, ymax)
plt.grid(axis='x', which='major')
plt.tight_layout()
plt.xlabel('Years')
plt.ylabel('$€_{2016}$/$GJ_{fuel}$')
plt.subplots_adjust(bottom=0.1)
plt.savefig(plot_path +
            '/param_ci_r%s.pdf' % R,
            bbox_inches='tight',
            compression=None,
            transparent=True)


#
# Ship costs multiplied with the lifetime (otherwise only annuity)
#

if scn in 'h2':
    clr = ['khaki', 'gold', 'firebrick', 'b', 'b', 'b', 'b', 'olive',
           'firebrick', 'olive', 'orange', 'orange', 'black', 'black', 'black',
           'gray', 'gray']
elif scn in ('lng', 'lng_mp'):
    clr = ['khaki', 'gold', 'b', 'b', 'firebrick', 'b', 'b', 'olive',
           'olive', 'olive', 'orange', 'firebrick', 'black', 'black',
           'black', 'gray', 'gray']
elif scn in ('lbg', 'lbg_mp'):
    clr = ['khaki', 'gold', 'b', 'firebrick', 'b', 'b', 'b', 'olive',
           'olive', 'olive', 'firebrick', 'orange', 'black', 'black',
           'black', 'gray', 'gray']
elif scn in 'nh3':
    clr = ['khaki', 'gold', 'b', 'b', 'b', 'b', 'firebrick', 'firebrick',
           'olive', 'olive', 'orange', 'orange', 'black', 'black', 'black',
           'gray', 'gray']
elif scn in ('ch3oh', 'ch3oh_mp'):
    clr = ['khaki', 'gold', 'b', 'b', 'b', 'firebrick', 'b', 'olive', 'olive',
           'firebrick', 'orange', 'orange', 'black', 'black', 'black', 'gray',
           'gray']
elif scn == 'rs_mp':
    clr = ['khaki', 'gold', 'b', 'firebrick', 'firebrick', 'firebrick', 'b',
           'olive', 'olive', 'firebrick', 'firebrick', 'firebrick', 'black',
           'black', 'black', 'gray', 'gray']
else:
    clr = ['khaki', 'gold', 'b', 'b', 'b', 'b', 'b', 'olive', 'olive', 'olive',
           'orange', 'orange', 'black', 'black', 'black', 'gray', 'gray']

cs = cs_df
cs = cs.groupby(cs.columns, axis=1).max()
cs = cs[cs > 1].dropna(axis=1, how='all').fillna(0)
cs = cs.rename(columns=dict_ship.name.to_dict())
dict_ship = dict_ship.sort_values(by=['name'])
c = list(dict_ship.loc[dict_ship.name.isin(cs.columns)].color)
cs = cs.sort_index(axis=1)
cs = cs.sort_values(by=cs.index[0], axis=1, ascending=False)
cs.plot(figsize=fs, lw=5, color=clr)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20))
plt.margins(0, 0)
plt.ylim(0, 4000)
plt.grid(axis='x', which='major')
plt.tight_layout()
plt.ylabel('$€_{2016}$/$GJ_{fuel}$')
plt.xlabel('Years')
plt.subplots_adjust(bottom=0.1)
plt.savefig(plot_path +
            '/param_cs_r%s.pdf' % R,
            bbox_inches='tight',
            compression=None,
            transparent=True)


#
# Transport supply
#

clr = ['khaki', 'b', 'b', 'b', 'b', 'b', 'gold', 'orange', 'orange', 'orange',
       'orange', 'orange', 'black', 'black', 'black', 'black', 'black', 'gray',
       'gray']


ts = ts_df.loc[:2050, (ts_df != 0).any(axis=0)] * 1000
ts = ts.groupby(ts.columns, axis=1).max()
ts = ts.rename(columns=dict_ship.name.to_dict())
c = list(dict_ship.loc[dict_ship.name.isin(ts.columns)].color)
ts = ts.sort_index(axis=1)
ts = ts.sort_values(by=ts.index[-1], axis=1, ascending=False)
ts.plot(figsize=fs, lw=5, color=clr)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20))
plt.margins(0, 0)
plt.ylim(0, int(ts.max().max()*1.1))
plt.grid(axis='x', which='major')
plt.tight_layout()
plt.ylabel('$tkm$/$MJ_{fuel}$')
plt.xlabel('Years')
plt.subplots_adjust(bottom=0.1)
plt.savefig(plot_path +
            '/param_ts_r%s.pdf' % R,
            bbox_inches='tight',
            compression=None,
            transparent=True)


#
# Transport demand
#

clr = ['b', 'firebrick', 'gold']
dict_td = {'td_noneca_Ttkm': 'Transport demand outside ECAs',
           'td_short_Ttkm': 'Transport demand on short distances',
           'td_total_Ttkm': 'Total transport demand'}

rs = td_rs.loc[:2050] * 1000
rs = rs.rename(columns=dict_td)
rs = rs.sort_index(axis=1)
rs = rs.sort_values(by=rs.index[-1], axis=1, ascending=False)

td = td_df.loc[:2050] * 1000
td = td.rename(columns=dict_td)
td = td.sort_index(axis=1)
td = td.sort_values(by=td.index[-1], axis=1, ascending=False)

ax = rs.plot(lw=2, figsize=fs, linestyle='--', color=clr)
td.plot(ax=ax, figsize=fs, lw=5, color=clr)

handles = ax.get_legend_handles_labels()[0][len(td.columns):]
labels = ax.get_legend_handles_labels()[1][len(td.columns):]

plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.20))
plt.margins(0, 0)
plt.ylim(0, int(td.max().max()*1.1))
plt.grid(axis='x', which='major')
plt.tight_layout()
plt.ylabel('$10^{12} tkm$')
plt.xlabel('Years')
plt.subplots_adjust(bottom=0.1)
plt.savefig(plot_path +
            '/param_td_r%s.pdf' % R,
            bbox_inches='tight',
            compression=None,
            transparent=True)


#
# Carbon dioxide emissions
#

ec = ec_df.loc[:2050, (ec_df != 0).any(axis=0)]
ec = ec[ec.columns[~ec.columns.str.contains('old')]]
#ec.columns = ec.columns.str.split('_').str[1]
#ec = ec.groupby(ec.columns, axis=1).mean()
ec = ec.rename(columns=dict_ship.name.to_dict())
ec = ec.sort_index(axis=1)
dict_ship = dict_ship.sort_values(by=['name'])
c = list(dict_ship.loc[dict_ship.name.isin(ec.columns)].color)
ec.plot(figsize=fs, lw=5, color=c)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20))
plt.margins(0, 0)
plt.ylim(0, int(ec.max().max() * 1.1))
plt.grid(axis='x', which='major')
plt.tight_layout()
plt.xlabel('Years')
plt.ylabel('$g_{CO2e}$/$MJ_{fuel}$')
plt.subplots_adjust(bottom=0.1)
plt.savefig(plot_path +
            '/param_co2_w2p_r%s.pdf' % R,
            bbox_inches='tight',
            compression=None,
            transparent=True)

#
# Meathane emissions
#

rs = em_rs.loc[:2050, (em_rs != 0).any(axis=0)]
rs = rs[rs.columns[~rs.columns.str.contains('old')]]
#rs.columns = rs.columns.str.split('_').str[1]
#rs = rs.groupby(rs.columns, axis=1).mean()
rs = rs.rename(columns=dict_ship.name.to_dict())
rs = rs.sort_index(axis=1)
dict_shiprs = dict_ship.sort_values(by=['name'])
crs = list(dict_shiprs.loc[dict_shiprs.name.isin(rs.columns)].color)

em = em_df.loc[:2050, (em_df != 0).any(axis=0)]
em = em[em.columns[~em.columns.str.contains('old')]]
#em.columns = em.columns.str.split('_').str[1]
#em = em.groupby(em.columns, axis=1).mean()
em = em.rename(columns=dict_ship.name.to_dict())
em = em.sort_index(axis=1)
dict_ship = dict_ship.sort_values(by=['name'])
c = list(dict_ship.loc[dict_ship.name.isin(em.columns)].color)

ax = rs.plot(figsize=fs, lw=2, linestyle='--', color=crs)
em.plot(ax=ax, lw=5, color=c)

handles = ax.get_legend_handles_labels()[0][len(em.columns):]
labels = ax.get_legend_handles_labels()[1][len(em.columns):]

plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.20))
plt.margins(0, 0)
#plt.ylim(0, int(ec.max().max() * 1.1))
plt.grid(axis='x', which='major')
plt.tight_layout()
plt.xlabel('Years')
plt.ylabel('$g_{CO2e}$/$MJ_{fuel}$')
plt.subplots_adjust(bottom=0.1)
plt.savefig(plot_path +
            '/param_ch4_w2p_r%s.pdf' % R,
            bbox_inches='tight',
            compression=None,
            transparent=True)

#
# Biofuel availability
#

ba = ba_df.loc[:2050] * 1000
ba.columns = ['Biofuel Availablity']
ba.plot(figsize=fs, lw=5, color='darkgreen')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20))
plt.margins(0, 0)
plt.ylim(0, int(ba.max() * 1.1))
plt.grid(axis='x', which='major')
plt.tight_layout()
plt.xlabel('Years')
plt.ylabel('$MJ$')
plt.subplots_adjust(bottom=0.1)
plt.savefig(plot_path +
            '/param_ba_r%s.pdf' % R,
            bbox_inches='tight',
            compression=None,
            transparent=True)


# =============================================================================
# Import model results
# =============================================================================

results = pd.read_csv('model_output/' + R + '/results_%s.csv' % scn,
                      encoding='utf8',
                      index_col=[0, 1])
results = results.round(10)
results = results.fillna(0)
results.index.rename(['', ''], inplace=True)
results = results.loc[2016:2050]
results_unstacked = results.unstack()

# Refit options
RO = list(zip(list(st_df[st_df.refit == 'yes'].index),
              st_df[st_df.refit == 'yes'].refit_option))

for s, r in RO:
    results_unstacked.fai_cap[s] -= results_unstacked.fai_cap[r]
    results_unstacked.fas_cap[s] -= results_unstacked.fas_cap[r]
results = results_unstacked.stack()

# =============================================================================
# Plot results
# =============================================================================

#
# Infrastructure capacity
#

ic = results_unstacked.fai_cap
ic = ic.groupby(ic.columns, axis=1).sum()
ic = ic[ic > 0.001].dropna(axis=1, how='all').fillna(0)
ic = ic.rename(columns=dict_ship.name.to_dict())
ic = ic.sort_index(axis=1)
dict_ship = dict_ship.sort_values(by=['name'])
clr = list(dict_ship.loc[dict_ship.name.isin(ic.columns)].color)
ic.plot(kind='area', stacked=True, lw=0, figsize=fs, color=clr)
plt.xlabel('Years')
plt.ylabel('PJ')
plt.xticks(np.arange(2020, 2060, 10), rotation=0)
plt.margins(0, 0)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20))
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.savefig(plot_path +
            '/res_ic_r%s.pdf' % R,
            bbox_inches='tight',
            compression=None,
            transparent=True)


#
# Ship capacity
#

sc = results_unstacked.fas_cap
sc = sc.groupby(sc.columns, axis=1).sum()
sc = sc[sc > 0.001].dropna(axis=1, how='all').fillna(0)
sc = sc.rename(columns=dict_ship.name.to_dict())
sc = sc.sort_index(axis=1)
dict_ship = dict_ship.sort_values(by=['name'])
clr = list(dict_ship.loc[dict_ship.name.isin(sc.columns)].color)
sc.plot(kind='area', stacked=True, lw=0, figsize=fs, color=clr)
plt.xlabel('Years')
plt.ylabel('PJ')
plt.xticks(np.arange(2020, 2060, 10), rotation=0)
plt.margins(0, 0)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20))
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.savefig(plot_path +
            '/res_sc_r%s.pdf' % R,
            bbox_inches='tight',
            compression=None,
            transparent=True)


#
# Infrastructure utilization per year
#

fa = results_unstacked.fa
fa = fa.groupby(fa.columns, axis=1).sum()
fa = fa[fa > 0.5].dropna(axis=1, how='all').fillna(0)
fa = fa.rename(columns=dict_ship.name.to_dict())
fa = fa.sort_index(axis=1)
dict_ship = dict_ship.sort_values(by=['name'])
clr = list(dict_ship.loc[dict_ship.name.isin(fa.columns)].color)

w = 1
ax = ic.plot(kind='bar', alpha=0.5, color=clr, width=w)
fa.plot(ax=ax, kind='bar', width=w, color=clr, figsize=fs)

handles = ax.get_legend_handles_labels()[0][len(ic.columns):]
labels = ax.get_legend_handles_labels()[1][len(ic.columns):]

plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.20))
plt.xlabel('Years')
plt.ylabel('$PJ_{fuel}$')
plt.xticks(np.arange(4, 44, 10), np.arange(2020, 2060, 10), rotation=0)
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.savefig(plot_path +
            '/res_infrastructure_utilization_r%s.pdf' % R,
            bbox_inches='tight',
            compression=None,
            transparent=True)


#
# Fuel fo transport demand
#

tf = results_unstacked.fa
# tf = tf.div(tf.sum(axis=1), axis=0)
# tf *= 100
# tf.columns = tf.columns.str.split('_', 1).str[1]
# tf = tf.groupby(tf.columns, axis=1).sum()
tf = tf.rename(columns=dict_ship.name.to_dict())
tf = tf[tf > 0.1].dropna(axis=1, how='all').fillna(0)
tf = tf.sort_index(axis=1)
dict_ship = dict_ship.sort_values(by=['name'])
clr = list(dict_ship.loc[dict_ship.name.isin(tf.columns)].color)
# x = clr
# clr = x[0:4] + x[5:6] + x[4:5] + x[6:]
# x = tf.columns.tolist()
# tf = tf[x[0:4] + x[5:6] + x[4:5] + x[6:]]
tf.plot(kind='bar', stacked=True, figsize=fs, color=clr, width=.9)
plt.xlabel('Years')
plt.ylabel('$PJ_{fuel}$')
plt.xticks(np.arange(4, 44, 10), tf.index[4::10], rotation=0)
plt.ylim(0, int(tf.sum(axis=1).max() * 1.01))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20))
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.savefig(plot_path +
            '/res_fuel_for_transport_r%s.pdf' % R,
            bbox_inches='tight',
            compression=None,
            transparent=True)


#
# System costs
#

ccc = results.columns[:3]
c_sys = results.loc[:, results.columns.isin(ccc)]
c_sgl = c_sys
c_sgl = c_sgl.rename(columns=dict_result.name.to_dict())
c_sgl = c_sgl[c_sgl > 0.001].dropna(axis=1, how='all').fillna(0)
c_sys = c_sys.reset_index(level=0).rename(columns={'': 'year'})
c_sys = c_sys.groupby(['year']).agg('sum')
c_sys = c_sys.rename(columns=dict_result.name.to_dict())
dict_result = dict_result.sort_values(by=['name'])
clr = list(dict_result.loc[dict_result.name.isin(c_sys.columns)].color)
c_sys.plot(kind='bar', stacked=True, figsize=fs, color=clr, width=0.9)
ymax = c_sys.sum(axis=1).max() * 1.01
plt.ylim(0, 15)
plt.xlabel('Years')
plt.ylabel('$10^9 €_{2016}$')
plt.xticks(np.arange(4, 44, 10), c_sys.index[4::10], rotation=0)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20))
plt.margins(0, 0)
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.savefig(plot_path +
            '/res_system_costs_r%s.pdf' % R,
            bbox_inches='tight',
            compression=None,
            transparent=True)


#
# Emissions
#

eee = results.columns[results.columns.str.contains('E')]
e_sys = results.loc[:, results.columns.isin(eee)]
e_sys /= 10**3
e_sgl = e_sys
e_sgl = e_sgl.rename(columns=dict_result.name.to_dict())
e_sgl = e_sgl[e_sgl > 0.0001].dropna(axis=1, how='all').fillna(0)
e_sys = e_sys[e_sys > 0.0001].dropna(axis=1, how='all').fillna(0)
e_sys = e_sys.reset_index(level=0).rename(columns={'': 'year'})
e_sys = e_sys.groupby(['year']).agg('sum')
e_sys = e_sys.rename(columns=dict_result.name.to_dict())
e_sys = e_sys.sort_index(axis=1)
dict_result = dict_result.sort_values('name')
clr = list(dict_result.loc[dict_result.name.isin(e_sys.columns)].color)
e_sys.plot(kind='area', stacked=True, figsize=fs, color=clr, lw=0)
ymax = e_sys.sum(axis=1).max() * 1.01
plt.ylim(0, ymax)
plt.xlabel('Years')
plt.ylabel('$Mt_{CO2e}$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20))
plt.margins(0, 0)
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.savefig(plot_path +
            '/res_system_emissions_r%s.pdf' % R,
            bbox_inches='tight',
            compression=None,
            transparent=True)
