# -*- coding: utf-8 -*-
"""
Future Marine Fuels

A investment model to calculate the minimum system costs of possilbe climate
compatible energy pathways. In this case for the Danish maritime cargo sector.


Copyright (C) 2018 Till ben Brahim

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or any
    later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You may obtain a copy of the License at http://www.gnu.org/licenses/.


@author: tilseb@dtu.dk

"""

# =============================================================================
# Import required packages
# =============================================================================

from pyomo.environ import (ConcreteModel, Var, Objective, Constraint, minimize,
                           NonNegativeReals)
from pyomo.opt import SolverFactory
import pandas as pd
import os


# =============================================================================
# Choose a scenario
# =============================================================================

# Reference scenario: rs
# Regulation scenarios: bau, ch4leak, snox, imo, tdvar
# Cost scenarios: batwind, bdo, ch3oh, lbg, lng, ch3oh_mp, lbg_mp, lng_mp, h2,
#                 nh3

R = '0'
scn = 'rs'


# =============================================================================
# Import data
# =============================================================================

path = 'model_input/' + R + '/' + scn

st_df = pd.read_csv(path + '/ship_data.csv', encoding='utf8', index_col=0)
ec_df = pd.read_csv(path + '/ef-co2-w2p.csv', encoding='utf8', index_col=0)
em_df = pd.read_csv(path + '/ef-ch4-w2p.csv', encoding='utf8', index_col=0)
cf_df = pd.read_csv(path + '/cf.csv', encoding='utf8', index_col=0)
ci_df = pd.read_csv(path + '/ci.csv', encoding='utf8', index_col=0)
cs_df = pd.read_csv(path + '/cs.csv', encoding='utf8', index_col=0)
td_df = pd.read_csv(path + '/td.csv', encoding='utf8', index_col=0)
ts_df = pd.read_csv(path + '/ts.csv', encoding='utf8', index_col=0)
ba_df = pd.read_csv(path + '/ba.csv', encoding='utf8', index_col=0)
reg = pd.read_csv(path + '/regs.csv', encoding='utf8', index_col=0)


# =============================================================================
# Scenario regulations
# =============================================================================

# Co2e emission budget [kt_co2e]
eb = reg[scn].eb_kt

# Co2e emnission target as a share of todays emissions [m/m]
et = reg[scn].et

# Co2e emnission target year [year]
et_yr = int(reg[scn].et_yr)

# Sox limit in emission control areas [m/m]
sl_eca = reg[scn].sl_eca

# Sox limit outside emission control areas [m/m]
sl_noneca = reg[scn].sl_noneca

# Sox limit year outside emission control areas [year]
sl_noneca_yr = int(reg[scn].sl_noneca_yr)

# Minimum level of Tier required in emission control areas [-]
tl_eca = reg[scn].tl_eca

# Year of Tier introduction to baltic and north sea ecas [year]
tl_eca_yr = int(reg[scn].tl_eca_yr)


# =============================================================================
# Create model
# =============================================================================

m = ConcreteModel(name=scn)


# =============================================================================
# Create model data
# =============================================================================

#
# Original sets
#

# Timesteps
T = list(cf_df.index)

# Ship types
S = list(st_df.index)


#
# Parameters
#

# Fuel amount in first year for each ship type [PJ]
fa_init = {(T[0], s): st_df.fa_PJ[s] for s in S}

# Biofuel availability in each timestep [PJ]
ba = ba_df['ba_PJ'].to_dict()

# Fuel costs per ship type and year [EUR/MJ fuel]
cf = {(t, s): cf_df.at[t, s] for t in T for s in S}

# Infrastructure costs per ship type and year [EUR/MJ fuel]
ci = {(t, s): ci_df.at[t, s] for t in T for s in S}

# Ship costs per ship type and year [EUR/MJ fuel]
cs = {(t, s): cs_df.at[t, s] for t in T for s in S}

# Co2 emissions per ship type [g/MJ fuel]
ec = {(t, s): ec_df.at[t, s] for t in T for s in S}

# Ch4 emissions per ship type [g/MJ fuel]
em = {(t, s): em_df.at[t, s] for t in T for s in S}

# Total transport demand for each year [10^12 tkm]
td_total = dict(zip(td_df.index, td_df.td_total_Ttkm))

# Transport demand on the short range for each year [10^12 tkm]
td_short = dict(zip(td_df.index, td_df.td_short_Ttkm))

# Transport demand in the emission control areas [10^12 tkm]
td_eca = dict(zip(td_df.index, td_df.td_short_Ttkm))

# Transport demand outside the emission control areas [10^12 tkm]
td_noneca = dict(zip(td_df.index, td_df.td_noneca_Ttkm))

# Transport supply per shiptype and year [10^3 tkm/MJ fuel]
ts = {(t, s): ts_df.at[t, s] for t in T for s in S}

# Fuel infrastructure lifetime per ship type [year]
li = dict(zip(st_df.index, st_df.li_yr))

# Ship lifetime per ship type [year]
ls = dict(zip(st_df.index, st_df.ls_yr))


#
# Special sets
#

# Old ships
OS = list(st_df.index[st_df.index.str.contains('old')])

# New ships
NS = list(set(S) - set(OS))

# Refit ship types
RS = list(set(st_df[st_df.refit == 'yes'].refit_option))

# Refit options
RO = list(zip(list(st_df[st_df.refit == 'yes'].index),
              st_df[st_df.refit == 'yes'].refit_option))

# Ships for short range operation
SR = list(st_df.index[st_df.range == 'short'])

# Ships not in compliance with ecas sox regulations
SNE = list(set(st_df[st_df['sox_g-g'] > sl_eca].index) &
           set(st_df[st_df.scrubber == 'no'].index))

# Ships not in compliance with global sox limit
SNG = list(set(st_df[st_df['sox_g-g'] > sl_noneca].index) &
           set(st_df[st_df.scrubber == 'no'].index))

# Ships not in compliance with Tier level
NT = list(st_df.index[st_df.tier_level < tl_eca])

# Ships running on biodiesel oil
SB = list(st_df.index[st_df.index.str.contains('bdo')])


#
# Variables
#

# Fuel amount per year and ship type [PJ]
m.fa = Var(T, S, within=NonNegativeReals)

# Cost-effective fuel amount of each ship types infrastructure per year [PJ]
m.fai_up = Var(T, S, within=NonNegativeReals)

# Sum of added fuel of each ship types infrastructure lifetime per year [PJ]
m.fai_cap = Var(T, S, within=NonNegativeReals)

# Cost-effective fuel amount of each ship types new-build/refit per year [PJ]
m.fas_up = Var(T, S, within=NonNegativeReals)

# Sum of added fuel of each ship types new-build/refit lifetime per year [PJ]
m.fas_cap = Var(T, S, within=NonNegativeReals)


# Total fuel costs per year and ship type [10^9 €]
m.CF = Var(T, S, bounds=(0, None))

# Total infrastructure costs per year and ship type [10^9 €]
m.CI = Var(T, S, bounds=(0, None))

# Total ship costs per year and ship type [10^9 €]
m.CS = Var(T, S, bounds=(0, None))

# Co2 emissions per year and ship type [10^9 g]
m.EC = Var(T, S, within=NonNegativeReals)

# Ch4 emissions per year and ship type [10^9 g]
m.EM = Var(T, S, within=NonNegativeReals)


# =============================================================================
# Construct equations
# =============================================================================

#
# Initializations
#

def init_fa_os_rule(m, t, s):
    """
    Returns the initial fuel amount for the old ships. [PJ]
    """
    return (m.fa[t, s] - fa_init[t, s] == 0)


m.init_fa_os = Constraint(T[0:1], OS, rule=init_fa_os_rule)


def init_fa_ns_rule(m, t, s):
    """
    Returns the initial fuel amount for the new ships. [PJ]
    """
    return (m.fa[t, s] - fa_init[t, s] <= 0)


m.init_fa_ns = Constraint(T[0:1], NS, rule=init_fa_ns_rule)


def init_fai_up_os_rule(m, t, s):
    """
    Returns the initially added amount of infrastrucutre for old ships. [PJ]
    """
    return (m.fai_up[t, s] - fa_init[t, s] >= 0)


m.init_fai_up_os = Constraint(T[0:1], OS, rule=init_fai_up_os_rule)


def init_fai_up_ns_rule(m, t, s):
    """
    Returns the initially added amount of infrastrucutre for new ships. [PJ]
    """
    return (m.fai_up[t, s] - fa_init[t, s] <= 0)


m.init_fai_up_ns = Constraint(T[0:1], NS, rule=init_fai_up_ns_rule)


def init_fas_up_os_rule(m, t, s):
    """
    Returns the initially added amount of ships for old ships. [PJ]
    """
    return (m.fas_up[t, s] - fa_init[t, s] >= 0)


m.init_fas_up_os = Constraint(T[0:1], OS, rule=init_fas_up_os_rule)


def init_fas_up_ns_rule(m, t, s):
    """
    Returns the initially added amount of ships for new ships. [PJ]
    """
    return (m.fas_up[t, s] - fa_init[t, s] <= 0)


m.init_fas_up_ns = Constraint(T[0:1], NS, rule=init_fas_up_ns_rule)


def init_fai_cap_os_rule(m, t, s):
    """
    Returns the initial infrastructure capacity for old ships. [PJ]
    """
    return (m.fai_cap[t, s] - fa_init[t, s] >= 0)


m.init_fai_cap_os = Constraint(T[0:1], OS, rule=init_fai_cap_os_rule)


def init_fai_cap_ns_rule(m, t, s):
    """
    Returns the initial infrastructure capacity for new ships. [PJ]
    """
    return (m.fai_cap[t, s] - fa_init[t, s] <= 0)


m.init_fai_cap_ns = Constraint(T[0:1], NS, rule=init_fai_cap_ns_rule)


def init_fas_cap_os_rule(m, t, s):
    """
    Returns the initial ship capacity for old ships. [PJ]
    """
    return (m.fas_cap[t, s] - fa_init[t, s] >= 0)


m.init_fas_cap_os = Constraint(T[0:1], OS, rule=init_fas_cap_os_rule)


def init_fas_cap_ns_rule(m, t, s):
    """
    Returns the initial ship capacity for new ships. [PJ]
    """
    return (m.fas_cap[t, s] - fa_init[t, s] <= 0)


m.init_fas_cap_ns = Constraint(T[0:1], NS, rule=init_fas_cap_ns_rule)


#
# Objective function
#

def obj_rule(m):
    """
    Returns the minimized total system costs. [10^9 €]
    """
    return sum(sum(m.CF[t, s] +
                   m.CI[t, s] +
                   m.CS[t, s] for t in T) for s in S)


m.system_costs = Objective(sense=minimize, rule=obj_rule)


#
# Subject to
#

def cf_rule(m, t, s):
    """
    Returns the total fuel costs of all ships in each timestep. [10^9 €]
    """
    return (m.CF[t, s] >= m.fa[t, s] * cf[t, s])


m.costs_fuel = Constraint(T, S, rule=cf_rule)


def ci_rule(m, t, s):
    """
    Returns the total infrastructure costs of all additional infrastructure in
    each timestep. [10^9 €]
    """
    if t == T[0]:
        return (m.CI[t, s] >= ((m.fai_up[t, s] - fa_init[t, s]) *
                               li[s] * ci[t, s]))
    else:
        return (m.CI[t, s] >= m.fai_up[t, s] * li[s] * ci[t, s])


m.costs_infrastructure = Constraint(T, S, rule=ci_rule)


def cs_rule(m, t, s):
    """
    Returns the total costs for all additional and refit ships in each
    timestep. [10^9 €]
    """
    if t == T[0]:
        return (m.CS[t, s] >= ((m.fas_up[t, s] - fa_init[t, s]) *
                               ls[s] * cs[t, s]))
    else:
        return (m.CS[t, s] >= m.fas_up[t, s] * ls[s] * cs[t, s])


m.costs_ship = Constraint(T, S, rule=cs_rule)


#
# Fuel constraints
#

def icap_rule(m, t, s):
    """
    Returns the infrastructure capacity as the sum of all added infrastructure
    over their lifetimes for each ship type. [PJ]
    """
    if t - li[s] + 1 <= T[0]:
        x = T[0]
        return (m.fai_cap[t, s] == sum(m.fai_up[i, s] for i in range(x, t)))
    else:
        x = t - li[s] + 1
        return (m.fai_cap[t, s] == sum(m.fai_up[i, s] for i in range(x, t)))


m.icap = Constraint(T[1:], S, rule=icap_rule)


def iup_rule(m, t, s):
    """
    Returns the cost-effective added infrastructure for each ship type in each
    timestep. [PJ]
    """
    return (m.fai_up[t, s] >= m.fa[t, s] - m.fai_cap[t, s])


m.iup = Constraint(T[1:], S, rule=iup_rule)


def scap_rule(m, t, s):
    """
    Returns the ship capacity as the sum of all added ships over their
    lifetimes for each ship type. [PJ]
    """
    if t - ls[s] + 1 <= T[0]:
        x = T[0]
        return (m.fas_cap[t, s] == sum(m.fas_up[i, s] for i in range(x, t)))
    else:
        x = t - ls[s] + 1
        return (m.fas_cap[t, s] == sum(m.fas_up[i, s] for i in range(x, t)))


m.scap = Constraint(T[1:], S, rule=scap_rule)


def sup_rule(m, t, s):
    """
    Returns the cost-effective added ships for each ship type in each
    timestep. [PJ]
    """
    return (m.fas_up[t, s] >= m.fa[t, s] - m.fas_cap[t, s])


m.sup = Constraint(T[1:], S, rule=sup_rule)


#
# Refit constraint
#

def refit_rule(m, t, s, r):
    """
    Returns the fuel amount available for refit ships. [PJ]
    """
    if t < T[0] + ls[s]:
        return (m.fa[t, s] + m.fa[t, r] - m.fa[T[0], s] <= 0)
    else:
        return (m.fa[t, s] + m.fa[t, r] == 0)


m.refit = Constraint(T[1:], RO, rule=refit_rule)


#
# Biofuel constraints
#

def biofuel_rule(m, t):
    """
    Returns the amount of biofuel availabe in each stimestep. [PJ]
    """
    return (sum(m.fa[t, s] for s in SB) - ba[t] <= 0)


m.biofuel = Constraint(T, rule=biofuel_rule)


#
# Demand constraints
#

def td_rule(m, t):
    """
    Returns the amount of fuel needed to (over-)supply the total transport
    demand for all ships in each timestep. [PJ]
    """
    return (td_total[t] <= sum(m.fa[t, s] * ts[t, s] for s in S))


m.transport_demand = Constraint(T, rule=td_rule)


def td_short_rule(m, t):
    """
    Returns the maximum amount of fuel available for the short range ships in
    each timestep. [PJ]
    """
    return (td_short[t] >= sum(m.fa[t, s] * ts[t, s] for s in SR))


m.transport_demand_short = Constraint(T, rule=td_short_rule)


def td_noneca_rule(m, t):
    """
    Returns the maximum amount of fuel available for ships not in compliance
    with the ecas sox regulations for each timestep. [PJ]
    """
    return (td_noneca[t] >= sum(m.fa[t, s] * ts[t, s] for s in SNE))


m.td_noneca = Constraint(T, rule=td_noneca_rule)


#
# Emission constraints
#

def co2_emission_rule(m, t, s):
    """
    Returns the co2 emissions for all ships and timesteps. [10^9 g]
    """
    return (m.EC[t, s] == m.fa[t, s] * ec[t, s])


m.co2_emissions = Constraint(T, S, rule=co2_emission_rule)


def ch4_emission_rule(m, t, s):
    """
    Returns the ch4 emissions for all ships and timesteps. [10^9 g]
    """
    return (m.EM[t, s] == m.fa[t, s] * em[t, s])


m.ch4_emissions = Constraint(T, S, rule=ch4_emission_rule)


def co2e_budget_rule(m):
    """
    Returns the total co2e emissions of all ships and timesteps that cannot
    exceed the emission budget of the Danish maritime sector. [[10^9 g]]
    """
    return (eb - sum(sum(m.EC[t, s] + m.EM[t, s] for s in S) for t in T) >= 0)


m.co2e_budget = Constraint(rule=co2e_budget_rule)


def co2e_target_rule(m, t):
    """
    Returns the co2e emission limit from the selected year onwards that connot
    be exceeded. [10^9 g]
    """
    return (sum(m.EC[T[0], s] + m.EM[T[0], s] for s in S) * et >=
            sum(m.EC[t, s] + m.EM[t, s] for s in S))


m.co2e_target = Constraint(T[et_yr-T[0]:], rule=co2e_target_rule)


def global_sox_rule(m, t, s):
    """
    Sets the fuel amount to zero for ships not in compliance with the global
    sulphur limit from the year onwards when the regulation takes effect. [PJ]
    """
    return (m.fa[t, s] == 0)


m.global_sox = Constraint(T[sl_noneca_yr-T[0]:], SNG, rule=global_sox_rule)


def tier_rule(m, t, s):
    """
    Sets the fuel amount to zero for ships not in compliance with the nox
    regulations inside the baltic and north see ecas from the year onwards when
    the regulation takes effect. [PJ]
    """
    return (m.fa[t, s] == 0)


m.tier = Constraint(T[tl_eca_yr-T[0]:], NT, rule=tier_rule)


# def co2e_evolution_rule(m, t):
#    """
#    Forces the model to continuously reduce the co2e emissions for all ships
#    in each timestep from the peak year onwards.
#    """
#    return (sum(m.EC[t, s] + m.EM[t, s] for s in S) <=
#            sum(m.EC[t-1, s] + m.EM[t-1, s] for s in S))
#
#
# m.co2e_evolution = Constraint(T[ep_yr-T[0]:], rule=co2e_evolution_rule)


# =============================================================================
# Make output directory
# =============================================================================

results_path = 'model_output/' + R

if not os.path.isdir(results_path):
    os.makedirs(results_path)


# =============================================================================
# Solve model
# =============================================================================

#
# Construct solver object
#

opt = SolverFactory('glpk')


#
# Apply solver and look at solution
#

results = opt.solve(m, tee=True)
results.write(num=1)


#
# Save results
#

results_var = {i.name: i.get_values() for i in m.component_objects(Var)}
variables = pd.DataFrame(results_var)
variables.to_csv(results_path + '/results_%s.csv' % scn, encoding='utf8')
