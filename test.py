# -*- coding: utf-8 -*-
"""
Created on Thu May  7 22:13:47 2020

@author: Rohini
"""

import altair as alt
import pandas as pd
import numpy as np

np.random.seed(0)

n_objects = 20
n_times = 50

DPS_resultfile = np.loadtxt('DPS.resultfile')
DPS_reference = np.loadtxt('DPS.reference')


x=-1*DPS_reference[:,0]
y=DPS_reference[:,1]

n_objects = len(DPS)
n_times = len(lake_state)

lake_state = np.arange(0,2.5,0.2)    
Y_new=np.zeros([len(lake_state),len(DPS_resultfile)]) 
for z in range(len(DPS_resultfile)):
    for i in range(len(lake_state)):
        Y_new[i,z] = DPSpolicy(lake_state[i], DPS_resultfile[z,0:6]) # DPS best reliability
        
        
# Create one (x, y) pair of metadata per object
locations = pd.DataFrame({
    'id': range(n_objects),
    'x': x,
    'y': y
})





# Create a 50-element time-series for each object
timeseries = pd.DataFrame(Y_new,columns=locations['id'],index=pd.Index(['0', '0.2', '0.4', '0.6','0.8','1','1.2','1.4','1.6','1.8','2','2.2','2.4'], name='time'))


# Create a 50-element time-series for each object
timeseries = pd.DataFrame(np.random.randn(n_times, n_objects).cumsum(0),
                          columns=locations['id'],
                          index=pd.RangeIndex(0, n_times, name='Lake P Concentration'))

# Melt the wide-form timeseries into a long-form view
timeseries = timeseries.reset_index().melt('Lake P Concentration')

# Merge the (x, y) metadata into the long-form view
timeseries['id'] = timeseries['id'].astype(int)  # make merge not complain
data = pd.merge(timeseries, locations, on='id')

# Data is prepared, now make a chart

selector = alt.selection_single(empty='all', fields=['id'])

base = alt.Chart(data).properties(
    width=250,
    height=250
).add_selection(selector)

points = base.mark_point(filled=True, size=200).encode(
    x='mean(x)',
    y='mean(y)',
    color=alt.condition(selector, 'id:O', alt.value('lightgray'), legend=None),
)

timeseries = base.mark_line().encode(
    x='time',
    y=alt.Y('value', scale=alt.Scale(domain=(-15, 15))),
    color=alt.Color('id:O', legend=None)
).transform_filter(
    selector
)

points | timeseries






import altair as alt
from vega_datasets import data

n_objects_success=len(DPS_success)
n_objects_fail=len(DPS_fail)


data_success = pd.DataFrame({
    'id': range(n_objects_success),
    'x': DPS_success[:,0],
    'y': DPS_success[:,1],
    'state': 'Success'
})

data_failure = pd.DataFrame({
    'id': range(n_objects_fail),
    'x': DPS_fail[:,0],
    'y': DPS_fail[:,1],
    'state': 'Failure'
})    
    
frames=[data_success,data_failure]    
data = pd.concat(frames)

brush = alt.selection(type='interval')

points = alt.Chart(source).mark_point().encode(
    x='Horsepower:Q',
    y='Miles_per_Gallon:Q',
    color=alt.condition(brush, 'Origin:N', alt.value('lightgray'))
).add_selection(
    brush
)

bars = alt.Chart(source).mark_bar().encode(
    y='Origin:N',
    color='Origin:N',
    x='count(Origin):Q'
).transform_filter(
    brush
)

points & bars













