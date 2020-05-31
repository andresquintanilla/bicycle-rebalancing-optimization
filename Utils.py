# %%
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 05 20:47:23 2020

@author: AndresQuintanilla
"""
# Importing necesary libraries/modules
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics.pairwise import haversine_distances
from sklearn.metrics import mean_squared_error, r2_score


def plot_timeseries(df, ts, features, label_x, label_y, locator, date_format='%Y-%b', font_size=22, fig_size=(30,15)):
  plt.figure(figsize=fig_size)
  for i, feature in enumerate(features, start=0):
    plt.plot(df[ts],df[feature])
  plt.grid(linestyle=':', linewidth='1', color='grey')
  plt.legend()
  plt.xlabel(label_x,fontsize=30)
  plt.xticks(fontsize=font_size, rotation=90)
  plt.ylabel(label_y,fontsize=30)
  plt.yticks(fontsize=font_size)
  plt.gca().xaxis.set_major_locator(locator)
  plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_format))
  #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
  plt.gca().set_facecolor('white')
  plt.show()

def plot_line(df, x, y_arr, label_x, label_y, font_size=22, fig_size=(30,15)):
  plt.figure(figsize=fig_size)
  for i, y in enumerate(y_arr, start=0):
    plt.plot(df[x],df[y],marker="o")
  plt.legend(fontsize=30)
  #plt.grid(linestyle=':', linewidth='1', color='grey')
  plt.xlabel(label_x,fontsize=30)
  plt.xticks(ticks=df[x],fontsize=font_size, rotation=90)
  plt.ylabel(label_y,fontsize=30)
  plt.yticks(fontsize=font_size)
  plt.gca().set_facecolor('white')
  plt.show()

def plot_pie_chart(labels, values, explode, colors, font_size=35, fig_size=(30,15)):
  plt.figure(figsize=fig_size)
  plt.pie(values, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': font_size}, colors=colors)
  plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
  plt.show()

def plot_distribution(plot_x, plot_y, label_x, label_y, font_size=22, fig_size=(30,15)):
  plt.figure(figsize=fig_size)
  plt.bar(plot_x, plot_y, tick_label=plot_x, color='royalblue')
  plt.xlabel(label_x,fontsize=30)
  plt.xticks(fontsize=font_size, rotation=90)
  plt.ylabel(label_y,fontsize=30)
  plt.yticks(fontsize=font_size)
  plt.show()


def eval_model(model, X_train, y_train, X_test, y_test):
  ypred = model.predict(X_test)
  r2score = round(r2_score(y_test, ypred), 2)
  rmse = round(mean_squared_error(y_test, ypred), 2)
  
  return r2score, rmse

def plot_predicted_vs_actual(df, actual, pred):
  fig = plt.figure(figsize=(15,8))
  x = df[actual]
  y = df[pred]
  poly_model = np.poly1d(np.polyfit(x, y, 1))
  plt.plot(x, y, '.')
  plt.plot(range(math.floor(x.min()),math.ceil(x.max())+1),poly_model(range(math.floor(x.min()),math.ceil(x.max())+1)),'-')
  plt.xlim(0, x.max().round(decimals=2))
  plt.ylim(0, x.max().round(decimals=2))
    
  textstr = '\n'.join((r'R2=%.2f'% round(r2_score(df[actual], df[pred]), 2)
                      ,r'MSE=%.2f' % round(mean_squared_error(df[actual], df[pred]), 2) 
                      ))
  
  props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
  plt.xlabel('Actual')
  plt.ylabel('Predict')
  plt.text(0.05, 0.95, textstr, fontsize=18,verticalalignment='top', bbox=props)
  plt.title('Predicted vs Actual')
  plt.show()

def haversine(row):
  from_station = [row['rad_lat_i'],row['rad_lon_i']]
  to_station = [row['rad_lat_j'],row['rad_lon_j']]
  
  distance = haversine_distances([from_station,to_station])
  distance = distance * 6371000/1000  # multiply by Earth radius to get kilometers
  
  return distance[0][1]