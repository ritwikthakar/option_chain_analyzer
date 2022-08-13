#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.stats import norm, bernoulli
from scipy.sparse import csc_matrix
import datetime as dt
import streamlit as st

# In[2]:


x = dt.date.today()
y = dt.date(2023, 1, 20)
z = y-x
# z


# In[3]:


# Leg A
r = 0.05 #Risk Free Rate
s = 36.15 #Uderlying
k = 40 #Strik
T = 168/365 #Time
sigma = 0.7483 #Vol


# In[4]:


def bsm(r,s,k,T,sigma,type = 'p'):
  d1 = (np.log(s/k) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
  d2 = d1 - sigma*np.sqrt(T)
  try:
    if type == 'c':
      price = s*norm.cdf(d1, 0, 1) - k*np.exp(-r*T)*norm.cdf(d2, 0, 1)
    elif type == 'p':
      price = k*np.exp(-r*T)*norm.cdf(-d2, 0, 1) - s*norm.cdf(-d1, 0, 1)
    return price
  except:
    print('Please Confirm all Parameteres')

def delta(r,s,k,T,sigma,type = 'p'):
  d1 = (np.log(s/k) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
  try:
    if type == 'c':
      delta = norm.cdf(d1, 0, 1)
    elif type == 'p':
      delta = -norm.cdf(-d1, 0, 1)
    return delta
  except:
    print('Please Confirm all Parameteres')

def gamma(r,s,k,T,sigma,type = 'c'):
  d1 = (np.log(s/k) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
  try:
      gamma = norm.pdf(d1, 0, 1)/(s*sigma*np.sqrt(T))
      return gamma
  except:
    print('Please Confirm all Parameteres')

def vega(r,s,k,T,sigma,type = 'c'):
  d1 = (np.log(s/k) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
  d2 = d1 - sigma*np.sqrt(T)
  try:
    vega = s*norm.pdf(d1, 0, 1)*np.sqrt(T)
    return vega
  except:
    print('Please Confirm all Parameteres')

def theta(r,s,k,T,sigma,type = 'c'):
  d1 = (np.log(s/k) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
  d2 = d1 - sigma*np.sqrt(T)
  try:
    if type == 'c':
      theta = -s*norm.pdf(d1, 0, 1)*sigma/(2*np.sqrt(T)) - r*k*np.exp(-r*T)*norm.cdf(d2, 0, 1)
    elif type == 'p':
      theta = -s*norm.pdf(d1, 0, 1)*sigma/(2*np.sqrt(T)) + r*k*np.exp(-r*T)*norm.cdf(-d2, 0, 1)
    return theta
  except:
    print('Please Confirm all Parameteres')

def rho(r,s,k,T,sigma,type = 'c'):
  d1 = (np.log(s/k) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
  d2 = d1 - sigma*np.sqrt(T)
  try:
    if type == 'c':
      rho = k*T*np.exp(-r*T)*norm.cdf(d2, 0, 1)
    elif type == 'p':
      rho = -k*T*np.exp(-r*T)*norm.cdf(-d2, 0, 1)
    return rho
  except:
    print('Please Confirm all Parameteres')


# In[5]:


import pandas as pd
import yfinance as yf
import datetime



# In[6]:

st.header("Option Chain Analyzer")
symbol = st.sidebar.text_input('Enter Ticker', 'SPY')
tk = yf.Ticker(symbol)
expiry = st.sidebar.text_input('Enter Expiry Date', '2023-01-20')
close = tk.info['regularMarketPrice']
st.write('Available Expiry Dates', tk.options)
# In[7]:


# Get options exp
options = pd.DataFrame()
opt = tk.option_chain(expiry.strip())
opt = pd.DataFrame().append(opt.calls).append(opt.puts)
opt['expirationDate'] = expiry
options = options.append(opt, ignore_index=True)

# Add 1 day to get the correct expiration date
options['expirationDate'] = pd.to_datetime(options['expirationDate']) + datetime.timedelta(days = 1)
options['dte'] = (options['expirationDate'] - datetime.datetime.today()).dt.days / 365

# Boolean column if the option is a CALL
options['CALL'] = options['contractSymbol'].str[4:].apply(
   lambda x: "C" in x)

options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)

# Drop unnecessary and meaningless columns
options = options.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate'])

options['CALL'] = options['CALL'].replace(True, 'c')
options['CALL'] = options['CALL'].replace(False, 'p')


# In[8]:


call_df = options.loc[lambda options: options['CALL'] == 'c']
put_df = options.loc[lambda options: options['CALL'] == 'p']


# In[10]:


call_df['BSM Value'] = bsm(r, close, call_df['strike'], call_df['dte'], call_df['impliedVolatility'], type = 'c')
call_df['Delta'] = delta(r, close, call_df['strike'], call_df['dte'], call_df['impliedVolatility'], type = 'c')
call_df['Gamma'] = gamma(r, close, call_df['strike'], call_df['dte'], call_df['impliedVolatility'], type = 'c')
call_df['Vega'] = vega(r, close, call_df['strike'], call_df['dte'], call_df['impliedVolatility'], type = 'c')
call_df['Theta'] = theta(r, close, call_df['strike'], call_df['dte'], call_df['impliedVolatility'], type = 'c')
call_df['Rho'] = rho(r, close, call_df['strike'], call_df['dte'], call_df['impliedVolatility'], type = 'c')
st.subheader('Call Options')
call_df


# In[11]:


put_df['BSM Value'] = bsm(r, close, put_df['strike'], put_df['dte'], put_df['impliedVolatility'], type = 'p')
put_df['Delta'] = delta(r, close, put_df['strike'], put_df['dte'], put_df['impliedVolatility'], type = 'p')
put_df['Gamma'] = gamma(r, close, put_df['strike'], put_df['dte'], put_df['impliedVolatility'], type = 'c')
put_df['Vega'] = vega(r, close, put_df['strike'], put_df['dte'], put_df['impliedVolatility'], type = 'c')
put_df['Theta'] = theta(r, close, put_df['strike'], put_df['dte'], put_df['impliedVolatility'], type = 'p')
put_df['Rho'] = rho(r, close, put_df['strike'], put_df['dte'], put_df['impliedVolatility'], type = 'p')
st.subheader('Put Options')
put_df






