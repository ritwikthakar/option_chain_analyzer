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
import matplotlib.pyplot as plt



# In[6]:

st.header("Option Chain Analyzer")
symbol = st.sidebar.text_input('Enter Ticker', 'SPY')
tk = yf.Ticker(symbol)
r = st.sidebar.number_input('Enter Risk Free Rate', 1, value=(5))
r = r/100
expiry = st.sidebar.date_input("Expiry Date",datetime.date(2023, 1, 20))
expiry = expiry.strftime('%Y-%m-%d')
close = tk.info['regularMarketPrice']
st.write('Available Expiry Dates', tk.options)
# In[7]:


# Get options exp
options = pd.DataFrame()
opt = tk.option_chain(expiry)
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
st.subheader('Call Option')
#call_df

call_strike = call_df['strike']
cs = st.sidebar.selectbox('Select Call Strike:', call_strike)
st.write(call_df[call_df['strike'] == cs])
# In[11]:


put_df['BSM Value'] = bsm(r, close, put_df['strike'], put_df['dte'], put_df['impliedVolatility'], type = 'p')
put_df['Delta'] = delta(r, close, put_df['strike'], put_df['dte'], put_df['impliedVolatility'], type = 'p')
put_df['Gamma'] = gamma(r, close, put_df['strike'], put_df['dte'], put_df['impliedVolatility'], type = 'c')
put_df['Vega'] = vega(r, close, put_df['strike'], put_df['dte'], put_df['impliedVolatility'], type = 'c')
put_df['Theta'] = theta(r, close, put_df['strike'], put_df['dte'], put_df['impliedVolatility'], type = 'p')
put_df['Rho'] = rho(r, close, put_df['strike'], put_df['dte'], put_df['impliedVolatility'], type = 'p')
st.subheader('Put Option')
#put_df

put_strike = put_df['strike']
ps = st.sidebar.selectbox('Select Put Strike:', put_strike)
st.write(put_df[put_df['strike'] == ps])


def options_chain(tk, expiry):
    '''
    Get's the option chain for a given symbol and expiry date and add it to panda df
    Credit: https://medium.com/@txlian13/webscrapping-options-data-with-python-and-yfinance-e4deb0124613
    '''
    # Get options exp
    options = pd.DataFrame()
    opt = tk.option_chain(expiry)
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
    options = options.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate', 'lastPrice', 'contractSymbol', 'bid', 'ask', 'impliedVolatility', 'inTheMoney', 'dte'])
    
    return options

chain = options_chain(tk, expiry)

def total_loss_on_strike(chain, expiry_price):
    '''
    Get's the total loss at the given strike price
    '''    
    # call options with strike price below the expiry price -> loss for option writers
    callChain = chain.loc[chain['CALL'] == True]
    callChain = callChain.dropna()       
    in_money_calls = callChain[callChain['strike'] < expiry_price][["openInterest", "strike"]]
    in_money_calls["CLoss"] = (expiry_price - in_money_calls['strike'])*in_money_calls["openInterest"]

    # get puts n drop null values
    putChain = chain.loc[chain['CALL'] == False]
    putChain = putChain.dropna()    
    
    # put options with strike price above the expiry price -> loss for option writers
    in_money_puts = putChain[putChain['strike'] > expiry_price][["openInterest", "strike"]]
    in_money_puts["PLoss"] = (in_money_puts['strike'] - expiry_price)*in_money_puts["openInterest"]
    total_loss = in_money_calls["CLoss"].sum() + in_money_puts["PLoss"].sum()

    return total_loss

strikes = chain.get(['strike']).values.tolist()
losses = [total_loss_on_strike(chain, strike[0]) for strike in strikes] 

# max pain min loss to option writers/sellers at strike price
flat_strikes = [item for sublist in strikes for item in sublist]
point = losses.index(min(losses))
max_pain = flat_strikes[point]
buffer = 3
bufferHigh = max_pain + (max_pain * (buffer/100))
bufferLow = max_pain - (max_pain * (buffer/100))


# calc put to call ratio
callChain = chain.loc[chain['CALL'] == True]
putChain = chain.loc[chain['CALL'] == False]
pcr = putChain["volume"].sum() / callChain["volume"].sum()


# get the cummulated losses
total = {}
for i in range(len(flat_strikes)):
    if flat_strikes[i] not in total: total[flat_strikes[i]] = losses[i]
    else: total[flat_strikes[i]] += losses[i]


# plot
keys = set(list(total.keys()))
fig = plt.figure(figsize = (15, 6))
plt.bar(list(keys), list(total.values()), width=1)
plt.xlabel('Strike Price')
plt.ylabel('Maximum Loss')
plt.title(f'{symbol.upper()} Max Pain')

fig1 = plt.figure(figsize = (15, 6))
plt.bar(call_df['strike'],call_df['openInterest'], label="Calls")
plt.bar(put_df['strike'],put_df['openInterest'], label="Puts")
plt.xlabel('Strike Price')
plt.ylabel('Open Interest')
plt.legend(loc = 'upper left')
plt.title(f'{symbol.upper()} Open Interest')

fig2 = plt.figure(figsize = (15, 6))
plt.bar(call_df['strike'],call_df['openInterest'], label="Calls")
plt.bar(put_df['strike'],put_df['openInterest'], label="Puts")
plt.xlabel('Strike Price')
plt.ylabel('Volume')
plt.legend(loc = 'upper left')
plt.title(f'{symbol.upper()} Option Volume')

st.subheader('Maximum Pain')
st.pyplot(fig)
st.write(f"Maximum Pain: {bufferLow} < {max_pain} < {bufferHigh}")
st.write("Put to call ratio:", round(pcr,2))

st.subheader('Open Interest')
st.pyplot(fig1)

st.subheader('Option Volume')
st.pyplot(fig2)
