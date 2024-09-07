import numpy as np
from scipy.stats import norm, bernoulli
from scipy.sparse import csc_matrix
import datetime as dt
import streamlit as st
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, rho, vega

# In[2]:


x = dt.date.today()
y = dt.date(2024, 1, 19)
z = y-x
# z


# In[3]:


# Leg A
r = 0.05 #Risk Free Rate
s = 36.15 #Uderlying
k = 40 #Strik
T = 168/365 #Time
sigma = 0.0998 #Vol


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
    vega = s*norm.pdf(d1)*np.sqrt(T)
    return vega/100
  except:
    print('Please Confirm all Parameteres')

def theta(r,s,k,T,sigma,type = 'c'):
  d1 = (np.log(s/k) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
  d2 = d1 - sigma*np.sqrt(T)
  try:
    if type == 'c':
      theta = -s*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*k*np.exp(-r*T)*norm.cdf(d2)
    elif type == 'p':
      theta = -s*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*k*np.exp(-r*T)*norm.cdf(-d2)
    return theta/365
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
    return rho/100
  except:
    print('Please Confirm all Parameteres')


# In[5]:


import pandas as pd
import yfinance as yf
from yahoo_fin import options as op
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import plotly.subplots as sp


# In[6]:

st.header("Option Chain Analyzer")
symbol = st.sidebar.text_input('Enter Ticker', 'SPY')
tk = yf.Ticker(symbol)
r1 = st.sidebar.number_input('Enter Risk Free Rate', 1, value=(5))
r2=r1/100
sigma1 = st.sidebar.number_input('Enter Implied Volatility',1, value=(10))
sigma2 = sigma1/100
expiry = st.sidebar.date_input("Start Date",datetime.date(2025, 1, 17))
expiry = expiry.strftime('%Y-%m-%d')
close_data = tk.history(period='1d')
close = close_data.Close.values
st.write('Available Expiry Dates', tk.options)
# In[7]:


# Get options exp
options = pd.DataFrame()
opt = tk.option_chain(expiry)

call_opts = opt.calls
put_opts = opt.puts
call_opts['CALL'] = 'c'
put_opts['CALL'] = 'p'
call_opts['expirationDate'] = expiry
put_opts['expirationDate'] = expiry
# Add 1 day to get the correct expiration date
call_opts['expirationDate'] = pd.to_datetime(call_opts['expirationDate']) + datetime.timedelta(days = 1)
call_opts['dte'] = (call_opts['expirationDate'] - datetime.datetime.today()).dt.days / 365
# Add 1 day to get the correct expiration date
put_opts['expirationDate'] = pd.to_datetime(put_opts['expirationDate']) + datetime.timedelta(days = 1)
put_opts['dte'] = (put_opts['expirationDate'] - datetime.datetime.today()).dt.days / 365
# Drop unnecessary and meaningless columns
call_opts = call_opts.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate'])
put_opts = put_opts.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate'])

# # Add 1 day to get the correct expiration date
# options['expirationDate'] = pd.to_datetime(options['expirationDate']) + datetime.timedelta(days = 1)
# options['dte'] = (options['expirationDate'] - datetime.datetime.today()).dt.days / 365

# # Boolean column if the option is a CALL
# options['CALL'] = options['contractSymbol'].str[4:].apply(lambda x: "C" in x)

# options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)

# # Drop unnecessary and meaningless columns
# options = options.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate'])

# options['CALL'] = options['CALL'].replace(True, 'c')
# options['CALL'] = options['CALL'].replace(False, 'p')


# In[8]:


call_df = call_opts
put_df = put_opts


# In[10]:


call_df['BSM Value'] = bsm(r2, close, call_df['strike'], call_df['dte'], call_df['impliedVolatility'], type = 'c')
call_df['Delta'] = delta(r2, close, call_df['strike'], call_df['dte'], call_df['impliedVolatility'], type = 'c')
call_df['Gamma'] = gamma(r2, close, call_df['strike'], call_df['dte'], call_df['impliedVolatility'], type = 'c')
call_df['Vega'] = vega(r2, close, call_df['strike'], call_df['dte'], call_df['impliedVolatility'], type = 'c')
call_df['Theta'] = theta(r2, close, call_df['strike'], call_df['dte'], call_df['impliedVolatility'], type = 'c')
call_df['Rho'] = rho(r2, close, call_df['strike'], call_df['dte'], call_df['impliedVolatility'], type = 'c')
call_df['Theta/Vega'] = call_df['Theta']/call_df['Vega']
call_df['GEX'] = call_df['Gamma'] * call_df['openInterest'] * close * close * 0.01 * 100
# st.subheader('Call Option')
#call_df

call_strike = call_df['strike']
cs = st.sidebar.selectbox('Select Call Strike:', call_strike)
# st.write(call_df[call_df['strike'] == cs])
# In[11]:


put_df['BSM Value'] = bsm(r2, close, put_df['strike'], put_df['dte'], put_df['impliedVolatility'], type = 'p')
put_df['Delta'] = delta(r2, close, put_df['strike'], put_df['dte'], put_df['impliedVolatility'], type = 'p')
put_df['Gamma'] = gamma(r2, close, put_df['strike'], put_df['dte'], put_df['impliedVolatility'], type = 'p')
put_df['Vega'] = vega(r2, close, put_df['strike'], put_df['dte'], put_df['impliedVolatility'], type = 'p')
put_df['Theta'] = theta(r2, close, put_df['strike'], put_df['dte'], put_df['impliedVolatility'], type = 'p')
put_df['Rho'] = rho(r2, close, put_df['strike'], put_df['dte'], put_df['impliedVolatility'], type = 'p')
put_df['Theta/Vega'] = put_df['Theta']/put_df['Vega']
put_df['GEX'] = put_df['Gamma'] * put_df['openInterest'] * close * close * 0.01 * 100 * -1
# st.subheader('Put Option')
#put_df

put_strike = put_df['strike']
ps = st.sidebar.selectbox('Select Put Strike:', put_strike)
# st.write(put_df[put_df['strike'] == ps])

options['total_gex'] = call_df['GEX'] + put_df['GEX']

def options_chain(tk, expiry):
    '''
    Get's the option chain for a given symbol and expiry date and add it to panda df
    Credit: https://medium.com/@txlian13/webscrapping-options-data-with-python-and-yfinance-e4deb0124613
    '''
    # Get options exp
    options = pd.DataFrame()
    opt = tk.option_chain(expiry.strip())
    opt = pd.concat([opt.calls, opt.puts], ignore_index=True)
    opt['expirationDate'] = expiry
    options = pd.concat([options, opt], ignore_index=True)

    # Add 1 day to get the correct expiration date
    options['expirationDate'] = pd.to_datetime(options['expirationDate']) + datetime.timedelta(days=1)
    options['dte'] = (options['expirationDate'] - datetime.datetime.today()).dt.days / 365

    # Boolean column if the option is a CALL
    options['CALL'] = options['contractSymbol'].str[4:].apply(
        lambda x: "C" in x)

    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)

    # Drop unnecessary and meaningless columns
    options = options.drop(
        columns=['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate', 'lastPrice', 'contractSymbol',
                 'bid', 'ask', 'impliedVolatility', 'inTheMoney', 'dte'])

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
    in_money_calls["CLoss"] = (expiry_price - in_money_calls['strike']) * in_money_calls["openInterest"]

    # get puts n drop null values
    putChain = chain.loc[chain['CALL'] == False]
    putChain = putChain.dropna()

    # put options with strike price above the expiry price -> loss for option writers
    in_money_puts = putChain[putChain['strike'] > expiry_price][["openInterest", "strike"]]
    in_money_puts["PLoss"] = (in_money_puts['strike'] - expiry_price) * in_money_puts["openInterest"]
    total_loss = in_money_calls["CLoss"].sum() + in_money_puts["PLoss"].sum()

    return total_loss

strikes = chain.get(['strike']).values.tolist()
losses = [total_loss_on_strike(chain, strike[0]) for strike in strikes]

# max pain min loss to option writers/sellers at strike price
flat_strikes = [item for sublist in strikes for item in sublist]
point = losses.index(min(losses))
max_pain = flat_strikes[point]
buffer = 3
bufferHigh = max_pain + (max_pain * (buffer / 100))
bufferLow = max_pain - (max_pain * (buffer / 100))
print(f"Maximum Pain: {bufferLow} < {max_pain} < {bufferHigh}")

# calc put to call ratio
callChain = chain.loc[chain['CALL'] == True]
putChain = chain.loc[chain['CALL'] == False]
pcr = putChain["volume"].sum() / callChain["volume"].sum()
print("Put to call ratio:", round(pcr, 2))

# get the cummulated losses
total = {}
for i in range(len(flat_strikes)):
    if flat_strikes[i] not in total:
        total[flat_strikes[i]] = losses[i]
    else:
        total[flat_strikes[i]] += losses[i]

# plot
keys = set(list(total.keys()))
#fig = plt.figure(figsize = (15, 6))
#plt.bar(list(keys), list(total.values()), width=1)
#plt.xlabel('Strike Price')
#plt.title(f'{symbol.upper()} Max Pain')

fig = go.Figure()

# Add call bars
fig.add_trace(go.Bar(
    x=list(keys),
    y=list(total.values()),
    name='Max Pain',
    marker_color='blue'
))

# Update layout
fig.update_layout(
    title='Max Pain',
    xaxis_title='Strike Price',
    yaxis_title='Max Pain'
)


#fig1 = plt.figure(figsize = (15, 6))
#plt.bar(call_df['strike'],call_df['openInterest'], label="Calls")
#plt.bar(put_df['strike'],put_df['openInterest'], label="Puts")
#plt.xlabel('Strike Price')
#plt.ylabel('Open Interest')
#plt.legend(loc = 'upper left')
#plt.title(f'{symbol.upper()} Open Interest')

fig1 = go.Figure()

# Add call bars
fig1.add_trace(go.Bar(
    x=call_df['strike'],
    y=call_df['openInterest'],
    name='Call Open Interest',
    marker_color='green'
))

# Add put bars
fig1.add_trace(go.Bar(
    x=put_df['strike'],
    y=put_df['openInterest'],
    name='Put Open Interest',
    marker_color='red'
))

# Update layout
fig1.update_layout(
    title='Open Interest',
    xaxis_title='Strike Price',
    yaxis_title='Open Interest'
)

#fig2 = plt.figure(figsize = (15, 6))
#plt.bar(call_df['strike'],call_df['impliedVolatility'], label="Calls")
#plt.bar(put_df['strike'],put_df['impliedVolatility'], label="Puts")
#plt.xlabel('Strike Price')
#plt.ylabel('Implied Volatility')
#plt.legend(loc = 'upper left')
#plt.title(f'{symbol.upper()} IV')

fig2 = go.Figure()

# Add call bars
fig2.add_trace(go.Line(
    x=call_df['strike'],
    y=call_df['impliedVolatility'],
    name='Call IV',
    marker_color='green'
))

# Add put bars
fig2.add_trace(go.Line(
    x=put_df['strike'],
    y=put_df['impliedVolatility'],
    name='Put IV',
    marker_color='red'
))

# Update layout
fig2.update_layout(
    title='Skew',
    xaxis_title='Strike Price',
    yaxis_title='Implied Volatility'
)

#fig3 = plt.figure(figsize = (15, 6))
#plt.bar(call_df['strike'],call_df['GEX'], label="Call Gamma")
#plt.bar(put_df['strike'],put_df['GEX'], label="Put Gamma")
#plt.xlabel('Strike Price')
#plt.ylabel('Gamma Exposure')
#plt.legend(loc = 'upper left')
#plt.title(f'{symbol.upper()} Gamma Exposure')

fig3 = go.Figure()

# Add call bars
fig3.add_trace(go.Bar(
    x=call_df['strike'],
    y=call_df['GEX'],
    name='Call GEX',
    marker_color='green'
))

# Add put bars
fig3.add_trace(go.Bar(
    x=put_df['strike'],
    y=put_df['GEX'],
    name='Put',
    marker_color='red'
))

# Update layout
fig3.update_layout(
    title='GEX by Calls & Puts',
    xaxis_title='Strike Price',
    yaxis_title='GEX'
)

# st.subheader('Maximum Pain')
# st.pyplot(fig)
# st.write(f"Maximum Pain: {bufferLow} < {max_pain} < {bufferHigh}")
# st.write("Put to call ratio:", round(pcr,2))

# st.subheader('Open Interest')
# st.pyplot(fig1)

# st.subheader('Option Volume')
# st.pyplot(fig2)

fig_g = go.Figure()

# Add call bars
fig_g.add_trace(go.Bar(
    x=call_df['strike'],
    y=options['total_gex'],
    name='Call GEX',
    marker_color='blue'
))

# Update layout
fig_g.update_layout(
    title='Total Gamma',
    xaxis_title='Strike Price',
    yaxis_title='Total Gex'
)

# #to match stock's price with the strike price
# options['abs'] = abs(close - options['strike'])
# closest = options.sort_values('abs')
atmCall = opt.calls.loc[opt.calls['inTheMoney'] == False].iloc[0]
atmPuts = opt.puts.loc[opt.puts['inTheMoney'] == False].iloc[0]

move = (atmCall['lastPrice'] +
        atmPuts['lastPrice']) * 2.23

upper_move = close + move
lower_move = close - move

start = dt.datetime.today()-dt.timedelta(180)
end = dt.datetime.today()
data = yf.download(symbol, start, end)
#adding the future dates into the data dataframe

fig5 = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Adj Close'], increasing_line_color='green', decreasing_line_color='red')])
# fig5.add_trace(go.Scatter(x=[data.index[0], data.index[-1]], y=[upper_move, upper_move] ,mode = 'markers',marker=dict(color='red', symbol='star', size = 14), name = 'Expected Move Up'))
# fig5.add_trace(go.Scatter(x=[data.index[0], data.index[-1]], y=[lower_move,lower_move] ,mode = 'markers',marker=dict(color='green', symbol='star', size = 14), name = 'Expected Move Down'))
#fig5.add_trace(go.Scatter(x=[data.index[0], data.index[-1]], y=[upper_move,upper_move] ,line=dict(color = 'red', width=2, dash='dash'), name = 'Expected Move Up'))
#fig5.add_trace(go.Scatter(x=data.index, y=lower_move ,mode = 'lines',line=dict(color='green', width = 10), name = 'Expected Move Down'))


layout = go.Layout(
#         xaxis_rangeslider_visible=False, 
#         xaxis_tradingcalendar=True,
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=20,
    height=600, width=1000,)


fig5.update_xaxes(
            rangeslider_visible=False,
            rangebreaks=[
                # NOTE: Below values are bound (not single values), ie. hide x to y
                dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                    # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                ]
                    )
fig5.update_layout(layout)

#fig5 = plt.figure(figsize = (15, 6))
#plt.plot(data['Adj Close'])
#plt.axhline(upper_move, linestyle = '--', alpha = 0.5, color = 'red')
#plt.axhline(lower_move, linestyle = '--', alpha = 0.5, color = 'green')
#plt.title(f'{symbol.upper()} Expected Move for Selected Expiry Date based on Options Implied Volatlilty')
#plt.xlabel('Price')
#plt.ylabel('Date')



def getVerticalSpreadPrice(ticker, spreadType, expNo, longStrike, shortStrike):
    
# #inputs
# ticker = 'PLTR'
# spreadType = 'put' #call or put
# expNo = 5
# longStrike = 20
# shortStrike = 19

    try:
    
        #get expiration dates
        expirationDates = op.get_expiration_dates(ticker)
        
        
        #get data based on spread type, call or put spread
        if spreadType == 'call':
            chainData = op.get_calls(ticker, date = expirationDates[expNo])
        elif spreadType == 'put':
            chainData = op.get_puts(ticker, date = expirationDates[expNo])
        else: 
            st.write('Please enter call or put for spreadType.')
            return
        
        #trim data 
        chainData = chainData[['Strike', 'Bid', 'Ask', 'Last Price']][
            (chainData['Strike'] == longStrike) |
            (chainData['Strike'] == shortStrike)]
        
        #reset index
        chainData = chainData.reset_index(drop = True)
        
        #change to numeric data type
        chainData['Strike'] = pd.to_numeric(
                chainData['Strike'], errors = 'coerce')
        chainData['Bid'] = pd.to_numeric(
                chainData['Bid'], errors = 'coerce')
        chainData['Ask'] = pd.to_numeric(
                chainData['Ask'], errors = 'coerce')
        chainData['Last Price'] = pd.to_numeric(
                chainData['Last Price'], errors = 'coerce')
        
        #create mid price for reference
        chainData['Mid'] = (chainData['Bid'] + chainData['Ask']) / 2
        
        st.write(chainData)
        st.write('')
        
        st.write('- - - Vertical Spread - - -\n')
        
        #get spread price
        spreadPrice = chainData['Ask'][chainData['Strike'] == longStrike].iloc[0
                  ] - chainData['Bid'][chainData['Strike'] == shortStrike].iloc[0]
        
        #print pricing
        if spreadType == 'call' and longStrike < shortStrike:
            st.write('Long Call Spread Price is $' + str(round(spreadPrice, 4)))
        elif spreadType == 'call' and longStrike > shortStrike:
            st.write('Short Call Spread Price is -$' + str(abs(round(spreadPrice, 4))))
        elif spreadType == 'put' and longStrike > shortStrike:
            st.write('Long Put Spread Price is $' + str(round(spreadPrice, 4)))
        elif spreadType == 'put' and longStrike < shortStrike:
            st.write('Short Put Spread Price is -$' + str(abs(round(spreadPrice, 4))))
        elif longStrike == shortStrike:
            st.write('Strike prices are the same, no vertical spreads here.')
            return
            
        #output
        return  round(spreadPrice, 4)
    
    except IndexError:
        st.write('Strike data not available, try again.')

def getDiagonalSpreadPrice(ticker, spreadType, longExpNo, shortExpNo,
                            longStrike, shortStrike):
    
# #inputs
# ticker = 'PLTR'
# spreadType = 'put' #call or put
# longExpNo = 7
# shortExpNo = 5
# longStrike = 20
# shortStrike = 19

    try:
    
        #get expiration dates
        expirationDates = op.get_expiration_dates(ticker)
        
        st.write('- - - Diagonal Spread - - -\n')
        
        #get data based on spread type, call or put spread
        if spreadType == 'call':
            longChainData = op.get_calls(ticker, date = expirationDates[longExpNo])
            shortChainData = op.get_calls(ticker, date = expirationDates[shortExpNo])
        elif spreadType == 'put':
            longChainData = op.get_puts(ticker, date = expirationDates[longExpNo])
            shortChainData = op.get_puts(ticker, date = expirationDates[shortExpNo])
        else: 
            st.write('Please enter call or put for spreadType.')
            return
        
        #trim data 
        longChainData = longChainData[['Strike', 'Bid', 'Ask', 'Last Price']][
            longChainData['Strike'] == longStrike]
        shortChainData = shortChainData[['Strike', 'Bid', 'Ask', 'Last Price']][
            shortChainData['Strike'] == shortStrike]
        
        #reset index
        longChainData = longChainData.reset_index(drop = True)
        shortChainData = shortChainData.reset_index(drop = True)
        
        #change to numeric data type
        longChainData['Strike'] = pd.to_numeric(
                longChainData['Strike'], errors = 'coerce')
        longChainData['Bid'] = pd.to_numeric(
                longChainData['Bid'], errors = 'coerce')
        longChainData['Ask'] = pd.to_numeric(
                longChainData['Ask'], errors = 'coerce')
        longChainData['Last Price'] = pd.to_numeric(
                longChainData['Last Price'], errors = 'coerce')
        
        shortChainData['Strike'] = pd.to_numeric(
                shortChainData['Strike'], errors = 'coerce')
        shortChainData['Bid'] = pd.to_numeric(
                shortChainData['Bid'], errors = 'coerce')
        shortChainData['Ask'] = pd.to_numeric(
                shortChainData['Ask'], errors = 'coerce')
        shortChainData['Last Price'] = pd.to_numeric(
                shortChainData['Last Price'], errors = 'coerce')
        
        #create mid price for reference
        longChainData['Mid'] = (longChainData['Bid'] + longChainData['Ask']) / 2
        shortChainData['Mid'] = (shortChainData['Bid'] + shortChainData['Ask']) / 2
        
        #add expiration to dataframe
        longChainData['Expiration'] = expirationDates[longExpNo]
        shortChainData['Expiration'] = expirationDates[shortExpNo]
        
        st.write('Long Chain Data')
#        st.write('')
        st.write(longChainData, '\n - - - - - - - - -')
        st.write('Short Chain Data')
#        st.write('')
        st.write(shortChainData, ' \n')
        
        #get spread price
        spreadPrice = longChainData['Ask'].iloc[0] - shortChainData['Bid'].iloc[0]
        
        #print pricing
        if spreadType == 'call' and spreadPrice > 0:
            st.write('Long Call Diagonal Spread Price is $' + str(round(spreadPrice, 4)))
        elif spreadType == 'call' and spreadPrice < 0:
            st.write('Short Call Diagonal Spread Price is -$' + str(abs(round(spreadPrice, 4))))
        elif spreadType == 'put' and spreadPrice > 0:
            st.write('Long Put Diagonal Spread Price is $' + str(round(spreadPrice, 4)))
        elif spreadType == 'put' and spreadPrice < 0:
            st.write('Short Put Diagonal Spread Price is -$' + str(abs(round(spreadPrice, 4))))
        elif longExpNo == shortExpNo and shortStrike == longStrike:
            st.write('Strike prices are the same, no diagonal spreads here.')
            return

        #output
        return round(spreadPrice, 4)
    
    except IndexError:
        st.write('Strike data not available, try again.')
        
        
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['Expected Move IV','Max Pain' , "Open Interest", "Skew", 'Gamma Exposure', 'Option Chain', 'Individual Strike'])

with tab1:
    st.header("Expected Move")
    st.write("Expected price move between", np.round(float(upper_move),2), "and", np.round(float(lower_move),2), "in the range of", np.round(move), "for", expiry, "option expiry")
    #st.pyplot(fig5)
    st.plotly_chart(fig5)

with tab2:
    st.header("Max Pain")
#    st.pyplot(fig)
    st.plotly_chart(fig)
    st.write(f"Maximum Pain: {bufferLow} < {max_pain} < {bufferHigh}")
    st.write("Put to call ratio:", round(pcr,2))

with tab3:
    st.header("Open Interest")
    st.plotly_chart(fig1)

with tab4:
    st.header("Skew")
    st.plotly_chart(fig2)

with tab5:
    st.header("Gamma Exposure")
    st.plotly_chart(fig3)
    st.write("Total Call GEX:", round(call_df['GEX'].sum(),2))
    st.write("Total Put GEX:", round(put_df['GEX'].sum(),2))
    st.write("Gamma Exposure:", round((call_df['GEX'].sum()+put_df['GEX'].sum()),2))
    st.plotly_chart(fig_g)

with tab6:
    st.header("Option Chain")
    st.subheader('Call Options Chain')
    st.dataframe(call_df)
    st.subheader('Put Options Chain')
    st.dataframe(put_df)

with tab7:
    st.header("Individual Strike Price Analysis")
    st.subheader("Call Strike Analysis")
    st.write(call_df[call_df['strike'] == cs])
    st.subheader("Put Strike Analysis")
    st.write(put_df[put_df['strike'] == ps])
    
# with tab8:
#     strike1 = st.selectbox('Select Long Strike:', call_strike)
#     strike2 = st.selectbox('Select Short Strike:', call_strike)
#     option_type = st.selectbox('Select Call or Put', ('call', 'put'))
#     expirationDates = op.get_expiration_dates(symbol)
#     select_expiry_l = st.selectbox('Select Long Expiry', (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26))
#     select_expiry_s = st.selectbox('Select Short Expiry', (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26))
#     getVerticalSpreadPrice(symbol, option_type, select_expiry_l, strike1, strike2)
#     getDiagonalSpreadPrice(symbol, option_type, select_expiry_l, select_expiry_s,strike1, strike2)
#     for i,each in enumerate(expirationDates,start=1):
#         st.write("{}.{}".format(i,each))
