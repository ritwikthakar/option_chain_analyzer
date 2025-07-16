import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- CLEANED UP INPUT DATA ---
# Assume call_df and put_df already exist and are visible in Tab 6

# Convert necessary columns to numeric and clean up
for df in [call_df, put_df]:
    for col in ['strike', 'openInterest', 'impliedVolatility', 'GEX']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['strike', 'openInterest', 'impliedVolatility', 'GEX'], inplace=True)
    df = df[df['openInterest'] > 0]

# Reassign cleaned dataframes
call_df_clean = call_df.copy()
put_df_clean = put_df.copy()

# --- FIGURE 1: Open Interest ---
fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=call_df_clean['strike'],
    y=call_df_clean['openInterest'],
    name='Call Open Interest',
    marker_color='green'))
fig1.add_trace(go.Bar(
    x=put_df_clean['strike'],
    y=put_df_clean['openInterest'],
    name='Put Open Interest',
    marker_color='red'))
fig1.update_layout(
    title='Open Interest by Strike',
    xaxis_title='Strike Price',
    yaxis_title='Open Interest',
    barmode='group')

# --- FIGURE 2: Implied Volatility Skew ---
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=call_df_clean['strike'],
    y=call_df_clean['impliedVolatility'],
    name='Call IV',
    mode='lines+markers',
    line=dict(color='green')))
fig2.add_trace(go.Scatter(
    x=put_df_clean['strike'],
    y=put_df_clean['impliedVolatility'],
    name='Put IV',
    mode='lines+markers',
    line=dict(color='red')))
fig2.update_layout(
    title='Implied Volatility Skew',
    xaxis_title='Strike Price',
    yaxis_title='Implied Volatility')

# --- FIGURE 3: Gamma Exposure ---
fig3 = go.Figure()
fig3.add_trace(go.Bar(
    x=call_df_clean['strike'],
    y=call_df_clean['GEX'],
    name='Call GEX',
    marker_color='blue'))
fig3.add_trace(go.Bar(
    x=put_df_clean['strike'],
    y=put_df_clean['GEX'],
    name='Put GEX',
    marker_color='orange'))
fig3.update_layout(
    title='Gamma Exposure (GEX)',
    xaxis_title='Strike Price',
    yaxis_title='GEX',
    barmode='relative')

# --- DISPLAY IN STREAMLIT TABS ---
tab1, tab2, tab3 = st.tabs(['Open Interest', 'Implied Volatility Skew', 'Gamma Exposure'])

with tab1:
    st.header("Open Interest")
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.header("Implied Volatility Skew")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.header("Gamma Exposure")
    st.plotly_chart(fig3, use_container_width=True)
    st.write("Total Call GEX:", round(call_df_clean['GEX'].sum(), 2))
    st.write("Total Put GEX:", round(put_df_clean['GEX'].sum(), 2))
    st.write("Net GEX:", round(call_df_clean['GEX'].sum() + put_df_clean['GEX'].sum(), 2))
