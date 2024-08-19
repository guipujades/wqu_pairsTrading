# Trading Strategy Backtesting and Performance Analysis

This repository contains a Python script designed for backtesting trading strategies, specifically focusing on a long/short equity strategy. The script includes functions to manage trading positions, handle cash flow, and calculate key performance metrics.

## Table of Contents

- [Introduction](#introduction)
- [Functions Overview](#functions-overview)
  - [find_last_day_of_month](#find_last_day_of_month)
  - [metrics](#metrics)
  - [plot_performance](#plot_performance)
  - [handle_cash_flow](#handle_cash_flow)
  - [handling_positions](#handling_positions)
  - [make_positions](#make_positions)

## Introduction

This script is intended for users interested in backtesting and analyzing trading strategies. It provides a set of functions to execute trades, calculate performance metrics, and visualize the results. The strategy is primarily focused on long/short equity trades, allowing users to simulate trading conditions and evaluate the effectiveness of their strategies.

## Functions Overview

find_last_day_of_month
Finds the last trading day of the specified month.
    Parameters:
        period (datetime): The target period (year and month).
        date_index (pd.DatetimeIndex): The index of dates to search within.
    Returns: The last trading day of the month or None if not found.

metrics
Calculates various performance metrics for the trading strategy.

    Parameters:
        df (pd.DataFrame): DataFrame containing strategy returns and benchmarks.
        rf (float): Risk-free rate for Sharpe/Sortino calculations (default 0).
        period_param (int): Periods per year (default 252).
    Returns: A dictionary of calculated performance metrics.

plot_performance
Plots cumulative performance of the strategy against Ibovespa and CDI.

    Parameters:
        returns (pd.Series): Strategy returns.
        ibov (pd.Series): Ibovespa returns.
        metrics_plot (pd.DataFrame): Metrics data to display.
        cdi (pd.Series): CDI returns.
        input_data_plot (str): Plot title or label.

handle_cash_flow
Updates the cash flow with information from the current trading round.

    Parameters:
        cash_flow (pd.DataFrame): DataFrame tracking cash flow.
        date (datetime): Current round date.
        equity (float): Total equity available.
        total_equity_usage_buy (float): Equity used for buying.
        total_equity_usage_short (float): Equity used for shorting.
        cash_buy (float): Cash remaining after buying.
        cash_short (float): Cash remaining after shorting.
        round_control (int): Current trading round.

handling_positions
Executes and handles trading positions, applying stop-loss if necessary.

    Parameters:
        pos_control (pd.DataFrame): Position data.
        cash_flow_control (pd.DataFrame): Cash flow control DataFrame.
        prices_enfoque (pd.DataFrame): Stock prices DataFrame.
        date (datetime): Current date.
        fee (float): Transaction fee.
        round_control (int): Current trading round.
        cum_cdi (pd.Series): Cumulative CDI for adjusting short positions.
        stop (bool): Apply stop-loss (default False).
    Returns:
        pd.DataFrame: Updated position data.
        float: Updated total equity.

make_positions
Creates trading positions based on available equity, ATR values, and other parameters.

    Parameters:
        use_equity (pd.DataFrame): Assets to be used for positions.
        prices_enfoque (pd.DataFrame): Stock prices.
        equity (float): Total equity available.
        date (datetime): Current date.
        fee (float): Transaction fee.
        round_control (int): Current trading round.
        buy (bool): Create buy positions (default True).
        adjust_equity (float): Equity proportion for positions (default 0.3).
        atr_values (pd.DataFrame): ATR values for adjusting positions (default None).
        long_biased (bool): Favor long positions (default False).
    Returns:
        pd.DataFrame: Created positions.
        float: Total equity usage.
        float: Remaining cash after creating positions.