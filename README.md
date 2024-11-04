# Bike Sharing Rebalancing Problem

This project focuses on optimizing the rebalancing of bikes in a Bike Sharing System (BSS). In such systems, bikes need to be redistributed among stations to meet demand patterns, ensuring that stations are neither overcrowded nor empty at peak times. This rebalancing task can be formulated as a mathematical optimization problem, which we aim to solve efficiently.

## Problem Overview

The Bike Sharing Rebalancing Problem is a variant of the Vehicle Routing Problem with Pickup and Delivery (VRPPD), tailored to address the needs of bike-sharing systems. Given a network of bike stations and a fleet of vehicles with limited capacity, the goal is to determine routes and actions (pickup/drop-off) that minimize operational costs while satisfying station demand.

The key objectives of the problem are:

- Minimize the total travel distance of rebalancing vehicles.
- Ensure that all the demands of the stations were met.
- Respect the capacity constraints of each vehicle used for rebalancing.

## Installation

Ensure you have Python 3.8+ installed. The project also requires the following libraries:

- mip: Mixed-Integer Programming library
- pandas: Data manipulation library
- numpy: Numerical operations library
- cbc: Branch-and-Cut Solver

Install dependencies by running:

```bash
pip install -r requirements.txt
```

To install cbc solver:

https://github.com/coin-or/Cbc?tab=readme-ov-file

## Usage

- Prepare Data: Place data files for bike stations, initial bike counts, and demand in the data/ directory.
- Run the Model on specific formulation.
