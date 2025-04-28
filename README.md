# task-5
# Avocado Dataset

This repository contains the Avocado dataset, which includes data on avocado prices, sales volume, and region from various retail stores in the United States. The dataset can be used for data analysis, machine learning models, and data visualization tasks.

## Dataset Overview

The Avocado dataset consists of historical data on avocado sales, including the following key features:

- **Date**: The date the data was recorded.
- **AveragePrice**: The average price of avocados for that particular date.
- **TotalVolume**: The total volume of avocados sold.
- **4046**: The volume of 4046-sized avocados sold.
- **4225**: The volume of 4225-sized avocados sold.
- **4770**: The volume of 4770-sized avocados sold.
- **Type**: The type of avocado (Conventional or Organic).
- **Region**: The region where the avocado sales occurred.
  
## Notes
-------
import pandas as pd
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('avocado.csv')
-------
-------
selected_columns = ['AveragePrice', 'Total Volume', '4046', '4225', '4770']
sns.pairplot(df[selected_columns])
plt.show()
------
------
# Select only numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])
# Now compute correlation matrix
corr = numeric_df.corr()
# Plot heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
------
------
# Parse the Date column into datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Quick check
df.info()
# Filter for selected regions
regions_of_interest = ['California', 'New York', 'Chicago']
filtered_df = df[df['region'].isin(regions_of_interest)]
# Plot
plt.figure(figsize=(14,8))
sns.lineplot(data=filtered_df, x='Date', y='AveragePrice', hue='region')
plt.title('Average Avocado Price Over Time by Region')
plt.xlabel('Date')
plt.ylabel('Average Price ($)')
plt.legend(title='Region')
plt.show()
-----
-----
# Using FacetGrid for separate line charts
g = sns.FacetGrid(filtered_df, col="region", col_wrap=2, height=4)
g.map_dataframe(sns.lineplot, x="Date", y="AveragePrice")
g.set_titles("{col_name}")
g.set_axis_labels("Date", "Average Price ($)")
plt.tight_layout()
plt.show()
------
------
# Filter again for the same regions
regions_of_interest = ['California', 'New York', 'Chicago']
filtered_df = df[df['region'].isin(regions_of_interest)]

# Plot
plt.figure(figsize=(14,8))
sns.lineplot(data=filtered_df, x='Date', y='Total Volume', hue='region')
plt.title('Total Avocado Sales Volume Over Time by Region')
plt.xlabel('Date')
plt.ylabel('Total Volume Sold')
plt.legend(title='Region')
plt.show()
------
-----
# Separate line charts for each region
g = sns.FacetGrid(filtered_df, col="region", col_wrap=2, height=4)
g.map_dataframe(sns.lineplot, x="Date", y="Total Volume")
g.set_titles("{col_name}")
g.set_axis_labels("Date", "Total Volume Sold")
plt.tight_layout()
plt.show()
------
------
# Plot histogram for AveragePrice
plt.figure(figsize=(8,6))
sns.histplot(df['AveragePrice'], bins=30, kde=True)
plt.title('Distribution of Average Avocado Price')
plt.xlabel('Average Price ($)')
plt.ylabel('Frequency')
plt.show()
------
------
# Boxplot of AveragePrice grouped by region (selecting a few regions for clarity)
regions_of_interest = ['California', 'New York', 'Chicago']
filtered_df = df[df['region'].isin(regions_of_interest)]

plt.figure(figsize=(10,6))
sns.boxplot(data=filtered_df, x='region', y='AveragePrice')
plt.title('Average Avocado Price by Region')
plt.xlabel('Region')
plt.ylabel('Average Price ($)')
plt.show()
------
------
# Scatter plot between Total Volume and Average Price
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Total Volume', y='AveragePrice')
plt.title('Total Volume vs Average Price')
plt.xlabel('Total Volume Sold')
plt.ylabel('Average Price ($)')
plt.show()
-----
------
# Import seaborn and matplotlib if not already
import seaborn as sns
import matplotlib.pyplot as plt

# Create a jointplot
sns.jointplot(
    data=df,
    x='Total Volume',
    y='AveragePrice',
    kind='scatter',   # You can also try 'reg', 'hex', 'kde'
    height=8
)
plt.suptitle('Jointplot: Total Volume vs Average Price', y=1.02)
plt.show()
-----
-----
sns.jointplot(
    data=df,
    x='Total Volume',
    y='AveragePrice',
    kind='reg',   # Regression line
    height=8,
    scatter_kws={'alpha':0.5}
)
plt.suptitle('Jointplot with Regression Line', y=1.02)
plt.show()
-----
-----
sns.jointplot(
    data=df,
    x='Total Volume',
    y='AveragePrice',
    kind='kde',   # Kernel Density Estimate
    fill=True,
    cmap='rocket',
    height=8
)
plt.suptitle('Joint Density Plot (KDE)', y=1.02)
plt.show()
------
