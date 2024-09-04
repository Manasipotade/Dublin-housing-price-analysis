# Dublin-housing-price-analysis
Introduction
As my partner and I embark on the exciting journey of finding our dream home, I wanted to ensure that we are making the most informed decision possible. The process of buying a home can be overwhelming, especially with the myriad of factors that influence housing prices across different regions in Ireland. With this in mind, I set out to conduct an in-depth analysis of the Irish housing market.

Ireland’s housing market has been a subject of considerable attention over the past decade, with significant fluctuations in prices influenced by economic conditions, government policies, and global events. This analysis is not only intended to guide us but also to help other potential homebuyers navigate the complexities of the market. In this blog, we delve into the intricacies of Ireland’s housing prices, supported by visual data, to uncover the key trends and insights that have shaped the market from 2010 to 2024.

Understanding the data
Install Required Libraries: You’ll need to install the following libraries if you don’t have them already.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
2. Loading the Data: Load the data into a pandas DataFrame.

file_path = '/content/drive/MyDrive/Dublin_Housing_Data.csv'
df = pd.read_csv(file_path)
3. Exploring the Data Structure: It’s important to note the data types of each column, as this will influence how we handle and clean the data.

# Display the first few rows to understand the structure
df.head()

# Check the shape of the dataset
df.shape
(674687, 9)

Data Cleaning and Preparation
Before diving into the visualizations, it is essential to discuss the data cleaning and preparation steps undertaken to ensure the accuracy and reliability of the analysis.

Data Cleaning Steps

Missing Value: There are two columns where data was missing. Since it's a categorical value, we chose to drop those rows.
df.isna().sum()
2. Removing Duplicates:

df.drop_duplicates(inplace=True)
3. Correcting Data Types: Data types must be accurate for the operations you intend to perform. For example, date columns should be in datetime format, and categorical variables should be properly encoded.

df['Date of Sale'] = pd.to_datetime(df['Date of Sale'], format='%d/%m/%Y')
df['Price'] = df['Price'].replace({'[€]': '', ',': ''}, regex=True).astype(float)
4. Feature Engineering: Feature engineering involves creating new features or modifying existing ones to improve the performance of your models or the clarity of your analysis.

# Extract the year and month from 'Date of Sale'
df['Year'] = df['Date of Sale'].dt.year
df['Month'] = df['Date of Sale'].dt.month
5. Dealing with Outliers: Outliers can distort statistical analyses and models, especially when they are not representative of the data population.

upper_limit = df['Price'].quantile(0.99)
lower_limit = df['Price'].quantile(0.01)

df = df[(df['Price'] > lower_limit) & (df['Price'] < upper_limit)]


plt.figure(figsize=(12, 6))
bp = plt.boxplot(df['Price'], patch_artist=True) 
for box in bp['boxes']:
    box.set(facecolor='#2f4b7c')
for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='#de425b')
plt.title('Boxplot of House Prices without outliers')
plt.grid(axis='y')
plt.show()
Data Visualization
1. Overall Housing Price Trends (2010–2024)
After a significant drop in 2011 and 2012, likely due to the aftermath of the global financial crisis, the housing market began to recover. There has been a steady increase in price since 2013. The graph shows a sharp increase in housing prices post-2020, which could be attributed to several factors like shift in demand due to COVID-19, the housing crisis and low interest rates.

Average housing price trend over the years across all counties
# Grouping the data by year and calculating the average price per year
avg_price_overall_by_year = df.groupby('Year')['Price'].mean()
plt.figure(figsize=(12, 6))
avg_price_overall_by_year.plot(marker='o', linestyle='-', color=['#2f4b7c'])
plt.title('Average Housing Price Trend in Ireland (Overall by Year)')
plt.ylabel('Average Price (€)')
plt.xlabel('Year')
plt.grid(True)
plt.show()
2. Average Housing price by county
Dublin remains the most expensive county. This could be driven by its status as capital and employment opportunities. Followed by Wicklow and Kildare which could be due to their proximity to the capital. These areas could be attractive for those who work in Dublin but prefer suburban living. Counties with lower average prices such as Longford and Leitrim, tend to be more rural where the demand for housing is less intense compared to urban areas.

County-wise Average Prices
Average Housing Price Trends in Ireland for top 10 counties
avg_price_by_county = df.groupby('County')['Price'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
avg_price_by_county.plot(kind='bar', color=['#2f4b7c'], edgecolor='black')
plt.title('Average Housing Price by County in Ireland')
plt.ylabel('Average Price (€)')
plt.xlabel('County')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.show()

# Grouping the data by year and county and calculating the average price per year per county
avg_price_by_year_county = df.groupby(['Year', 'County'])['Price'].mean().unstack()
top_10_counties = avg_price_by_year_county.mean().nlargest(10).index 
avg_price_by_year_county = avg_price_by_year_county[top_10_counties]
plt.figure(figsize=(12, 6))
avg_price_by_year_county.plot(marker='o', linestyle='-', colormap= 'coolwarm')
plt.title('Average Housing Price Trends in Ireland (By Year and County)')
plt.ylabel('Average Price (€)')
plt.xlabel('Year')
plt.legend(title='County', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()
3. Impact of property type on the price
New houses have a higher average price compared to second-hand houses. This is likely due to higher quality builds, modern amenities, energy-efficient designs and modern architectural styles. Initiatives such as the Help to buyer scheme and the first buyers scheme introduced in 2017 have further driven up the prices.

price_by_type = df.groupby(['Description of Property', 'County'])['Price'].mean().unstack()
price_by_type = price_by_type.reindex(columns=sorted(price_by_type.columns, key=lambda x: price_by_type[x].sum(), reverse=True))
plt.figure(figsize=(12, 8))
price_by_type.T.plot(kind='bar', figsize=(14, 8), color=['#2f4b7c', '#de425b'])
plt.title('Average Housing Prices by Property Type and County in Ireland')
plt.xlabel('County')
plt.ylabel('Average Price (€)')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Property Type', bbox_to_anchor=(0.97, 1))
plt.grid(axis='y')
plt.show()

4. Impact of HTB Scheme on Prices: New vs. Second-Hand Properties
The HTB scheme was introduced in 2017 which makes it easier for first-time buyers to purchase new homes. The data shows a noticeable increase in the prices of new properties post-2017.


new_houses = df[df['Description of Property'] == 'New Dwelling house /Apartment']
second_hand_houses = df[df['Description of Property'] == 'Second-Hand Dwelling house /Apartment']

# Split the data into pre- and post-2017 periods for both new and second-hand properties
pre_2017_new = new_houses[new_houses['Year'] < 2017]
post_2017_new = new_houses[new_houses['Year'] >= 2017]

pre_2017_second_hand = second_hand_houses[second_hand_houses['Year'] < 2017]
post_2017_second_hand = second_hand_houses[second_hand_houses['Year'] >= 2017]

avg_price_pre_2017_new = pre_2017_new['Price'].mean()
avg_price_post_2017_new = post_2017_new['Price'].mean()

avg_price_pre_2017_second_hand = pre_2017_second_hand['Price'].mean()
avg_price_post_2017_second_hand = post_2017_second_hand['Price'].mean()

price_comparison = pd.DataFrame({
    'Property Type': ['New Dwelling', 'New Dwelling', 'Second-Hand Dwelling', 'Second-Hand Dwelling'],
    'Period': ['Pre-2017', 'Post-2017', 'Pre-2017', 'Post-2017'],
    'Average Price (€)': [avg_price_pre_2017_new, avg_price_post_2017_new, avg_price_pre_2017_second_hand, avg_price_post_2017_second_hand]})

plt.figure(figsize=(12, 6))
price_comparison.pivot(index = "Period", columns = "Property Type", values = "Average Price (€)").plot(kind='bar', color=['#2f4b7c', '#de425b'])
plt.title('Impact of HTB Scheme on Prices: New vs. Second-Hand Properties')
plt.ylabel('Average Price (€)')
plt.xlabel('Period')
plt.grid(axis='y')
plt.xticks(rotation=0)
plt.show()
5. Average Housing Prices for New Property by County and Sale Type (Full Market vs. Not Full Market)
Dublin and Wicklow stand out as the counties with the highest average housing prices for new properties, regardless of the sale type. In counties like Wicklow and Kildare, the difference is much larger, indicating potential market distortions or a high demand for premium properties sold under special conditions. The differences between full market and not full market prices in several counties suggest the possible impact of government housing programs or subsidies. The higher prices for not full market properties in some areas might indicate that government-backed schemes or affordable housing initiatives are influencing the market in these regions.


new_houses = df[df['Description of Property'] == 'New Dwelling house /Apartment']
avg_price_by_county_and_sale_type = new_houses.groupby(['County', 'Not Full Market Price'])['Price'].mean().unstack()

avg_price_by_county_and_sale_type = avg_price_by_county_and_sale_type.reindex(
    index=avg_price_by_county_and_sale_type.sum(axis=1).sort_values(ascending=False).index)
plt.figure(figsize=(14, 8))
avg_price_by_county_and_sale_type.plot(kind='bar', figsize=(14, 8), color=['#2f4b7c', '#de425b'], edgecolor='black')
plt.title('Average Housing Prices for New Property by County and Sale Type (Full Market vs. Not Full Market)')
plt.xlabel('County')
plt.ylabel('Average Price (€)')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Not Full Market Price', bbox_to_anchor=(0.97, 1))
plt.grid(axis='y')
plt.show()
6. The COVID-19 Effect
The charts depict the impact of COVID-19 on housing prices (both during and post-pandemic).

Shift in Demand: More people sought larger homes due to remote work, leading to increased demand, especially in suburban and rural areas.
Low-Interest Rates: Central banks kept interest rates low, making borrowing cheaper and encouraging property purchases.
Supply Constraints: Construction delays and material shortages further tightened the housing supply, driving prices up.
By 2023, the year-over-year difference in average prices appears to stabilize, suggesting that the market might be reaching a new equilibrium after the rapid growth of the previous year



# Define the pre-COVID and post-COVID periods
pre_covid_period = df[df['Date of Sale'] < '2020-03-01']
post_covid_period = df[df['Date of Sale'] >= '2020-03-01']

# Calculate the average prices for pre-COVID and post-COVID periods
avg_price_pre_covid = pre_covid_period['Price'].mean()
avg_price_post_covid = post_covid_period['Price'].mean()

# Calculate the overall price trends during the COVID period (2020-2021)
covid_period_data = df[(df['Date of Sale'] >= '2020-03-01') & (df['Date of Sale'] <= '2021-12-31')]
avg_price_during_covid = covid_period_data.groupby('Month')['Price'].mean()

plt.figure(figsize=(12, 6))
avg_price_during_covid.plot(marker='o', linestyle='-', color=['#2f4b7c'])
plt.title('Average Housing Prices During COVID-19 (March 2020 - December 2021)')
plt.ylabel('Average Price (€)')
plt.xlabel('Month')
plt.grid(True)
plt.show()

# Grouping the data by year and county and calculating the average price per year per county
avg_price_by_year_county = post_covid_period.groupby(['Year', 'County'])['Price'].mean().unstack()
plt.figure(figsize=(14, 10))
avg_price_by_year_county.plot(marker='o', linestyle='-', colormap='RdYlBu')
plt.title('Average Housing Price Trends in Ireland Post Covid')
plt.ylabel('Average Price (€)')
plt.xlabel('Year')
plt.legend(title='County', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()
7. Year-Over-Year Difference in Average Housing Prices in Ireland
The negative values in 2010–2012 indicate a decline in housing prices during the post-crisis period. From 2013 onwards, there has been consistent growth in housing prices, with 2021 and 2022 showing the most substantial increases, likely due to the pandemic-driven demand surge.


# Calculate the average price per year
average_price_per_year = df.groupby(df['Date of Sale'].dt.year)['Price'].mean()

# Calculate the year-over-year difference in average prices
price_difference_per_year = average_price_per_year.diff().fillna(0)  # Fill NaN for the first year with 0

plt.figure(figsize=(10, 6))
price_difference_per_year.plot(kind='bar', color='#2f4b7c',edgecolor='black')
plt.title('Year-Over-Year Difference in Average Housing Prices in Ireland')
plt.xlabel('Year')
plt.ylabel('Price Difference (€)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()
Conclusion:
The analysis of Ireland’s housing market over the past decade reveals significant trends influenced by economic factors, government policies, and global events. The data shows a strong recovery post-2012, regional price disparities, and the impact of the COVID-19 pandemic on housing demand.
