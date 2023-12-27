import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Traffic_Volume in city-in-seasons dataset
traffic_volume_seasons_dataset = pd.read_csv('Traffic_Volume.csv')

# Investigate the impact of holidays on traffic volume
traffic_volume_by_holiday = traffic_volume_seasons_dataset.groupby('holiday')['traffic_volume'].mean()
traffic_volume_by_holiday.to_csv('traffic_volume_by_holiday.txt', sep='\t', index=False)

# Analyze the relationship between temperature and traffic volume
plt.figure(figsize=(10, 6))
sns.scatterplot(x='temp', y='traffic_volume', data=traffic_volume_seasons_dataset)
plt.xlabel('Temperature')
plt.ylabel('Traffic Volume')
plt.title('Traffic Volume vs. Temperature')
plt.savefig('traffic_volume_vs_temperature.png')
plt.close()

# Explore how precipitation affects traffic volume
precipitation_columns = ['rain_1h', 'snow_1h']
for column in precipitation_columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=column, y='traffic_volume', data=traffic_volume_seasons_dataset)
    plt.xlabel(column.capitalize())
    plt.ylabel('Traffic Volume')
    plt.title(f'Traffic Volume vs. {column.capitalize()}')
    plt.savefig(f'traffic_volume_vs_{column}.png')
    plt.close()

# Examine the influence of cloud cover, weather types, and descriptions on traffic volume
weather_columns = ['clouds_all', 'weather_main', 'weather_description']
for column in weather_columns:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=column, y='traffic_volume', data=traffic_volume_seasons_dataset)
    plt.xlabel(column.capitalize())
    plt.ylabel('Traffic Volume')
    plt.title(f'Traffic Volume by {column.capitalize()}')
    plt.xticks(rotation=45)
    plt.savefig(f'traffic_volume_by_{column}.png')
    plt.close()

# Analyze the distribution of traffic volume based on different time intervals
traffic_volume_seasons_dataset['date_time'] = pd.to_datetime(traffic_volume_seasons_dataset['date_time'])
traffic_volume_seasons_dataset['hour'] = traffic_volume_seasons_dataset['date_time'].dt.hour

# Generate a text file for traffic volume by time intervals
traffic_volume_by_time_intervals = traffic_volume_seasons_dataset.groupby('hour')['traffic_volume'].mean()
traffic_volume_by_time_intervals.to_csv('traffic_volume_by_time_intervals.txt', sep='\t', index=False)

# Traffic volume visualization by time intervals
plt.figure(figsize=(10, 6))
sns.lineplot(x='hour', y='traffic_volume', data=traffic_volume_seasons_dataset)
plt.xlabel('Hour')
plt.ylabel('Traffic Volume')
plt.title('Traffic Volume by Time Intervals')
plt.savefig('traffic_volume_by_time_intervals.png')
plt.close()

# Create a correlation matrix and plot it as a heatmap
correlation_matrix = traffic_volume_seasons_dataset.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()