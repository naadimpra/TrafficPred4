import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

traffic_volume_dataset = pd.read_csv('Train.csv')

# Distribution of traffic volume based on different time intervals
traffic_volume_dataset['date_time'] = pd.to_datetime(traffic_volume_dataset['date_time'])
traffic_volume_dataset['hour'] = traffic_volume_dataset['date_time'].dt.hour

# Generate a text file for traffic volume by time intervals
traffic_volume_by_time_intervals = traffic_volume_dataset.groupby('hour')['traffic_volume'].mean()
traffic_volume_by_time_intervals.to_csv('traffic_volume_by_time_intervals.txt', sep='\t', index=False)

# Traffic volume visualization by time intervals
plt.figure(figsize=(10, 6))
sns.lineplot(x='hour', y='traffic_volume', data=traffic_volume_dataset)
plt.xlabel('Hour')
plt.ylabel('Traffic Volume')
plt.title('Traffic Volume by Time Intervals')
plt.savefig('traffic_volume_by_time_intervals.png')
plt.close()

# Investigate whether traffic volume is affected by holidays
traffic_volume_by_holiday = traffic_volume_dataset.groupby('is_holiday')['traffic_volume'].mean()
traffic_volume_by_holiday.to_csv('traffic_volume_by_holiday.txt', sep='\t', index=False)

# Analyze the relationship between air pollution levels and traffic volume
plt.figure(figsize=(10, 6))
sns.scatterplot(x='air_pollution_index', y='traffic_volume', data=traffic_volume_dataset)
plt.xlabel('Air Pollution Index')
plt.ylabel('Traffic Volume')
plt.title('Traffic Volume vs. Air Pollution Index')
plt.savefig('traffic_volume_vs_air_pollution_index.png')
plt.close()

# Explore how weather conditions impact traffic volume
weather_columns = ['humidity', 'wind_speed', 'wind_direction', 'visibility_in_miles', 'dew_point', 'temperature', 'rain_p_h', 'snow_p_h', 'clouds_all']
for column in weather_columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=column, y='traffic_volume', data=traffic_volume_dataset)
    plt.xlabel(column.capitalize())
    plt.ylabel('Traffic Volume')
    plt.title(f'Traffic Volume vs. {column.capitalize()}')
    plt.savefig(f'traffic_volume_vs_{column}.png')
    plt.close()

# Analyze the relationship between cloud cover, weather types, weather descriptions, and traffic volume
weather_columns = ['clouds_all', 'weather_type', 'weather_description']
for column in weather_columns:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=column, y='traffic_volume', data=traffic_volume_dataset)
    plt.xlabel(column.capitalize())
    plt.ylabel('Traffic Volume')
    plt.title(f'Traffic Volume by {column.capitalize()}')
    plt.xticks(rotation=45)
    plt.savefig(f'traffic_volume_by_{column}.png')
    plt.close()

plt.figure(figsize=(12, 8))
correlation_matrix = traffic_volume_dataset.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()