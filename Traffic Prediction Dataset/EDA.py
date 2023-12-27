import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Traffic Volume dataset
traffic_volume_dataset = pd.read_csv('traffic.csv')

# Analyze the distribution of traffic volume based on different time intervals (hourly, daily, monthly)
traffic_volume_dataset['DateTime'] = pd.to_datetime(traffic_volume_dataset['DateTime'])
traffic_volume_dataset['Hour'] = traffic_volume_dataset['DateTime'].dt.hour
traffic_volume_dataset['Day'] = traffic_volume_dataset['DateTime'].dt.day
traffic_volume_dataset['Month'] = traffic_volume_dataset['DateTime'].dt.month

# Generate a text file for traffic volume by hour
traffic_volume_by_hour = traffic_volume_dataset.groupby('Hour')['Vehicles'].mean()
traffic_volume_by_hour.to_csv('traffic_volume_by_hour.txt', sep='\t', index=False)

# Generate a text file for traffic volume by day
traffic_volume_by_day = traffic_volume_dataset.groupby('Day')['Vehicles'].mean()
traffic_volume_by_day.to_csv('traffic_volume_by_day.txt', sep='\t', index=False)

# Generate a text file for traffic volume by month
traffic_volume_by_month = traffic_volume_dataset.groupby('Month')['Vehicles'].mean()
traffic_volume_by_month.to_csv('traffic_volume_by_month.txt', sep='\t', index=False)

# Plot traffic volume by hour
plt.figure(figsize=(10, 6))
sns.lineplot(x='Hour', y='Vehicles', data=traffic_volume_dataset)
plt.xlabel('Hour')
plt.ylabel('Traffic Volume')
plt.title('Traffic Volume by Hour')
plt.savefig('traffic_volume_by_hour.png')
plt.close()

# Plot traffic volume by day
plt.figure(figsize=(10, 6))
sns.lineplot(x='Day', y='Vehicles', data=traffic_volume_dataset)
plt.xlabel('Day')
plt.ylabel('Traffic Volume')
plt.title('Traffic Volume by Day')
plt.savefig('traffic_volume_by_day.png')
plt.close()

# Plot traffic volume by month
plt.figure(figsize=(10, 6))
sns.lineplot(x='Month', y='Vehicles', data=traffic_volume_dataset)
plt.xlabel('Month')
plt.ylabel('Traffic Volume')
plt.title('Traffic Volume by Month')
plt.savefig('traffic_volume_by_month.png')
plt.close()

# Explore how traffic volume varies across different junctions
plt.figure(figsize=(10, 6))
sns.lineplot(x='Junction', y='Vehicles', data=traffic_volume_dataset)
plt.xlabel('Junction')
plt.ylabel('Traffic Volume')
plt.title('Traffic Volume by Junction')
plt.savefig('traffic_volume_by_junction.png')
plt.close()

# Analyze the distribution of the number of vehicles
plt.figure(figsize=(10, 6))
sns.histplot(traffic_volume_dataset['Vehicles'], bins=20)
plt.xlabel('Number of Vehicles')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Vehicles')
plt.savefig('distribution_of_vehicles.png')
plt.close()
