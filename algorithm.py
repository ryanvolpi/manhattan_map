import numpy as np
import pandas as pd
# Functions
def vector_angle_diff(a1, a2):
    return np.abs(np.where(abs(a1-a2)>180, abs(a1-a2)-360, abs(a1-a2)))

def vector_distance(lat1, lon1, lat2, lon2):
    return ((lat1-lat2)**2 + (lon1-lon2)**2)**0.5

def algorithm(_df, max_wait_mins=7, max_occupancy=6, angle_thresh=20, dist_thresh=0.15):
    agg_rides = []
    if len(_df.shape) == 1:
        return _df
    df = _df.copy()

    agg_ride_dict_ = {f'ride_id_{i}':float("NaN") for i in range(max_occupancy)}
    while df.shape[0]>1:
        agg_ride_dict = agg_ride_dict_.copy()
        i = df.index[0]
        agg_ride_dict['ride_id_0'] = i
        row = df.loc[i,:]
        initial_passengers = row['passenger_count']
        df.drop(index=i, inplace=True)

        # Find rows going in similar directions
        angle_diffs = vector_angle_diff(df['direction'], row['direction'])
        matches = df[angle_diffs<angle_thresh]

        # Find rows with aligned start points
        fractional_distance_start = vector_distance(
            lat1 = matches['pickup_latitude'],
            lon1 = matches['pickup_longitude'],
            lat2 = row['pickup_latitude'],
            lon2 = row['pickup_longitude'],
        ) / np.minimum(matches['euclidean_length'], row['euclidean_length'])
        time_diff_start = matches['tpep_pickup_datetime'] - row['tpep_pickup_datetime']
        aligned_start = (
                time_diff_start.dt.total_seconds().between(0,max_wait_mins*60)
                & (fractional_distance_start<dist_thresh)
        )

        # Find rows with aligned end points
        fractional_distance_end = vector_distance(
            lat1 = matches['dropoff_latitude'],
            lon1 = matches['dropoff_longitude'],
            lat2 = row['dropoff_latitude'],
            lon2 = row['dropoff_longitude'],
        ) / np.minimum(matches['euclidean_length'], row['euclidean_length'])
        time_diff_end = matches['tpep_dropoff_datetime'] - row['tpep_dropoff_datetime']
        aligned_end = (
                time_diff_end.dt.total_seconds().between(0,max_wait_mins*60)
                & (fractional_distance_end<dist_thresh)
        )

        matches.loc[:,'alignment'] = aligned_start.astype(int) + aligned_end.astype(int)
        matches = matches[matches['alignment']>0]
        matches.sort_values(by=['alignment','euclidean_length'],ascending=False, inplace=True)
        cumulative_passengers = matches['passenger_count'].cumsum()
        matches = matches[cumulative_passengers<=(max_occupancy-initial_passengers)]

        for i, idx in enumerate(matches.index):
            agg_ride_dict[f'ride_id_{i+1}'] = idx
        df.drop(index=matches.index.values, inplace=True)
        agg_rides.append(agg_ride_dict)
    return pd.DataFrame(agg_rides)