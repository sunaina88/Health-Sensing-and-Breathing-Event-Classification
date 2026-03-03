# in create_dataset.py, we process the raw breathing signals and events of each participant

# first, we import the libraries
import os
import argparse
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt


# this function reads the raw signal files, formats their timestamps and converts them to dataframes
def read_signal_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    data_start = None
    for i, line in enumerate(lines):
        if line.strip() == "Data:":
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError(f"No 'Data:' section is found in {filepath}.")
    df = pd.read_csv(
        filepath,
        sep=";",
        skiprows=data_start,
        names=["timestamp", "value"],
        engine="python"
    )
    df["timestamp"] = pd.to_datetime(
        df["timestamp"].str.strip(),
        format="%d.%m.%Y %H:%M:%S,%f"
    )
    df["value"] = df["value"].astype(float)
    df.set_index("timestamp", inplace=True)
    return df


# this function reads the raw event files and extracts breathing event intervals
def read_events_file(filepath):
    events = []
    with open(filepath, "r") as f:
        lines = f.readlines()
    for line in lines:
        if ";" not in line or "-" not in line:
            continue
        parts = line.strip().split(";")
        if len(parts) < 3:
            continue
        time_part = parts[0].strip()
        event_type = parts[2].strip()
        start_str, end_time_only = time_part.split("-")
        try:
            start_time = pd.to_datetime(
                start_str.strip(),
                format="%d.%m.%Y %H:%M:%S,%f"
            )
            date_part = start_time.strftime("%d.%m.%Y")
            end_str = f"{date_part} {end_time_only.strip()}"
            end_time = pd.to_datetime(
                end_str,
                format="%d.%m.%Y %H:%M:%S,%f"
            )
            events.append({
                "start": start_time,
                "end": end_time,
                "type": event_type
            })
        except:
            continue
    return pd.DataFrame(events)


# this function uses a Butterworth bandpass filter to keep breathing frequencies (within 0.17 Hz - 4 Hz)
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


# this function splits the filtered signals into overlapping windows
def create_windows(df, window_size, step_size):
    windows = []
    indices = []
    for start in range(0, len(df) - window_size, step_size):
        end = start + window_size
        window_data = df["filtered"].iloc[start:end].values
        window_start_time = df.index[start]
        window_end_time = df.index[end]
        windows.append(window_data)
        indices.append((window_start_time, window_end_time))
    return windows, indices


# this function labels windows based on event overlaps
def label_window(start_time, end_time, events_df):
    window_duration = (end_time - start_time).total_seconds()
    for _, event in events_df.iterrows():
        overlap_start = max(start_time, event["start"])
        overlap_end = min(end_time, event["end"])
        overlap = (overlap_end - overlap_start).total_seconds()
        if overlap > 0.5 * window_duration:
            return event["type"]
    return "Normal"


# the main function
def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, required=True,
                        help="Participant ID")
    args = parser.parse_args()
    participant_id = args.name
    participant_path = os.path.join(base_dir, "Data", participant_id)
    print("This participant is being processed: ", participant_id)

    files = os.listdir(participant_path)
    flow_path = None
    event_path = None
    for file in files:
        if "Flow" in file and "Events" not in file:
            flow_path = os.path.join(participant_path, file)
        elif "Events" in file:
            event_path = os.path.join(participant_path, file)

    flow_df = read_signal_file(flow_path)
    events_df = read_events_file(event_path)
    print("Total number of events: ", len(events_df))

    fs = 32 # airflow signal sampling rate (32 Hz)
    flow_df["filtered"] = bandpass_filter(
        flow_df["value"].values,
        lowcut=0.17,
        highcut=0.4,
        fs=fs
    )
    window_size = 30 * fs
    step_size = 15 * fs
    windows, time_ranges = create_windows(flow_df, window_size, step_size)
    labels = []
    for start_time, end_time in time_ranges:
        labels.append(label_window(start_time, end_time, events_df))
    X = np.array(windows)
    y = np.array(labels)
    label_map = {
        "Normal": 0,
        "Hypopnea": 1,
        "Obstructive Apnea": 2
    }
    cleaned_labels = []
    for l in y:
        if l in label_map:
            cleaned_labels.append(l)
        else:
            cleaned_labels.append("Normal")
    y_encoded = np.array([label_map[l] for l in cleaned_labels])
    dataset_dir = os.path.join(base_dir, "Dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    np.save(os.path.join(dataset_dir, f"{participant_id}_X.npy"), X)
    np.save(os.path.join(dataset_dir, f"{participant_id}_y.npy"), y_encoded)
    print(f"The dataset for {participant_id} has been saved.")
    print("Shape:", X.shape)


if __name__ == "__main__":
    main()