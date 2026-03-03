# in vis.py, we visualize the breathing patterns and save them in the form of PDFs

# first, we import the libraries
import os
import argparse
import pandas as pd
from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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
        if ";" not in line:
            continue
        if "-" not in line:
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
    filtered = filtfilt(b, a, signal)
    return filtered


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
            if event["type"] in ["Hypopnea", "Obstructive Apnea"]:
                return event["type"]
            else:
                return "Normal"
    return "Normal"


# main function
def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, required=True,
                        help="Participant ID")
    args = parser.parse_args()
    participant_id = args.name
    participant_path = os.path.join(base_dir, "Data", participant_id)
    files = os.listdir(participant_path)

    flow_path = None
    thorac_path = None
    spo2_path = None
    event_path = None
    for file in files:
        if "Flow" in file and "Events" not in file:
            flow_path = os.path.join(participant_path, file)
        elif "Thorac" in file:
            thorac_path = os.path.join(participant_path, file)
        elif "SPO2" in file:
            spo2_path = os.path.join(participant_path, file)
        elif "Events" in file:
            event_path = os.path.join(participant_path, file)
    print("Events: ", event_path)

    print("The detected files are:\n")
    print("Flow: ", flow_path)
    print("Thorac: ", thorac_path)
    print("SpO2: ", spo2_path)
    flow_df = read_signal_file(flow_path)
    thorac_df = read_signal_file(thorac_path)
    spo2_df = read_signal_file(spo2_path)
    events_df = read_events_file(event_path)
    print("Total number of events: ", len(events_df))

    fs = 32  # airflow signal sampling rate (32 Hz)
    flow_df["filtered"] = bandpass_filter(
        flow_df["value"].values,
        lowcut=0.17,
        highcut=0.4,
        fs=fs
    )
    thorac_df["filtered"] = bandpass_filter(
        thorac_df["value"].values,
        lowcut=0.17,
        highcut=0.4,
        fs=fs
    )
    print("\nOverview: ")
    print("Flow shape: ", flow_df.shape)
    print("Thorac shape: ", thorac_df.shape)
    print("SpO2 shape: ", spo2_df.shape)
    print("\nTime range: ")
    print("Flow: ", flow_df.index.min(), "to", flow_df.index.max())
    print("SpO2: ", spo2_df.index.min(), "to", spo2_df.index.max())
    output_dir = os.path.join(base_dir, "Visualizations")
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{participant_id}_visualization.pdf")


    with PdfPages(pdf_path) as pdf:
        current_time = flow_df.index.min()
        end_time = flow_df.index.max()
        # here, we plot signals in 5-minute segments
        while current_time < end_time:
            segment_end = current_time + pd.Timedelta(minutes=5)
            flow_seg = flow_df.loc[current_time:segment_end]
            thorac_seg = thorac_df.loc[current_time:segment_end]
            spo2_seg = spo2_df.loc[current_time:segment_end]
            if len(flow_seg) == 0:
                break
            fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
            axes[0].plot(flow_seg.index, flow_seg["value"])
            axes[0].set_title("Nasal Airflow")
            axes[1].plot(thorac_seg.index, thorac_seg["value"], color="orange")
            axes[1].set_title("Thoracic Movement")
            axes[2].plot(spo2_seg.index, spo2_seg["value"], color="black")
            axes[2].set_title("SpO2")
            segment_events = events_df[
                (events_df["end"] >= current_time) &
                (events_df["start"] <= segment_end)
                ]
            for _, event in segment_events.iterrows():
                color = "yellow" if "Hypopnea" in event["type"] else "red"
                for ax in axes:
                    ax.axvspan(
                        max(event["start"], current_time),
                        min(event["end"], segment_end),
                        color=color,
                        alpha=0.35
                    )
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            current_time = segment_end
    print("The visualization PDF is saved at:", pdf_path)


if __name__ == "__main__":
    main()