"""
Operate EventTracking label files
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from moviepy.video.io.VideoFileClip import VideoFileClip
from typing import Union, List, Tuple

class Event:
        def __init__(self, index, label, start_time, end_time, back_ground='#000000'):
            self.Index = index
            self.Label = label
            self.StartTime = start_time if isinstance(start_time, datetime) else datetime.strptime(self.format_time(start_time), "%H:%M:%S.%f") 
            self.EndTime = end_time if isinstance(end_time, datetime) else datetime.strptime(self.format_time(end_time), "%H:%M:%S.%f")
            self.back_ground = back_ground
        
        def format_time(self, time_str):
            if '.' not in time_str:
                return f"{time_str}.000000"
            else:
                return time_str[:-1]

class Events:

    def __init__(self, file: str = None):
        self.events = []
        if file:
            self.load_json(file)

    def load_json(self, file: str):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            for record in json_data:
                index = record['Index']
                back_ground = record.get('BackGroundBrush', '#000000')  # Default value
                label = record['Label']
                start_time = record['StartTime']
                end_time = record['EndTime']
                if None in (index, label, start_time, end_time) or "00:00:00" in (index, label, start_time, end_time):
                    print("Invalid record in JSON data. 检查起止时间. Skipping. File: {}".format(file))
                    continue
                self.events.append(Event(index, label, start_time, end_time, back_ground))
            if self.has_overlap('儿童发音'):
                print("The label 儿童发音 has overlap.")
        except Exception as e:
            print(f"Error loading JSON file: {e}")

    def save_to_json(self, file):
        """
        Save event list as JSON file        
        Args:
            file (str): Filename to save (including path).
        Returns:
            None: This function has no return value, writes directly to file.
        
        """
        json_data = []
        for event in self.events:
            json_data.append({
                'BackGroundBrush': event.back_ground,
                'Label': event.Label,
                'StartTime': event.StartTime.strftime("%H:%M:%S.%f"),
                'EndTime': event.EndTime.strftime("%H:%M:%S.%f"),
                'Index': event.Index
            })
        
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

    # Check if a label has time overlap
    def has_overlap(self, label: str) -> bool:
        """Check if a given label has time overlap

        Args:
            label (str): Specified label

        Returns:
            bool: Whether there is overlap
        """
        time_sequence = []
        for event in self.events:
            if event.Label == label:
                time_sequence.extend([(event.StartTime - datetime(1900, 1, 1)).total_seconds(), (event.EndTime - datetime(1900, 1, 1)).total_seconds()])
            
        flag = False
        for i in range(1, len(time_sequence)):
            if time_sequence[i] < time_sequence[i - 1]:
                minutes, seconds = divmod(time_sequence[i-1], 60)
                formatted_time = f"{int(minutes)}:{int(seconds)}"
                print(formatted_time)
                flag = True
        return flag
       

    # Filter events by label
    def filter_by_label(self, range_labels=None, exclude_range_labels=None, stay_labels=None, exclude_labels=None, duration_min=None, duration_max=None):
        """
        Event filter
        range_labels: Search range. (Keep all events within the range of events in range_labels) Default value is all ranges
        exclude_range_labels: Exclude range. (Exclude all events whose label names are in exclude_range_labels) Default value is empty
        stay_labels: Keep labels. (Keep all events whose label names are in stay_labels) Default value is all labels
        exclude_labels: Delete range. (Treat events with labels in exclude_labels as interference and delete them) Default value is empty
        # Example: events = events.filter_by_label(stay_labels=['纵向走路'])
        """
        if not self.events:
            print("Events is empty.")
            return Events()  # Return an empty Events object
        
        # Determine range
        filtered_events = Events()
        if range_labels:
            range_events = self.filter_by_label(stay_labels=range_labels)
            for r_event in range_events.events:
                filtered_events.events.extend([event for event in self.events 
                                               if not event.EndTime<=r_event.StartTime and not event.StartTime>=r_event.EndTime])
        else:
            filtered_events.events = self.events
        
        # Exclude range
        filtered_events_2 = Events()
        if exclude_range_labels:
            exclude_range_labels = [label for label in exclude_range_labels if filtered_events.has_label(label)]
        if exclude_range_labels:
            exclude_range_events = filtered_events.filter_by_label(stay_labels=exclude_range_labels)
            for event in filtered_events.events:
                if all([(event.EndTime<e_event.StartTime or event.StartTime>e_event.EndTime) for e_event in exclude_range_events.events]) and event.Label not in exclude_range_labels:
                    filtered_events_2.events.append(event)
            filtered_events.events = filtered_events_2.events

        # Keep specified labels
        if stay_labels and not exclude_labels:
            events_list = []
            events_list.extend([event for event in filtered_events.events if event.Label in stay_labels])
            filtered_events.events = events_list

        # Exclude label segments
        if exclude_labels and not stay_labels:
            exclude_events = filtered_events.filter_by_label(stay_labels=exclude_labels)

            for e_event in exclude_events.events:
                events_list = []
                for event in filtered_events.events:
                    if event.Label in exclude_labels: continue

                    # If the start time of exclusion is before the useful segment
                    if e_event.StartTime <= event.StartTime and e_event.EndTime > event.StartTime and e_event.EndTime < event.EndTime:
                        events_list.append(Event(event.Index, event.Label, e_event.EndTime, event.EndTime))

                    # If the end time of exclusion is within the useful segment
                    elif event.StartTime < e_event.StartTime and e_event.EndTime < event.EndTime:
                        events_list.append(Event(event.Index, event.Label, event.StartTime, e_event.StartTime))
                        events_list.append(Event(event.Index, event.Label, e_event.EndTime, event.EndTime))

                    # If the end time of exclusion is after the useful segment
                    elif event.StartTime < e_event.StartTime and e_event.StartTime < event.EndTime and e_event.EndTime >= event.EndTime:
                        events_list.append(Event(event.Index, event.Label, event.StartTime, e_event.StartTime))

                    # If the excluded segment contains the useful segment
                    elif e_event.StartTime < event.StartTime and e_event.EndTime >= event.EndTime:
                        continue
                    else:
                        events_list.append(Event(event.Index, event.Label, event.StartTime, event.EndTime))
                filtered_events.events = events_list

        # Keep label segments + exclude label segments
        if stay_labels and exclude_labels:
            filtered_events = filtered_events.filter_by_label(range_labels=stay_labels, stay_labels=stay_labels+exclude_labels)
            filtered_events = filtered_events.filter_by_label(exclude_labels=exclude_labels)

        # Select duration
        if duration_min or duration_max:
            # Convert nan values
            events_list = []
            for event in filtered_events.events:
                duration = event.EndTime-event.StartTime
                if (duration_min is None or duration.total_seconds() >= duration_min) and (duration_max is None or duration.total_seconds() <= duration_max):
                    events_list.append(event)
            filtered_events.events = events_list
        
        return filtered_events
    

    # Return time list of events [[start_time, end_time]]
    def get_time_list(self):
        if not self.events:
            print("Events is empty.")
            return []
        
        time_list_seconds = []
        for event in self.events:
            start_time_seconds = (event.StartTime - datetime(1900, 1, 1)).total_seconds()
            end_time_seconds = (event.EndTime - datetime(1900, 1, 1)).total_seconds()
            time_list_seconds.append([start_time_seconds, end_time_seconds])

        return time_list_seconds
    
    # Return frame position list of events [[start_frame, end_frame]]
    def get_frame_list(self, sr):
        """
        Return frame position list of events [[start_frame, end_frame]]
        sr: Sample rate
        """
        if not self.events:
            print("Events is empty.")
            return []

        def convert(sec, sr):
            return int(sec * sr)
        return [[convert(start_time, sr), convert(ent_time, sr)] for [start_time, ent_time] in self.get_time_list()]

    def has_label(self, label: str) -> bool:
        '''
        Check if a specific label exists
        Args:
            label (str): Event label to check
        Return:
            bool: Whether the label exists
        '''
        return any(event.Label == label for event in self.events)

    # Print all events
    def print_events(self):
        if not self.events:
            print("Events is empty.")
            return
        
        for event in self.events:
            print(f"Index: {event.Index}, Label: {event.Label}, StartTime: {event.StartTime.strftime('%H:%M:%S.%f')[:-3]}, EndTime: {event.EndTime.strftime('%H:%M:%S.%f')[:-3]}")


    # Output event statistics
    def events_statistics(self, is_print=True):
        """
        Output event statistics
        """
        if not self.events:
            print("Events is empty.")
            return

        label_duration = {}

        for event in self.events:
            if event.Label not in label_duration: 
                label_duration[event.Label] = [(event.EndTime - event.StartTime).total_seconds()]
            else:
                label_duration[event.Label].append((event.EndTime - event.StartTime).total_seconds())

        # List for storing statistical information
        result_data = []

        for label, duration in label_duration.items():
            # Print label statistics
            if is_print:
                print(f"Label: {label}, Count: {len(duration)}, Total-duration: {sum(duration):.3f} s, Mean-duration: {sum(duration)/len(duration):.3f}, Max-duration: {max(duration):.3f}, Min-duration:{min(duration):.3f}")
            
            result_data.append({
            'Label': label,
            'Count': len(duration),
            'Total-duration': sum(duration),
            'Mean-duration': sum(duration) / len(duration),
            'Max-duration': max(duration),
            'Min-duration': min(duration)
            })

        result_df = pd.concat([pd.DataFrame([row]) for row in result_data], ignore_index=True)
        return result_df

def extract_frames_by_positions(original_frames: Union[list, pd.DataFrame, np.ndarray], 
                                position_list: List[Tuple[int, int]], 
                                zero_length: int = 0):
    """
    original_frames (list, DataFrame, ndarray): Time series representing original frames
    position_list (List[Tuple[int, int]]): Position list, each element is a tuple containing start frame and end frame
    zero_length (int): Number of zeros to insert between two segments of data
    
    Return extracted frames (list or DataFrame), with a certain number of zeros inserted in between
    """
    if isinstance(original_frames, (list, np.ndarray)):
        extracted_frames = []
        for i, (start_frame, end_frame) in enumerate(position_list):
            # Check if index is valid
            if start_frame < 0 or end_frame >= len(original_frames):
                raise ValueError(f"Index {start_frame}-{end_frame} out of range")
            # Extract frames
            extracted_frames.extend(original_frames[start_frame:end_frame + 1])
            # If not the last segment, insert zero_length zeros
            if i < len(position_list) - 1 and zero_length > 0:
                extracted_frames.extend([0] * int(zero_length))
        return extracted_frames
    
    elif isinstance(original_frames, pd.DataFrame):
        extracted_frames_list = []
        for i, (start_frame, end_frame) in enumerate(position_list):
            # Check if index is valid
            if start_frame < 0 or end_frame >= len(original_frames):
                raise ValueError(f"Index {start_frame}-{end_frame} out of range")
            # Extract frames
            extracted_frames_list.append(original_frames.iloc[start_frame:end_frame + 1])
            # If not the last segment, insert zero_length rows of all-zero data
            if i < len(position_list) - 1 and zero_length > 0:
                zero_df = pd.DataFrame(0, index=range(int(zero_length)), columns=original_frames.columns)
                extracted_frames_list.append(zero_df)
        # Use pd.concat to concatenate
        extracted_frames = pd.concat(extracted_frames_list).reset_index(drop=True)
        return extracted_frames
    
    else:
        raise TypeError("original_frames must be of type list, np.ndarray or pd.DataFrame")


def merge_json_files(base_json_path: str, add_json_path: str, base_video_path: str, output_json_path: str):
    """
    Merge two json files, base_json_path is the base file, add_json_path is the file to be merged
    base_video_path: Base video path
    output_json_path: Output file path
    """
    base_events = Events(base_json_path)
    add_events = Events(add_json_path)

    def get_duration_from_moviepy(url: str) -> timedelta:
        clip = VideoFileClip(url)
        return timedelta(seconds=clip.duration)
    
    try:
        duration_timedelta = get_duration_from_moviepy(base_video_path)
        
        for event in add_events.events:
            event.StartTime += duration_timedelta
            event.EndTime += duration_timedelta
        
        base_events.events.extend(add_events.events)
        base_events.save_to_json(output_json_path)
        print('Merge json file successfully.')
    except Exception as e:
        print(f"Error merging JSON files: {e}")

    


    


