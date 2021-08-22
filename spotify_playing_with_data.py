# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 22:24:00 2021

@author: remns
"""
"""
IDEAS TODO

- Find out when a song was skipped => find all occurrences of the song to deduce
its total duration. Whenever that song was played a shorter time (lets give it
some percentual margin)AND another song started right away, it was skipped.
- Connect the streaming data to the playlists in some way
"""


from os import listdir
from os.path import isfile, join

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

unit_conversion_ms = {'sec':0.001, 'min':0.001/60, 'hour':0.001/3600}
entity_map = {'track':'trackName', 'artist':'artistName', 'date':'endTime'}

def total_time_played_individual(df, unit='sec', track=None, artist=None, date=None):
    """Return the total time Spotify played music or for a particular unique entry.
    -unit: Choose the unit of time measure ('sec', 'min', 'hour').
    -Use the optional arguments to obtain the time played for a particular track,
    artist, or day ('dd/mm/yyyy').
    """
    conversion = unit_conversion_ms[unit]
    if not(track or artist or date):
        return int(df['msPlayed'].sum() * conversion)
    elif track:
        return int(df.groupby('trackName')['msPlayed'].sum()[track] * conversion)
    elif artist:
        return int(df.groupby('artistName')['msPlayed'].sum()[artist] * conversion)
    elif date:
        time_filtered_df = df[df['endTime'].dt.strftime('%d/%m/%Y') == date]
        return int(time_filtered_df['msPlayed'].sum() * conversion)

def total_played(df, unit='sec', entity='track', sort=1, sort_by='sum'):
    """Return dataframe grouped by the entity column, adding time played and
    counting all occurences.
    sort:   1 => sort ascending order
            0 => do not sort
            -1 => sort descending order
    sort_by: either 'sum' (time played) or 'count'
    """
    sorting = {1:True, -1:False}
    entity_column = entity_map[entity]
    if sort:
        ascending = sorting[sort]
        return df.groupby(entity_column)['msPlayed'].agg(['sum', 'count']).sort_values(by=sort_by, ascending=ascending)
    else:
        return df.groupby(entity_column)['msPlayed'].agg(['sum', 'count'])

def count_reproduced(df, entity='track'):
    """Return dataframe grouped by the trackName column, counting all occurences"""
    entity_column = entity_map[entity]
    return streaming_df.groupby(entity_column).count()

def max_duration(df):
    """Obtain the longest playing time found for each song. Add new column with
    this information."""
    max_duration = streaming_df.groupby('trackName').msPlayed.aggregate(max)
    df['maxTime'] = df['trackName'].map(max_duration)
    return df

def perc_played(df):
    """For each occurrence of a song, what percentage of its maximum playing time
    it was played. Add new column with this information.
    Tacks whose  maxTime is lower than 30 sec. are never considered to have played."""
    if not 'maxTime' in df.columns.tolist():
        df = max_duration(df)
    df['percentagePlayed'] = 100 * (df['msPlayed'] / df['maxTime'])
    df.loc[df.maxTime < 3000, 'percentagePlayed'] = 0
    return df

def skipped(df):
    """Add a column where the value is True for songs that were skipped (played
    for less than 85% their total playback time)"""
    df['skipped'] = df['percentagePlayed'] < 85
    
def skipped_when(df, percentage, interval = None):
    """Number of tracks that where skipped before reaching the given percentage
    of playback time"""
    if not interval:
        return df[df.percentagePlayed <= percentage].count().iloc[:][0]
    else:
        return df[(df.percentagePlayed < percentage) &\
                  (df.percentagePlayed > percentage - interval)].count().iloc[:][0]

def plot_skip_when(df, step=5, lim_inf=0, lim_sup=99, kind='line'):
    """Different plots to visualize at which point tracks are skipped"""
    if not 'percentagePlayed' in df.columns.tolist():
        perc_played(max_duration(df))
    skip_pattern = {}
    sample = int((lim_sup - lim_inf) / step)
    if kind == 'line':
        for i in np.linspace(lim_inf, lim_sup, sample):
            skip_pattern[i] = skipped_when(streaming_df, i)
        plt.plot(*zip(*sorted(skip_pattern.items())))
    elif kind == 'bar':
        for i in np.linspace(lim_inf+step, lim_sup, sample):
            skip_pattern[i] = skipped_when(streaming_df, i, interval=step)
        plt.bar(*zip(*sorted(skip_pattern.items())))
    elif kind == 'hist':
        # for i in np.linspace(lim_inf+step, lim_sup, sample):
        #     skip_pattern[i] = skipped_when(streaming_df, i, interval=step)
        # print(skip_pattern.keys())
        
        plt.hist(df[(df['percentagePlayed'] >= lim_inf) & (df['percentagePlayed'] < lim_sup)]['percentagePlayed'], bins=sample)
    plt.show()
    
def plays_count(df):
    """Return a pandas Series object listing the tracks and the count
    of their plays"""
    if not 'percentagePlayed' in df.columns.tolist():
        perc_played(max_duration(df))
    count = df[df['percentagePlayed'] >= 80].groupby('trackName').count()
    count.sort_values(['endTime'], ascending = False, inplace=True)
    count.drop(count.columns.tolist()[1:], axis=1, inplace=True)
    count.rename(columns={count.columns.tolist()[0]: "Played"})
    return count

def daytime_played(df, info=False, plot=True):
    """Add information about only at what time of the day the song was played each time.
    - info = True:  return the resulting pandas Series with the timestamps as strings
                    if set to Flase, the information remains internally"""
    df['dayTime'] = df.loc[:,'endTime'].dt.strftime('%H:%M:%S')
    if plot:
        print('plotting')
        fig, ax = plt.subplots()
        ax.hist(pd.to_datetime(df.loc[:,'dayTime']).dt.hour, bins=24, range=(0,24), align='mid')
        plt.xticks(np.arange(0, 25, 2))
    if info:
            return df.loc[:,'dayTime']

def compare_record_to_playlist(tracks, playlists, checking_dict):
    for playlist in playlists:
        checking_dict[playlist.name] = True if\
            set(tracks).intersection(set(playlist.tracks.index)) == set(tracks)\
                else False
    return checking_dict

def identify_playlist_playing(stream, playlists, number_of_songs=3, return_names=True):
    """Identify for each entry of the streaming record whether a known playlist
    was being played and return a Series object containing the playlists' names.
        - stream: pandas DataFrame of the streaming history records.
        - playlists: list of Playlist objects.
        - number_of_songs: how many consecutively played tracks are considered
        the threshold to aknowledge that a playlist was being played.
        
    TODO: check whether the artist was always the same in the consecutive songs.
    Maybe Spotify was reproducing an artist and not a playlist, but since many of the
    songs are in a playlist, it would be recognized as a playlist"""
    playlist_played = [None] * len(stream)
    # Build a dictionary to store 'trackName'=True if all the analyzed songs are
    # in that playlist or 'trackName'=False otherwise (initially set all entries to false).
    playlist_names = [playlist.name for playlist in playlists]
    extra = 0
    suspect_artist = None
    # Consider numer_of_songs consecutive songs to check weather a playlist
    # was being played and update the flags in the dictionary.
    for idx in range(len(stream)-number_of_songs):
        current_playlist = None
        previous_playlist = None
        playlist_tracks = 0 #number of tracks consecutively played from the current playlist
        wild_tracks = 0 # number of tracks consecutively played from not known playlist
        check_flags = [False] * len(playlists)
        checking_dict = dict(zip(playlist_names, check_flags))
        #print('Check nº ', idx, checking_dict.values())
        tracks = stream.iloc[idx:idx+number_of_songs,:]['trackName'].tolist()
        #print(tracks)
        checking_dict = compare_record_to_playlist(tracks, playlists, checking_dict)
        #print(checking_dict.values())
        # In case the consecutives songs are all in more than one playlist,
        # check following songs until it can be found out which playlist was
        # actually being played.
        if extra > 0:   # if extra > 0 from a previous conflict, we shoul't repeat
                        # al the comparisons, just start from the last previous one
            extra -= 1
        while sum(list(checking_dict.values())) > 1:
           # print('Conflict')
            extra += 1
            tracks = stream.iloc[idx:idx+number_of_songs+extra,:]['trackName'].tolist()
            #print(tracks)
            checking_dict = compare_record_to_playlist(tracks, playlists, checking_dict)
        if sum(list(checking_dict.values())) > 0:
            current_playlist = list(checking_dict.keys())[list(checking_dict.values()).index(True)]
            # Store the playlist name in the list in the positions where tracks from it were played
            if return_names:
                for i in range(idx, idx+number_of_songs+extra):
                    playlist_played[i] = current_playlist
            else:
                # Store the results in the list that will form the Series of historical playlists:
                # the index corresponding to the found playlist's name in the playlist_names list
                playlist_name_index = playlist_names.index(list(checking_dict.keys())[list(checking_dict.values()).index(True)])
                for i in range(idx, idx+number_of_songs):
                    playlist_played[i] = playlist_name_index
        """if  not current_playlist and (current_playlist != previous_playlist):
            # The previous playlist stopped being played and the tracks playing
            # don't belong to any playlist
           suspect_artist = stream.iloc[idx+number_of_songs+extra-1,:]['artistName'] if\
               stream.iloc[idx+number_of_songs+extra-1,:]['artistName'] ==\
                   stream.iloc[idx+number_of_songs+extra-2,:]['artistName'] else None
        elif current_playlist:   # one more track from the current playlist is counted
            playlist_tracks += 1
        else:
            wild_tracks += 1
        if suspect_artist and current_playlist and\
            suspect_artist == stream.iloc[idx+number_of_songs+extra-1,:]['artistName']:
            cumulate_played = playlist_tracks + wild_tracks
            playlist_played[idx+number_of_songs+extra-cumulate_played:idx+number_of_songs+extra] = 'artist'
            playlist_tracks, wild_tracks = 0, 0
        previous_playlist = current_playlist"""
    return pd.Series(playlist_played)

def compare_daytime_plays(tracks, normalize=False):
    """Plot overlapping histograms for the daytime distribution of the
    tracks plays.
        - tracks: should be a list of Track objects
        - normalize=True: plot the histograms of the normalize data, such
        that their distribution in time is of comparable magnitude."""
    fig, ax = plt.subplots()
    for track in tracks:
        print('alpha: ', (1/len(tracks)))
        ax = track.daytime_plays(ax_in=ax, ax_ret=True, alpha=round((1/len(tracks)), 2), density=normalize)['ax']
    plt.xticks(np.arange(0, 25, 2))
    plt.legend(loc='upper right')
    return


def create_Track(df, name=None, index=None):
    """Takes in the full track record and an identifier of a track (name or index)"""
    if not name and not index:
        print('\nPlease, provide at least a name or an index identifying the track')
        return
    elif not name and index:
        name = df.iloc[index]['trackName']
    artist = df[df['trackName'] == name].artistName.iloc[0]
    duration = max_duration(df)['maxTime'].max()
    return Track(name, artist, duration, df[df['trackName'] == name].copy()) 

def create_Playlist(play_list, name):
    for playlist_dict in play_list:
        if playlist_dict['name'] == name:
            break
    return Playlist(playlist_dict)

def create_all_playlists(play_list):
    return [Playlist(playlist_dict) for playlist_dict in play_list]

class Track:
    """Pass a DataFrame including only this track's entries to create the object"""
    def __init__(self, name, artist, duration, record):
        self.name = name
        self.artist = artist
        self._duration = duration
        self.__record = record
        self.__record.reset_index(drop=True, inplace=True)
        self.__count = len(self.__record)

    def perc_played(self):
        """For each occurrence of the song, what percentage of its maximum playing time
        it was played. Add new column with this information.
        Tacks whose  maxTime is lower than 30 sec. are never considered to have played."""
        self.__record['percentagePlayed'] = 100 * (self.__record['msPlayed'] / self._duration)
        self.__record.loc[self.__record.maxTime < 3000, 'percentagePlayed'] = 0

    def times_skipped(self, time_limit=85):
        """How many times the song was skipped, according to the time limit
        -expressed as a percentage of the total duration- to consider a song skipped"""
        if not 'percentagePlayed'in self.__record.columns.tolist():
            self.perc_played()
        return self.__record[self.__record['percentagePlayed'] <= time_limit]

    def daytime_plays(self, info=False, plot=True, ax_in=None, ax_ret=False, alpha=1, density=False):
        """Add information about only at what time of the day the song was played each time.
        - info = True:  return the resulting pandas Series with the timestamps as strings
                        if set to Flase, the information remains internally
        - plot = True: plot a histogram of the plays in across the 24 hours"""
        returns = {'info':None, 'ax':None}
        self.__record['dayTime'] = self.__record.loc[:,'endTime'].dt.strftime('%H:%M:%S')
        if plot:
            print('plotting')
            if not ax_in:
                fig, ax = plt.subplots()
            else:
                ax = ax_in
            kwargs = dict(bins=24, align='mid', alpha=alpha, label=self.name, ec='k', histtype='stepfilled', density=density)
            ax.hist(pd.to_datetime(self.__record.loc[:,'dayTime']).dt.hour,  range=(0,24), **kwargs)
            plt.xticks(np.arange(0, 25, 2))
            if ax_ret:
                returns['ax'] = ax
        if info:
            returns['info'] = self.__record.loc[:,'dayTime']
        return returns

    def get_count(self):
        return self.__count
    
    def get_duration(self):
        return self._duration


class Playlist:
    def __init__(self, dict_playlist):
        self.name, self.updated = dict_playlist['name'], dict_playlist['lastModifiedDate']
        self.tracks = pd.DataFrame([track['track'] for track in dict_playlist['items']]).set_index('trackName')


# Path of the folder where the JSON files are saved
folder_path = r'F:\Documentos\Python_learning\Personal_projects\Spotify\MyData'

# READING AND STORING THE STREAMING HISTORY
# From that path, find the StreamingHistory files
streaming_files = [f for f in listdir(folder_path) if (isfile(join(folder_path, f))\
                   and 'StreamingHistory' in f)]

# The data is split into individual files, each containing 10000 elements max
# Read each file and transform it into a pandas DataFrame
dataframes_list = []

for file in streaming_files:
    with open(join(folder_path, file), encoding='utf-8') as file_content:
        streaming_list = json.load(file_content)
    dataframes_list.append(pd.DataFrame(streaming_list))

# Join all dataframes vertically (using concatenate) into a single DataFrame
streaming_df = pd.concat(dataframes_list, ignore_index=True)

# Parse the endTime column values from string to Timestamp (equivalent to Datetime)
streaming_df['endTime']=pd.to_datetime(streaming_df['endTime'])


# READING AND STORING THE PLAYLISTS
# From that path, find the Playlist files
playlist_files = [f for f in listdir(folder_path) if (isfile(join(folder_path, f))\
                   and 'Playlist' in f)]

# Read each file and collect all the playlists' dictionaries in a list
playlists_collection = []
for file in playlist_files:
    with open(join(folder_path, file), encoding='utf-8') as file_content:
        playlist_list = json.load(file_content)
    playlists_collection = playlists_collection + playlist_list['playlists']
