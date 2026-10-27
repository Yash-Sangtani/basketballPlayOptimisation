"""
Writes the list of games (one ``MM.DD.YYYY.AWAY.at.HOME`` per line) from the archives in
Tracking_Data/game logs. Run it as a script: ``python list_games.py [output_file]``.
Importing it does nothing (it used to rewrite allgames.txt on import).

The folder comes from config.py (TRACKING_DATA_DIR); the default output is Shots/allgames.txt.
"""

import os
import sys

import config


def list_files_in_folder(folder_path):
    files = os.listdir(folder_path)
    file_names_without_extensions = [os.path.splitext(file)[0] for file in files]
    return file_names_without_extensions


def main(output_file=None):
    file_names = list_files_in_folder(config.game_logs_dir())

    # Write the file names to a text file
    with open(output_file or config.game_list_path(), 'w') as file:
        for name in file_names:
            file.write(name + '\n')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
