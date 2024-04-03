# Command to open the default music player on macOS
import subprocess


def openScreenReader():
    # Command to open the screen reader on macOS
    screen_reader_command = "open -a VoiceOver"
    
    # Execute the command to open the screen reader
    subprocess.Popen(screen_reader_command, shell=True)


def openDefaultMusicPlayer():
    # Command to open the default music player on macOS
    music_player_command = "open -a Music"
    
    # Execute the command to open the default music player
    subprocess.Popen(music_player_command, shell=True)


# openScreenReader()
    

# Open Siri
subprocess.run(['open', '-a', 'Siri'])

# Open default email client
subprocess.run(['open', 'mailto:'])

# Take a screenshot
subprocess.run(['screencapture', 'screenshot.png'])

# Open Calendar app
subprocess.run(['open', '-a', 'Calendar'])

# Lock the computer
subprocess.run(['pmset', 'displaysleepnow'])