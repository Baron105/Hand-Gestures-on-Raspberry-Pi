import os
import shutil
import subprocess

ZIP_FILENAME = 'raspy.zip'
SOURCE_FOLDER = 'raspy'
REMOTE_USER = 'pi'
REMOTE_HOST = 'raspberrypi.local'
REMOTE_PATH = '~/project'

def main():
    if os.path.exists(ZIP_FILENAME):
        print(f'Removing existing {ZIP_FILENAME}...')
        os.remove(ZIP_FILENAME)

    print(f'Zipping {SOURCE_FOLDER} to {ZIP_FILENAME}...')
    shutil.make_archive(ZIP_FILENAME.replace('.zip', ''), 'zip', SOURCE_FOLDER)

    command = f'scp {ZIP_FILENAME} {REMOTE_USER}@{REMOTE_HOST}:{REMOTE_PATH}'

    try:
        print(f'Executing command: {command}')
        subprocess.run(command, shell=True, check=True)
        print('File transferred successfully.')
    except subprocess.CalledProcessError as e:
        print(f'An error occurred while transferring the file: {e}')

if __name__ == '__main__':
    main()
