# MeMo

## Application Description
Python Project (GUI) for visualizing the interactive MEMO demonstrator. The sensor values of the demonstrator's membrane are read out via a Raspberry Pi + Adafruit 1115 analogue digital converters. A previously trained AI model calculates the current load position on the membrane depending on the current sensor values and shows these in a plot. The project also contains the scripts for recording the training data and training the AI model.

## Contact Persons GUI / Demonstrator
- Hung, Le Xuan (SMA) -> Project information
- Koppelmann, Karl (SMA) -> Project information, design demonstrator
- Ngo, Phuong Ngoc -> Project information
- Florian Schmidt (SMA) -> Software Raspberry, GUI
- Baehr, Thomas (SMA) -> Circuits, Electronics

## Getting started

### Starting Live GUI
1. run the gui by starting following file: [Python-File](https://gitlab.mn.tu-dresden.de/sma/demonstratoren_techtextil/memo/-/blob/main/src/main/python/live_demo.py?ref_type=heads) -> to run .py Files on Raspberry Pi open Terminal an execute the following:
    - Navigate via `cd filepath` to project folder
    - Create virtual environment, if not createt yet by executing: `virtualenv venv`
    - Activate virtual environment by executing `source venv/bin/activate`
    - Run file by executing `python3 src/main/python/live_demo.py` 
2. Cancel script by pressing: `q`, go full-screen by pressing `f`

### Getting raw training data
- To create training data start following [Python-File](https://gitlab.mn.tu-dresden.de/sma/demonstratoren_techtextil/memo/-/blob/main/src/main/python/reading_sensordata_pi_adafruit.py?ref_type=heads)
- change user inputs defined at the top of the script. This changes the names of the files created by the script 

### Extracting useable datapoints 
- After getting the raw training data you have to extract the relevant data points by executing [Python-File](https://gitlab.mn.tu-dresden.de/sma/demonstratoren_techtextil/memo/-/blob/main/src/main/python/extracting_datapoints_pi.py?ref_type=heads)

### Train & save model
- Train your model by executing following file [Python-File](https://gitlab.mn.tu-dresden.de/sma/demonstratoren_techtextil/memo/-/blob/main/src/main/python/train_demonstrator_model.py?ref_type=heads)

## Setup Instructions for Raspberry Pi
The SD cards provide the main storage. Many data is written & read. This can lead to a failure of the SD card. Here are the steps explained to set up a new SD card:

- Load a micro SD card (>8GB) with Raspberry Pi system via "Raspberry Pi Imager" software
- Start Raspberry Pi and define username + password
- after system start connect via VPN/WEB with internet (WLAN)
- Install updates via `sudo apt-get update`, `sudo apt-get upgrade` and `sudo reboot`
- Install virtual environment via `sudo apt-get install python3-virtualenv`
- navigate to project folder after executing `git clone` command and create virtual environment via `virtualenv venv`
- activate virtual environment via `source venv/bin/activate`
- navigate to the [src](https://gitlab.mn.tu-dresden.de/sma/demonstratoren_techtextil/smartmeditex/-/tree/main/src?ref_type=heads) folder of the project and install all necessary Python packages via `pip install -r requirements.txt`

# Creating a Desktop Shortcut
- create a start_gui.sh file and perform activation of the virtual environment + start of the Python file there (file is also located in repository in the [doc](https://gitlab.mn.tu-dresden.de/sma/demonstratoren_techtextil/smartmeditex/-/tree/main/doc?ref_type=heads) folder)
- use `chmod +x /path/to/your/start_script.sh` to "activate" the shell file
