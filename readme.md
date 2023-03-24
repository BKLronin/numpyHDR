# numpyHDR

* Micro Library for HDR image creation using the Mertens Fusion Alghoritm
* Optional Raspicam integration with picamera2 lib
* CLI Application "makehdr.py" 
* Processing time on  raspi zero ca 15 min with full res 12MP files.

## Intention
- numpy and PIL, picamera2 for Raspicam funtionality.
- Whenever dependencies of other bigger image librarys aren´t satisfyable.
- 32-bit armv6 or oudated Kernels etc.

## Function
*Use makehdr.py CLI app for testing before implementation*

- Captures an exposure bracket stack from the raspicam as fast as possible
- Alternatively from image sequences form file
- Either direct processing or save files in between
- Uses a conversion of the Mertens Fusion alghoritm
- Stretches Information to the full spectrum like Contrast or compression
- Lifts the shadows softly with an envelope.
- Clips to Image range and saves via PIL

## Setup
- Download and install dependencies from requirements.txt into env
- `pip install -r requirements.txt`
- Install picamera2 and libcamera
- `pip install picamera2`
- `pip install libcamera`

## Usage

- Start makehdr.py
- `python3 makehdr.py`
- Follow commandline instructions
- Be patient on Raspi zero

- Example on how to use the library can be found in makehdr.py

## Using the library
*Import numpyHDR modules*
- `import file_utility as file`
- `import numpyHDR as hdr`
- `import picamburst as pcb`

*Get image arrays from raspicam*
- `stack = pcb.get_exposure_stack()`

*Get image arrays from files*
- `stack = file.openImageList(path_list, True)`

*Process HDR with mertens fusion and post effects*
- `result = hdr.process(stack, 1, 1, 1, True)`

*Save Result to File*
- The filename gets extended by ".jpg" and "_hdr" automatically
- `file = file.saveResultToFile(result, 'hdr/result', 75)`
  

