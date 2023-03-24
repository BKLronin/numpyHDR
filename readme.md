# numpyHDR

* Micro Library for HDR image creation using the Mertens Fusion Alghoritm*
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
- Download and isntall dependencies from requirements.txt into env
- 

## Usage

- Start make
  
Run function sequence() to start processing.
Example:

`hdr = numpyHDR.NumpyHDR()`

`hdr.input_image = photos/EV- stages/`

`hdr.compress_quality = 50`

`hdr.output_path = photos/result/`

`hdr.sequence()`

-returns: Nothing (Arrrr)