# numpyHDR

*Micro Library for HDR image creation using the Mertens Fusion Alghoritm*

## Intention
- numpy and PIL only.
- Whenever dependencies of other bigger image librarys aren´t satisfyable.
- 32-bit armv6 or oudated Kernels etc.

## Function
- Uses a conversion of the Mertens Fusion alghoritm
- Stretches Information to the full spectrum like Contrast or compression
- Lifts the shadows softly with an envelope.
- Clips to Image range and saves via PIL

## Setup
- Download into project path and import the class for now.

## Usage

- Instantiate then set attributes:
    - input_image = List containing path strings including .jpg Extension
    - output_path = String ot Output without jpg ending
    - compress_quality = 0-100 Jpeg compression level defaults to 75
  
Run function sequence() to start processing.
Example:

`hdr = numpyHDR.NumpyHDR()`

`hdr.input_image = photos/EV- stages/`

`hdr.compress_quality = 50`

`hdr.output_path = photos/result/`

`hdr.sequence()`

-returns: Nothing (Arrrr)