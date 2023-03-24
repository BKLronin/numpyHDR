import numpyHDR as hdr
import picamburst as pcb
import file_utility as file

'''Example of a complete HDR process starting with raspicam '''

#Get sequence from raspicam
stack = pcb.get_exposure_stack()

#Process HDR with mertens fusion and post effects
result = hdr.process(stack, 1, 1, 1, True)

#Save Result to File
file = file.saveResultToFile(result, 'hdr/', 75)


