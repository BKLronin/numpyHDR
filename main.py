import numpyHDR

#Testfile
hdr = numpyHDR.NumpyHDR()
liste = ['hdr/webcam23_3_2023_ev1.jpg','hdr/webcam23_3_2023_ev-2.jpg']
hdr.input_image = liste
hdr.output_path = 'hdr/fused_merten15'
hdr.compress_quality = 75
hdr.sequence(0.8, 0.5, 1, True)
