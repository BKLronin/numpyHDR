import numpyHDR

#Testfile
hdr = numpyHDR.NumpyHDR()
liste = ['hdr/webcam20_3_2023_ev0.jpg','hdr/webcam20_3_2023_ev1.jpg','hdr/webcam20_3_2023_ev-2.jpg']
hdr.input_image = liste
hdr.output_path = 'hdr/fused_merten7'
hdr.compress_quality = 75
hdr.sequence(0.8, 0.1)
