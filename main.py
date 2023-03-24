import numpyHDR

#Testfile
hdr = numpyHDR.NumpyHDR()
liste = ['test_hdr0.jpg','test_hdr1.jpg', 'test_hdr2.jpg']
hdr.input_image = liste
hdr.output_path = 'hdr/fused_merten17'
hdr.compress_quality = 75
hdr.sequence(1, 1, 1, True)
