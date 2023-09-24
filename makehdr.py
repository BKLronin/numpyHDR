#!//usr/bin/python3
import file_utility as file
import numpyHDR as hdr
import os

'''CLI application for HDR experiments'''


stack = []
select = input("Select Image Source: 1 - Raspicam, 2 - From File, 3 - Image sequence, 4 debug, 5 - compile Cython: ")

if int(select) == 1:
    import picamburst as pcb
    #Get sequence from raspicam
    stack = pcb.get_exposure_stack()

if int(select) == 2:
    path_list = []
    i= 0
    nr = input("How many images? :")
    for image in range(int(nr)):
        i += 1
        image = input(f"Enter filename {i}: ")
        path_list.append(image)
    stack = file.openImageList(path_list, True)

if int(select) == 3:
    path_list = []
    i = 0
    nr = input("How many images? :")
    image = input(f"Enter first filename without seq Nr and .jpg {i}: ")
    for i in range(int(nr)):
        filename = f"{image}{i}.jpg"
        path_list.append(filename)
    stack = file.openImageList(path_list, True)

if int(select) == 4:
    path_list = ['webcam25_3_2023_ev0.jpg','webcam25_3_2023_ev1.jpg','webcam25_3_2023_ev2.jpg']
    stack = file.openImageList(path_list, True)

if int(select) == 5:
    try:
        os.system('python3 setup.py build_ext --inplace')
    except Exception as e:
        print("Error while compiling cython function", e)
    print("Please restart")
    exit()

#print(path_list)

#Process HDR with mertens fusion and post effects, blur
#Set last value to false for double the speed but lesser blancaed hdr effect.
result = hdr.process(stack, 1, 1, 1, True, True)

#Save Result to File
file = file.saveResultToFile(result, 'hdr/result2', 75)




