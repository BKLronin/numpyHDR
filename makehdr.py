#!//usr/bin/python3
import numpyHDR as hdr
import file_utility as file

'''CLI application for HDR experiments'''

stack = []
select = input("Select Image Source: 1 - Raspicam, 2 - From File, 3 - Image sequence, 4 debug: ")

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

if int(select) == 3:
    path_list = []
    i = 0
    nr = input("How many images? :")
    image = input(f"Enter first filename without seq Nr and .jpg {i}: ")
    for i in range(int(nr)):
        filename = f"{image}{i}.jpg"
        path_list.append(filename)

if int(select) == 4:
    path_list = ['test_hdr0.jpg', 'test_hdr1.jpg', 'test_hdr2.jpg']

print(path_list)
stack = file.openImageList(path_list, True)

#Process HDR with mertens fusion and post effects
result = hdr.process(stack, 1, 1, 1, True)

#Save Result to File
file = file.saveResultToFile(result, 'hdr/result', 75)




