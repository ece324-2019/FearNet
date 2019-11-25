import PySimpleGUI as sg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ApplicablePhobias = []
# # All the stuff inside your window.
# layout = [  [sg.Text('Please select which of the following phobias apply to you:')],
#             [sg.Button('Heights'), sg.Button('Open Spaces'),sg.Button('Spiders'),sg.Button('Lightning'),sg.Button('Confined Spaces'),sg.Button('Clowns'),sg.Button('Dogs'),sg.Button('Skin defects'),sg.Button('Vomit'),sg.Button('Blood'),sg.Button('Water'),sg.Button('Birds'),sg.Button('Snakes'),sg.Button('Death'),sg.Button('Needles'),sg.Button('Holes'),sg.Button('None')] ]

# # Create the Window
# window = sg.Window('Select', layout)
# # Event Loop to process "events" and get the "values" of the inputs
# while True:
#     event, values = window.Read()
#     if event in (None, 'None'):	# if user closes window or clicks cancel
#         break
#     ApplicablePhobias += [event]

# window.close()
# print(ApplicablePhobias)

def AskUser(im,phobias):
    layout = [[sg.Text("The following potentially disturbing content has been detected in this image:")],[sg.Text(phobia) for phobia in phobias], [sg.Text("Would you like to view it anyway?")], [sg.Button('Yes'),sg.Button('No')] ]
    window = sg.Window('Select',layout)
    event, values = window.Read()
    if event == "Yes":
        plt.imshow(mpimg.imread(im))
        plt.show()
    else:
        pass

AskUser('data/AcrophobiaImages/worst-trust-exercise-ever-photo-u1.jpeg',['Heights','Spiders'])
