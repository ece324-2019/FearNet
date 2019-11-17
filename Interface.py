def startup():
    AllPhobias = ["Heights","Open Spaces","Spiders","Lightning","Loneliness","Cancer","Confined Spaces","Clowns","Dogs","Vomit","Blood","Water","Bacteria","Snakes","Birds","Death","Needle","Irregular patterns of holes"]
    ApplicablePhobias = []
    print("Select which of the following common phobias applies to you (answer yes/no):")
    for phobia in AllPhobias:
        while True:
            print(phobia,"?")
            answer = input()
            if answer.lower() == "yes":
                ApplicablePhobias += [phobia]
                break
            elif answer.lower() == "no":
                break
            else:
                print("We don't understand, please try again")
    return ApplicablePhobias

startup()

def AskUser(im,phobias):
    print("The following potentially disturbing content has been detected in this image:")
    for phobia in phobias:
        print(phobia)
    print("Would you like to view it anyway? Answer yes/no")
    answer = input()
    if answer.lower() == "yes":
        im.show()
    elif answer.lower() == "no":
        print("This image has been blocked as per your request")
    else:
        print("We couldn't understand your answer. Please try again.")
    