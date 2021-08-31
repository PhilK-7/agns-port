import demo_face_recognition
import demo_generate_eyeglasses
import time

if __name__ == '__main__':
    print('AGNs')
    print('Which of the demos do you want to launch?')

    print('> Enter \'1\' to test the face recognition models.')
    print('> Enter \'2\' to test the glasses generator.')
    # TODO add rest

    accepted_inputs = (1, 2)

    # select a demo
    demo_index = -1
    while demo_index not in accepted_inputs:
        inp = input('Which demo to start? Please enter the number below: \n')
        try:
            demo_index = int(inp)
            continue
        except ValueError:
            print('Please enter one of the numbers stated above.')

    # launch a demo
    print(100 * '=' + '\n\n')
    time.sleep(1.5)
    if demo_index == 1:
        print('Starting demo_face_recognition.')
        demo_face_recognition.main((1,))
    if demo_index == 2:
        print('Starting demo_generate_eyeglasses.')
        demo_generate_eyeglasses.main((1,))
