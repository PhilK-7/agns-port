import time
from demos import demo_dodging, demo_impersonation, demo_impersonation_real, demo_face_recognition, \
    demo_generate_eyeglasses

# >>> specify GPU(s) to use <<<
gpus = (1,)

if __name__ == '__main__':
    print('AGNs Demo Main')
    print('Which of the demos do you want to launch?')

    print('> Enter \'1\' to test the face recognition models.')
    print('> Enter \'2\' to test the glasses generator.')
    print('> Enter \'3\' to test the dodging attack.')
    print('> Enter \'4\' to test the impersonation attack.')
    print('> Enter \'5\' to test the physical impersonation attack.')

    accepted_inputs = (1, 2, 3, 4, 5)

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
    time.sleep(1.0)
    if demo_index == 1:
        print('Starting demo_face_recognition.')
        time.sleep(1.5)
        demo_face_recognition.main(gpus)
    if demo_index == 2:
        print('Starting demo_generate_eyeglasses.')
        time.sleep(1.5)
        demo_generate_eyeglasses.main(gpus)
    if demo_index == 3:
        print('Starting demo_dodging.')
        time.sleep(1.5)
        demo_dodging.main(gpus)
    if demo_index == 4:
        print('Starting demo_impersonation.')
        time.sleep(1.5)
        demo_impersonation.main(gpus, True)
    if demo_index == 5:
        print('Starting demo_impersonation_real.')
        time.sleep(1.5)
        demo_impersonation_real.main(gpus)

