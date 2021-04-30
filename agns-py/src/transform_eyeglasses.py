

def transform_eyeglasses(eyeglass_im, eyeglass_area, tform, out_size=-1):
    """

    :param eyeglass_im:
    :param eyeglass_area:
    :param tform:
    :param out_size:
    :return:
    """
    logical_area = False
    if isinstance(eyeglass_area, bool):
        eyeglass_area = int(eyeglass_area * 255)
        logical_area = True

    if out_size == -1:
        pass
    else:
        pass

    if logical_area:
        eyeglass_area = eyeglass_area > 200

    return eyeglass_im, eyeglass_area
