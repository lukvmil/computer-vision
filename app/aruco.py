import json

def load_markers(filename):
    with open(filename, 'r') as f:
        markers = json.load(f)

    marker_lookup_table = {}

    # converts integer representation of markers to binary grid for later use in image analysis
    for i, orientations in enumerate(markers):
        for marker in orientations:
            bin_repr1 = format(marker[0], '08b')
            bin_repr2 = format(marker[1], '08b')
            code = (bin_repr1[:4], bin_repr1[4:], bin_repr2[:4], bin_repr2[4:])
            marker_lookup_table[code] = i
    
    return marker_lookup_table


def read_marker(image, threshold, pixel_size, marker_size):
    marker_code = []

    for py in range(marker_size):
        sub_code = ''
        for px in range(marker_size):
            pr = int(px * pixel_size)
            pl = int(pr + pixel_size)
            pt = int(py * pixel_size)
            pb = int(pt + pixel_size)
            # print(pr, pt)
            pixel = image[pt:pb, pr:pl]
            # show_image(pixel, f"{px}:{py}")
            pixel_sum = pixel.sum()
            pixel_avg = pixel_sum / pixel_size ** 2
            pixel_value = int(pixel_avg) > threshold

            if (px != 0) and (py != 0) and (px != marker_size - 1) and (py != marker_size - 1):
                sub_code += '1' if pixel_value else '0'

        #     print('X' if pixel_value else ' ', end="")
        # print()
        if sub_code:
            marker_code.append(sub_code)

    return marker_code