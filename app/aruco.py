import json
import cv2

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
            mirror_code = (bin_repr1[:4][::-1], bin_repr1[4:][::-1], bin_repr2[:4][::-1], bin_repr2[4:][::-1])
            marker_lookup_table[mirror_code] = i
            print(code)
            print(mirror_code)
    
    return marker_lookup_table


def read_marker(image, threshold, pixel_size, marker_size, offset):
    marker_code = []

    for py in range(marker_size):
        sub_code = ''
        for px in range(marker_size):
            pl = int(px * pixel_size) 
            pr = int(pl + pixel_size)
            pt = int(py * pixel_size)
            pb = int(pt + pixel_size)
            print()
            print(f"{pt}:{pb}, {pl}:{pr}")
            print(f"{pt+offset}:{pb-offset}, {pl+offset}:{pr-offset}")
            pixel = image[pt:pb, pl:pr]
            pixel = pixel[offset:-offset, offset:-offset]
            pixel_sum = pixel.sum()
            pixel_avg = int(pixel_sum / (pixel_size  - 2 * offset) ** 2)
            print(pixel_size, offset, pixel_size - 2 * offset)
            print(pixel_sum, pixel_avg)

            pixel_value = int(pixel_avg) > threshold

            font_color = (0,0,0) if pixel_value else (255, 255, 255)
            cv2.putText(image, str(pixel_avg), (pl, pb), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1, cv2.LINE_AA)


            if (px != 0) and (py != 0) and (px != marker_size - 1) and (py != marker_size - 1):
                sub_code += '1' if pixel_value else '0'

        #     print('X' if pixel_value else ' ', end="")
        # print()
        if sub_code:
            marker_code.append(sub_code)

    return marker_code