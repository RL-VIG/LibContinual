# -----------------
# Custom Transfrom : Define your own custom transform function, and add to the list

custom_trfm_names = ['_convert_to_rgb']

def _convert_to_rgb(img):
    return img.convert('RGB')

# -----------------