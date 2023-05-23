import numpy as np
import streamlit as st

from image_processing import image_processing
from models import unet

st.set_page_config(layout="wide")
st.title("Wildfire Detection from Satellite Images")
tab1, tab2, tab3 = st.tabs(["Predictions", "Gallery", "Info"])


def predictions_tab(tab_):
    uploaded_file_ = tab_.file_uploader("Choose an image...")
    if uploaded_file_ is not None:
        models = {"U-Net": "./u-net-10-epochs-22052023.h5",
                  "SegNet": './u-net-10-epochs-22052023.h5',
                  "DeconvNet": "./u-net-10-epochs-22052023.h5"}

        model_name = tab_.selectbox("Choose a model:", list(models.keys()))
        model_path_ = models[model_name]
        print(f'model_name: {model_path_}')

        is_model_defined_status = True

        if "u-net" in model_path_:
            model_ = unet()  # todo train unet
        elif "segnet" in model_path_:
            model_ = unet()  # todo train segnet
        elif "deconvnet" in model_path_:
            model_ = unet()  # todo train deconvnet
        else:
            is_model_defined_status = False

        if not is_model_defined_status:
            pass

        model_.load_weights(model_path_)

        ground_data, bands_msg_ = image_processing(uploaded_file_)
        tab_.write(bands_msg_)

        prediction = model_.predict(np.array([ground_data]))

        images = {"Ground Image": ground_data, "Segmentation result": prediction}
        cols = tab_.columns(len(images))

        for i, (title, image) in enumerate(images.items()):
            cols[i].markdown(f'## {title}')
            cols[i].image(image, use_column_width=True)


predictions_tab(tab1)


def gallery_tab(tab_):
    visualisations = ["Images Gallery", "Metrics Values", "Metrics history Plots"]

    sub_tab_ = tab_.selectbox("Choose visualisation:", list(visualisations))

    if sub_tab_ == visualisations[0]:
        pass  # todo image gallery
    elif sub_tab_ == visualisations[1]:
        pass  # todo metrics values
    else:
        pass  # todo history metrics plots


gallery_tab(tab2)


def info_tab(tab_):
    with tab_.expander("Вступ"):
        st.write("""The chart above shows some numbers I picked for you.
        I rolled actual dice for these, so they're *guaranteed* to
        be random.""")


info_tab(tab3)
