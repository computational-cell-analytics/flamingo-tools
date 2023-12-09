import z5py


file_path = "/home/pape/Work/data/moser/lightsheet/new/converted/S000_t000000_V000_X000_Y000_I0_D0_P02995.n5"
# same inferface as h5py
with z5py.File(file_path, "r") as n5_file:
    dataset = n5_file["setup0/timepoint0/s1"]
    print(dataset.shape)
    print(dataset.chunks)

    # load it into memory
    sub_volume = dataset[:]
    print(sub_volume.shape)
    print(type(sub_volume))


import napari
v = napari.Viewer()
v.add_image(sub_volume)
napari.run()
