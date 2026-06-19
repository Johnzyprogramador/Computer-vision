from firevision.yolo import class_names, images_from_source, infer_label_path, labels_for


def test_yolo_names_list_and_mapping():
    assert class_names({"names": ["smoke", "fire"]}) == {0: "smoke", 1: "fire"}
    assert class_names({"names": {0: "flame", 2: "smoke"}}) == {0: "flame", 2: "smoke"}


def test_image_and_label_discovery(tmp_path):
    image = tmp_path / "images" / "train" / "sample.jpg"
    label = tmp_path / "labels" / "train" / "sample.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    image.touch()
    label.write_text("4 0.5 0.5 0.2 0.2\n7 0.4 0.4 0.1 0.1\n")
    assert images_from_source(image.parent) == [image]
    assert infer_label_path(image) == label
    assert labels_for(label, {4: "smoke", 7: "flame"}) == (1, 1)


def test_text_file_image_source(tmp_path):
    image = tmp_path / "frames" / "one.png"
    image.parent.mkdir()
    image.touch()
    listing = tmp_path / "test.txt"
    listing.write_text("frames/one.png\n")
    assert images_from_source(listing) == [image.resolve()]
