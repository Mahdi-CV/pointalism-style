from datasets import load_dataset
import matplotlib.pyplot as plt

dataset = load_dataset("Rapidata/2k-ranked-images-open-image-preferences-v1")
sample = dataset["train"][0]
img = sample["image"]  # this is a PIL image

plt.imshow(img)
plt.axis("off")
plt.show()