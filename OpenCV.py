# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import glob, os

# --- Load sample images ---
# Folder structure:
# data/
#   knit/
#       knit1.jpg, knit2.jpg, ...
#   purl/
#       purl1.jpg, purl2.jpg, ...

def load_images(folder):
    images, labels = [], []
    for label in ['knit', 'purl']:
        for path in glob.glob(os.path.join(folder, label,'*.jpg')):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

X, y = load_images('data')

# --- Extract texture features (using Local Binary Patterns) ---
def lbp_features(img):
    radius = 1
    n_points = 8 * radius
    lbp = cv2.cvtColor(cv2.merge([img, img, img]), cv2.COLOR_BGR2GRAY)
    lbp = cv2.Laplacian(lbp, cv2.CV_64F)  # simple texture cue
    hist = np.histogram(lbp.ravel(), bins=16, range=(0, 256))[0]
    return hist / np.sum(hist)

X_features = np.array([lbp_features(img) for img in X])

# --- Train a simple SVM ---
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
clf = LinearSVC()
clf.fit(X_train, y_train)

# --- Evaluate ---
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# --- Test on a new image ---
test_img = cv2.imread('data/test/test_pattern.jpg', cv2.IMREAD_GRAYSCALE)
test_img = cv2.resize(test_img, (128, 128))
feat = lbp_features(test_img).reshape(1, -1)
prediction = clf.predict(feat)
print(f"Predicted stitch type: {prediction[0]}")

plt.imshow(test_img, cmap='gray')
plt.title(f"Predicted: {prediction[0]}")
plt.axis('off')
plt.show()
