import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pickle

plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170

# Different variance
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)
print(y_varied)
# y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)
kmean = KMeans(n_clusters=3, random_state=random_state)
kmean.fit_predict(X_varied)



y_pred_2 = kmean.predict(X_varied)

plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred_2)
# plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_variqed)
plt.title("Unequal Variance")

plt.show()

# # save the model to disk
# filename = 'finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))
#
# # some time later...
#
# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))