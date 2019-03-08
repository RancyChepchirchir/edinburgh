###
# plot some of the data in 2d (with PCA and MDS)
# mainly copied from lab 3
###
# INPUT:
#	fulldata from join_data.py
#	keys_content_features: array(String) of column keys
# PRINTS:
#	2d graphs using PCA and MDS
###

from scripts import *

(X_content_full, keys_content_features) = read_transform_contentfeatures()
(y_full,keys_labels) = read_transform_labels()
fulldata = join_data(X_content_full, y_full)

X = fulldata[keys_content_features]
y = fulldata['Neutrality']

### From Lab 3, edited
def scatter_2d_label(X_2d, y, alpha=0.5):
    """Visualuse a 2D embedding with corresponding labels.
    
    X_2d : ndarray, shape (n_samples,2)
        Low-dimensional feature representation.
    
    y : ndarray, shape (n_samples,)
        Labels corresponding to the entries in X_2d.
        
    s : float
        Marker size for scatter plot.
    
    alpha : float
        Transparency for scatter plot.
        
    lw : float
        Linewidth for scatter plot.
    """
    targets = np.unique(y)
    
    colors = [plt.cm.RdYlGn( int(i*plt.cm.RdYlGn.N/(len(targets)-1)) ) for i in range(len(targets))]
    for color, target in zip(colors, targets):
        plotx = [x for i,x in enumerate(X_2d[:,0]) if (y[i]==target)]
        ploty = [x for i,x in enumerate(X_2d[:,1]) if (y[i]==target)]
        plt.scatter(plotx, ploty, color=color, label=target, alpha=alpha)
    plt.legend()
    plt.show()
	
from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler().fit_transform(X.astype(np.float))


### PCA

from sklearn.decomposition import PCA

plt.figure(figsize=(10,6))
pca = PCA( n_components=2 )
X_2d = pca.fit_transform(X_sc)
scatter_2d_label(X_2d, y)
plt.show()

### MDS

from sklearn.manifold import MDS
sns.set(font_scale=1.)
mds = MDS(n_components=2, metric=True, n_init=1, max_iter=100, random_state=10)
X_mds_2d = mds.fit_transform(X_sc)
plt.title('Metric MDS, stress: {}'.format(mds.stress_))
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend(loc='center left', bbox_to_anchor=[1.01, 0.5], scatterpoints=3)
scatter_2d_label(X_mds_2d, y)
plt.show()